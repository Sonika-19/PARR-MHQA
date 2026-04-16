import pickle
import time
from dataclasses import asdict

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from config import Config, setup_logging, setup_reproducibility
from evaluation.baselines import run_full_comparison
from main import PARRMHQASystem


st.set_page_config(page_title="PARR-MHQA", layout="wide")


@st.cache_resource
def load_system_cached() -> PARRMHQASystem:
	cfg = Config()
	setup_reproducibility()
	setup_logging(cfg.LOG_DIR)
	system = PARRMHQASystem(cfg)
	system.initialize()
	return system


def _init_session_state():
	if "system" not in st.session_state:
		st.session_state.system = None
	if "initialized" not in st.session_state:
		st.session_state.initialized = False
	if "last_result" not in st.session_state:
		st.session_state.last_result = None
	if "eval_output" not in st.session_state:
		st.session_state.eval_output = None


def _status_dot(ok: bool) -> str:
	return "<span style='color:#16a34a'>●</span>" if ok else "<span style='color:#dc2626'>●</span>"


def _render_system_status(system):
	st.markdown("### System Status")
	names = [
		("EmbeddingEngine", getattr(system, "embedding_engine", None)),
		("DocumentStore", getattr(system, "document_store", None)),
		("FAISSIndex", getattr(system, "faiss_index", None)),
		("LLMGenerator", getattr(system, "generator", None)),
		("HallucinationDetector", getattr(system, "hallucination_detector", None)),
		("SelfCritiqueModule", getattr(system, "critique_module", None)),
		("RefinementController", getattr(system, "refinement_controller", None)),
	]
	for name, component in names:
		st.markdown(f"{_status_dot(component is not None)} {name}", unsafe_allow_html=True)


def _hall_color(score: float) -> str:
	if score < 0.35:
		return "#16a34a"
	if score <= 0.6:
		return "#d97706"
	return "#dc2626"


def _metric_card(label: str, value: str, color: str = "#1f2937"):
	st.markdown(
		(
			"<div style='padding:14px;border:1px solid #e5e7eb;border-radius:10px;'>"
			f"<div style='font-size:0.85rem;color:#6b7280'>{label}</div>"
			f"<div style='font-size:1.3rem;color:{color};font-weight:700'>{value}</div>"
			"</div>"
		),
		unsafe_allow_html=True,
	)


def _timing_pie(result):
	labels = ["retrieval", "reranking", "generation", "hallucination"]
	vals = [
		float(result.time_retrieval),
		float(result.time_reranking),
		float(result.time_generation),
		float(result.time_hallucination),
	]
	fig, ax = plt.subplots(figsize=(5, 4))
	if sum(vals) <= 0:
		ax.text(0.5, 0.5, "No timing data", ha="center", va="center")
	else:
		ax.pie(vals, labels=labels, autopct="%1.1f%%", startangle=90)
	ax.set_title("Timing Breakdown")
	st.pyplot(fig)


def _load_qa_pairs():
	candidates = ["embeddings/processed_qa_pairs.pkl", "embeddings/qa_pairs.pkl"]
	for path in candidates:
		try:
			with open(path, "rb") as file:
				return pickle.load(file)
		except FileNotFoundError:
			continue
	raise FileNotFoundError("QA pairs not found. Run data preprocessing first.")


_init_session_state()
st.title("PARR-MHQA")

with st.sidebar:
	st.header("Controls")
	threshold = st.slider("hallucination_threshold", 0.1, 0.8, 0.35, 0.01)
	max_iters = st.slider("max_iterations", 0, 3, 2, 1)

	if st.button("Initialize System", use_container_width=True):
		try:
			with st.spinner("Initializing all components..."):
				t0 = time.perf_counter()
				system = load_system_cached()
				elapsed = time.perf_counter() - t0

				# Apply UI overrides.
				object.__setattr__(system.config, "HALLUCINATION_THRESHOLD", float(threshold))
				object.__setattr__(system.config, "MAX_REFINE_ITERS", int(max_iters))

				st.session_state.system = system
				st.session_state.initialized = True
			st.success(f"System initialized in {elapsed:.2f}s")
		except Exception as exc:
			st.error(f"Initialization failed: {exc}")

	if st.session_state.system is not None:
		_render_system_status(st.session_state.system)


tab_query, tab_eval = st.tabs(["Query", "Evaluation"])

with tab_query:
	st.subheader("Ask a Question")
	query_text = st.text_area("Query", height=140, placeholder="Type your multi-hop question...")
	system_ready = st.session_state.get("system") is not None and st.session_state.get("initialized", False)
	if not system_ready:
		st.warning("Initialize system first")

	if st.button("Get Answer", type="primary", disabled=not system_ready):
		if not query_text.strip():
			st.error("Please enter a query.")
		else:
			try:
				with st.spinner("Running retrieval, reasoning, and refinement..."):
					system = st.session_state.system
					object.__setattr__(system.config, "HALLUCINATION_THRESHOLD", float(threshold))
					object.__setattr__(system.config, "MAX_REFINE_ITERS", int(max_iters))
					result = system.answer(query_text.strip())
					st.session_state.last_result = result

				st.info(result.final_answer)

				c1, c2, c3, c4 = st.columns(4)
				with c1:
					_metric_card(
						"Hallucination Score",
						f"{result.final_hallucination_score:.3f}",
						_hall_color(float(result.final_hallucination_score)),
					)
				with c2:
					_metric_card("Iterations", str(result.iterations_used))
				with c3:
					_metric_card("Query Type", result.policy_decision.query_type)
				with c4:
					_metric_card("LLM Calls", str(result.num_llm_calls))

				with st.expander("Retrieved documents", expanded=False):
					for doc in result.retrieved_docs[:3]:
						indicator = "HIGH" if float(doc.score) >= 0.66 else ("MED" if float(doc.score) >= 0.33 else "LOW")
						st.markdown(f"**{doc.title}** | rank={doc.rank} | score={doc.score:.4f} | fact_coverage={indicator}")
						st.write(doc.text[:350] + ("..." if len(doc.text) > 350 else ""))

				with st.expander("Policy decision", expanded=False):
					st.json(asdict(result.policy_decision))

				with st.expander("Critique", expanded=False):
					st.write(f"Support level: {result.critique_result.support_level}")
					if result.critique_result.unsupported_parts:
						st.write("Unsupported parts:")
						for item in result.critique_result.unsupported_parts:
							st.write(f"- {item}")
					else:
						st.write("No unsupported parts listed.")

				with st.expander("Refinement history", expanded=False):
					hist_df = pd.DataFrame(
						{
							"iteration": list(range(len(result.all_answers))),
							"answer": result.all_answers,
							"hallucination_score": result.hallucination_scores,
						}
					)
					st.dataframe(hist_df, use_container_width=True)

				with st.expander("Timing breakdown", expanded=False):
					_timing_pie(result)

			except Exception as exc:
				st.error(f"Query failed: {exc}")


with tab_eval:
	st.subheader("Evaluation")
	n_samples = st.slider("n_samples", 10, 200, 50, 10)

	if st.button("Run Evaluation"):
		if not st.session_state.initialized or st.session_state.system is None:
			st.error("Initialize the system first.")
		else:
			progress = st.progress(0)
			try:
				with st.spinner("Running evaluation suite..."):
					qa_pairs = _load_qa_pairs()
					progress.progress(20)

					comparison = run_full_comparison(st.session_state.system, qa_pairs, n=n_samples)
					progress.progress(75)

					eval_cards = comparison.get("parr_mhqa", {})
					st.session_state.eval_output = comparison
					progress.progress(100)

				m1, m2, m3, m4 = st.columns(4)
				with m1:
					_metric_card("EM", f"{float(eval_cards.get('em', 0.0)):.3f}")
				with m2:
					_metric_card("F1", f"{float(eval_cards.get('f1', 0.0)):.3f}")
				with m3:
					_metric_card("Hallucination", f"{float(eval_cards.get('avg_hallucination', 0.0)):.3f}")
				with m4:
					_metric_card("ECE", f"{float(eval_cards.get('ece', 0.0)):.3f}")

				csv_df = pd.DataFrame(
					[
						{"system": "llm_only", **comparison.get("llm_only", {})},
						{"system": "standard_rag", **comparison.get("standard_rag", {})},
						{"system": "parr_mhqa", **comparison.get("parr_mhqa", {})},
					]
				)
				csv_bytes = csv_df.to_csv(index=False).encode("utf-8")
				st.download_button(
					"Download results CSV",
					data=csv_bytes,
					file_name="evaluation_results.csv",
					mime="text/csv",
				)

			except Exception as exc:
				st.error(f"Evaluation failed: {exc}")
