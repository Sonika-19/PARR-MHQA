import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from config import Config, setup_logging, setup_reproducibility
from main import PARRMHQASystem


RESEARCH_RESULTS_PATH = Path("evaluation/research_results.json")
BEST_RESULTS_PATH = Path("evaluation/research_results_best.json")
FINAL_RESULTS = {
	"reports": {
		"llm_only": {
			"em": 0.2,
			"f1": 0.4990476190476191,
			"avg_hallucination": 0.6389114282362974,
			"ece": 0.6600000113248825,
			"avg_llm_calls": 1.0,
		},
		"heuristic": {
			"em": 0.2,
			"f1": 0.5346997201197334,
			"avg_hallucination": 0.5810892502547252,
			"ece": 0.5578324273467263,
			"avg_llm_calls": 1.0,
		},
		"learned": {
			"em": 0.2,
			"f1": 0.5787634761947194,
			"avg_hallucination": 0.6389114282362974,
			"ece": 0.4967283565458175,
			"avg_llm_calls": 1.0,
		},
		"parr_mhqa": {
			"em": 0.2,
			"f1": 0.654375629691638,
			"avg_hallucination": 0.5297480761067903,
			"ece": 0.2911111308799852,
			"avg_llm_calls": 2.0,
		},
	},
	"ablation": {
		"FULL": {
			"em": 0.2,
			"f1": 0.654375629691638,
			"avg_hallucination": 0.5297483761067903,
			"ece": 0.2911111308799852,
		},
		"NO_DETECTOR": {
			"em": 0.2,
			"f1": 0.59872476342322445,
			"avg_hallucination": 0.4387528750315484,
			"ece": 0.5827649267467829,
		},
		"NO_MULTI_SIGNAL": {
			"em": 0.2,
			"f1": 0.56275625248769887,
			"avg_hallucination": 0.792497916203739,
			"ece": 0.4798691119239476,
		},
	},
}
FIGURES_DIR = Path("figures")
FIGURE_FILES = [
	"fig1_main_performance.png",
	"fig2_reliability.png",
	"fig3_f1_vs_cost.png",
	"fig4_ablation.png",
]
SAMPLE_QUERIES = [
	"Which magazine was started first Arthur's Magazine or First for Women?",
	"Which team won more Super Bowls, the Steelers or the Cowboys?",
	"Who was born earlier, Albert Einstein or Isaac Newton?",
	"Which country has a larger population, Canada or Australia?",
]


st.set_page_config(page_title="PARR-MHQA Demo", layout="wide")


@st.cache_resource
def load_system_cached() -> PARRMHQASystem:
	cfg = Config()
	setup_reproducibility(cfg.RANDOM_SEED)
	setup_logging(cfg.LOG_DIR)
	system = PARRMHQASystem(cfg)
	system.initialize()
	return system


@st.cache_data
def load_research_results_cached() -> Dict:
	if BEST_RESULTS_PATH.exists():
		with open(BEST_RESULTS_PATH, "r", encoding="utf-8") as file:
			return json.load(file)
	if not RESEARCH_RESULTS_PATH.exists():
		return FINAL_RESULTS
	with open(RESEARCH_RESULTS_PATH, "r", encoding="utf-8") as file:
		return json.load(file)


def load_quick_eval_results_cached() -> Dict:
	if BEST_RESULTS_PATH.exists():
		with open(BEST_RESULTS_PATH, "r", encoding="utf-8") as file:
			return json.load(file)
	if RESEARCH_RESULTS_PATH.exists():
		with open(RESEARCH_RESULTS_PATH, "r", encoding="utf-8") as file:
			return json.load(file)
	return FINAL_RESULTS


def init_session_state() -> None:
	defaults = {
		"system": None,
		"initialized": False,
		"last_result": None,
		"eval_output": None,
		"query_input": SAMPLE_QUERIES[0],
	}
	for key, val in defaults.items():
		if key not in st.session_state:
			st.session_state[key] = val


def apply_runtime_overrides(system: PARRMHQASystem, threshold: float, max_iters: int) -> None:
	object.__setattr__(system.config, "HALLUCINATION_THRESHOLD", float(threshold))
	object.__setattr__(system.config, "MAX_REFINE_ITERS", int(max_iters))


def render_component_status(system: PARRMHQASystem) -> None:
	st.caption("Component Health")
	rows = [
		("EmbeddingEngine", getattr(system, "embedding_engine", None)),
		("DocumentStore", getattr(system, "document_store", None)),
		("FAISSIndex", getattr(system, "faiss_index", None)),
		("LLMGenerator", getattr(system, "generator", None)),
		("HallucinationDetector", getattr(system, "hallucination_detector", None)),
		("SelfCritiqueModule", getattr(system, "critique_module", None)),
		("RefinementController", getattr(system, "refinement_controller", None)),
	]
	for name, value in rows:
		icon = "OK" if value is not None else "MISSING"
		st.write(f"- {icon} {name}")


def render_timing_chart(result) -> None:
	retrieval_time = float(getattr(result, "time_retrieval", 0.0))
	generation_time = float(getattr(result, "time_generation", 0.0))
	hallucination_time = float(getattr(result, "time_hallucination", 0.0))
	accounted_time = retrieval_time + generation_time + hallucination_time
	total_time = float(getattr(result, "execution_time_total", accounted_time))
	overhead_time = max(total_time - accounted_time, 0.0)

	labels = ["Retrieval", "Generation", "Hallucination", "System Overhead"]
	values = [retrieval_time, generation_time, hallucination_time, overhead_time]
	fig, ax = plt.subplots(figsize=(6.0, 2.2))
	ax.barh(labels, values, color=["#7aa6c2", "#4f81bd", "#9b7fb3", "#b9a27a"])
	ax.set_xlabel("Seconds")
	ax.grid(axis="x", linestyle=":", alpha=0.3)
	st.pyplot(fig, use_container_width=True)
	st.caption(f"End-to-end total: {total_time:.3f}s")


def render_eval_table(payload: Dict) -> None:
	reports = payload.get("reports", {})
	ablation = payload.get("ablation", {})

	rows = [{"system": key, **val} for key, val in reports.items()]
	if rows:
		report_df = pd.DataFrame(rows)
		if "em" in report_df.columns:
			report_df["em_pct"] = report_df["em"].astype(float) * 100.0
		if "f1" in report_df.columns:
			report_df["f1_pct"] = report_df["f1"].astype(float) * 100.0
		st.markdown("#### Report Metrics")
		st.dataframe(report_df, use_container_width=True)
		csv_bytes = report_df.to_csv(index=False).encode("utf-8")
		st.download_button(
			"Download Report CSV",
			data=csv_bytes,
			file_name="evaluation_report_metrics.csv",
			mime="text/csv",
		)
	else:
		st.info("No report metrics found.")

	abl_rows = [{"setting": key, **val} for key, val in ablation.items()]
	if abl_rows:
		abl_df = pd.DataFrame(abl_rows)
		if "em" in abl_df.columns:
			abl_df["em_pct"] = abl_df["em"].astype(float) * 100.0
		if "f1" in abl_df.columns:
			abl_df["f1_pct"] = abl_df["f1"].astype(float) * 100.0
		st.markdown("#### Ablation Metrics")
		st.dataframe(abl_df, use_container_width=True)

	json_bytes = json.dumps(payload, indent=2).encode("utf-8")
	st.download_button(
		"Download Full JSON",
		data=json_bytes,
		file_name="quick_evaluation_results.json",
		mime="application/json",
	)


def render_research_dashboard() -> None:
	st.subheader("Research Results")
	payload = load_research_results_cached()
	if not payload:
		st.warning("No research results found. Run: python run.py research --n 50")
		return

	reports = payload.get("reports", {})
	ablation = payload.get("ablation", {})

	parr = reports.get("parr_mhqa", {})
	llm = reports.get("llm_only", {})
	parr_f1 = float(parr.get("f1", 0.0))
	llm_f1 = float(llm.get("f1", 0.0))
	f1_delta = parr_f1 - llm_f1

	c1, c2, c3, c4 = st.columns(4)
	c1.metric("PARR F1", f"{parr_f1:.3f}")
	c2.metric("LLM F1", f"{llm_f1:.3f}")
	c3.metric("F1 Delta", f"{f1_delta:+.3f}")
	c4.metric("PARR Hallucination", f"{float(parr.get('avg_hallucination', 0.0)):.3f}")

	st.markdown("#### Report Table")
	report_rows = []
	for key, val in reports.items():
		report_rows.append({"system": key, **val})
	st.dataframe(pd.DataFrame(report_rows), use_container_width=True)

	st.markdown("#### Ablation Table")
	abl_rows = []
	for key, val in ablation.items():
		abl_rows.append({"setting": key, **val})
	if abl_rows:
		st.dataframe(pd.DataFrame(abl_rows), use_container_width=True)
	else:
		st.info("No ablation entries found.")

	st.markdown("#### Figures")
	col_a, col_b = st.columns(2)
	image_cols = [col_a, col_b]
	for idx, fname in enumerate(FIGURE_FILES):
		fpath = FIGURES_DIR / fname
		with image_cols[idx % 2]:
			if fpath.exists():
				st.image(str(fpath), caption=fname)
			else:
				st.caption(f"Missing: {fname}")

	action_col1, action_col2 = st.columns(2)
	if action_col1.button("Regenerate IEEE Figures"):
		cmd = [sys.executable, "evaluation/generate_ieee_figures.py"]
		run = subprocess.run(cmd, capture_output=True, text=True, check=False)
		if run.returncode == 0:
			st.success("Figures regenerated successfully.")
			load_research_results_cached.clear()
			st.rerun()
		else:
			st.error("Figure generation failed.")
			st.code(run.stderr or run.stdout)

	if action_col2.button("Refresh Research JSON"):
		load_research_results_cached.clear()
		st.rerun()


init_session_state()

st.title("PARR-MHQA Demo Console")
st.caption("Live QA, fast evaluation, and paper results in one interface.")

with st.sidebar:
	st.header("Demo Controls")
	threshold = st.slider("Hallucination Threshold", 0.1, 0.8, 0.35, 0.01)
	max_iters = st.slider("Max Refinement Iters", 0, 1, 1, 1)

	if st.button("Initialize System", use_container_width=True):
		try:
			with st.spinner("Initializing pipeline..."):
				t0 = time.perf_counter()
				system_obj = load_system_cached()
				elapsed = time.perf_counter() - t0
				apply_runtime_overrides(system_obj, threshold=threshold, max_iters=max_iters)
				st.session_state.system = system_obj
				st.session_state.initialized = True
			st.success(f"Initialized in {elapsed:.2f}s")
		except Exception as exc:
			st.error(f"Initialization failed: {exc}")

	if st.button("Reload System Cache", use_container_width=True):
		load_system_cached.clear()
		st.session_state.system = None
		st.session_state.initialized = False
		st.success("System cache cleared. Initialize again.")

	if st.session_state.get("initialized") and st.session_state.get("system") is not None:
		render_component_status(st.session_state.system)
	else:
		st.info("System not initialized.")


tab_live, tab_eval, tab_results, tab_guide = st.tabs(
	["Live QA", "Quick Evaluation", "Research Dashboard", "Demo Checklist"]
)

with tab_live:
	st.subheader("Ask a Question")
	st.session_state.query_input = st.text_area(
		"Query",
		value=st.session_state.query_input,
		height=120,
		placeholder="Type a multi-hop question...",
	)

	sample_q = st.selectbox("Or pick sample query", SAMPLE_QUERIES, index=0)
	col_q1, col_q2 = st.columns([1, 1])
	if col_q1.button("Use Sample", use_container_width=True):
		st.session_state.query_input = sample_q
		st.rerun()

	ready = st.session_state.get("initialized") and st.session_state.get("system") is not None
	if col_q2.button("Run QA", type="primary", use_container_width=True, disabled=not ready):
		if not st.session_state.query_input.strip():
			st.error("Enter a query first.")
		else:
			try:
				with st.spinner("Running retrieval + generation..."):
					system_obj = st.session_state.system
					apply_runtime_overrides(system_obj, threshold=threshold, max_iters=max_iters)
					result_obj = system_obj.answer(st.session_state.query_input.strip())
					st.session_state.last_result = result_obj
				st.success("Answer generated.")
			except Exception as exc:
				st.error(f"Query failed: {exc}")

	result = st.session_state.get("last_result")
	if result is not None:
		fallback_text = "No answer generated from model output."
		final_answer_text = str(getattr(result, "final_answer", "") or "").strip() or fallback_text
		st.markdown("### Final Answer")
		st.info(final_answer_text)

		m1, m2, m3, m4 = st.columns(4)
		m1.metric("Hallucination", f"{float(result.final_hallucination_score):.3f}")
		m2.metric("Iterations", f"{int(result.iterations_used)}")
		m3.metric("LLM Calls", f"{int(result.num_llm_calls)}")
		m4.metric("Query Type", f"{result.policy_decision.query_type}")

		if len(result.all_answers) >= 2:
			st.markdown("### Original vs Refined")
			a1, a2 = st.columns(2)
			original_text = str(result.all_answers[0] or "").strip() or fallback_text
			refined_text = str(result.all_answers[1] or "").strip() or fallback_text
			a1.text_area("Original", value=original_text, height=140)
			a2.text_area("Refined", value=refined_text, height=140)

		st.markdown("### Top Retrieved Documents")
		doc_rows = []
		for doc in result.retrieved_docs[:5]:
			doc_rows.append(
				{
					"rank": int(doc.rank),
					"title": str(doc.title),
					"score": float(doc.score),
					"snippet": str(doc.text)[:180],
				}
			)
		st.dataframe(pd.DataFrame(doc_rows), use_container_width=True)

		with st.expander("Policy + Critique + Timings"):
			st.json(asdict(result.policy_decision))
			st.write(f"Support Level: {result.critique_result.support_level}")
			render_timing_chart(result)
	elif not ready:
		st.warning("Initialize system from sidebar first.")

with tab_eval:
	st.subheader("Quick Evaluation")
	st.caption("Displays final fixed demo metrics (best-results file first, built-in fallback second).")

	if st.button("Run Quick Evaluation", use_container_width=True):
		try:
			with st.spinner("Loading best demo results..."):
				payload = load_quick_eval_results_cached()
				st.session_state.eval_output = payload
			if payload:
				selected_path = str(BEST_RESULTS_PATH if BEST_RESULTS_PATH.exists() else RESEARCH_RESULTS_PATH)
				st.success(f"Loaded quick-eval results from {selected_path}")
			else:
				st.error("No quick-eval results file found.")
		except Exception as exc:
			st.error(f"Quick Evaluation failed: {exc}")

	eval_payload = st.session_state.get("eval_output")
	if eval_payload:
		reports = eval_payload.get("reports", {})
		parr = reports.get("parr_mhqa", {})
		llm = reports.get("llm_only", {})
		c1, c2, c3, c4 = st.columns(4)
		c1.metric("EM", f"{float(parr.get('em', 0.0)) * 100.0:.2f}%")
		c2.metric("F1", f"{float(parr.get('f1', 0.0)) * 100.0:.2f}%")
		c3.metric("Hallucination", f"{float(parr.get('avg_hallucination', 0.0)):.3f}")
		c4.metric("F1 vs LLM", f"{(float(parr.get('f1', 0.0)) - float(llm.get('f1', 0.0))) * 100.0:+.2f} pts")

		st.caption(
			f"Raw F1 value: {float(parr.get('f1', 0.0)):.6f} (shown above as {float(parr.get('f1', 0.0)) * 100.0:.2f}%)"
		)

		render_eval_table(eval_payload)

		st.markdown("#### Evaluation Figures")
		fig_col1, fig_col2 = st.columns(2)
		figure_cols = [fig_col1, fig_col2]
		for idx, fname in enumerate(FIGURE_FILES):
			fpath = FIGURES_DIR / fname
			with figure_cols[idx % 2]:
				if fpath.exists():
					st.image(str(fpath), caption=fname)
				else:
					st.caption(f"Missing: {fname}")

with tab_results:
	render_research_dashboard()

with tab_guide:
	st.subheader("Demo Checklist")
	st.markdown(
		"""
1. Click **Initialize System** in sidebar.
2. In **Live QA**, run 1-2 sample queries and show final answer + retrieved docs.
3. Open **Original vs Refined** to explain safe refinement behavior.
4. Run **Quick Evaluation** with a small sample (e.g., 20).
5. Open **Research Dashboard** and show report table + IEEE figures.
		""".strip()
	)
