#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

STEP1="[ ]"
STEP2="[ ]"
STEP3="[ ]"
STEP4="[ ]"

echo "[1/4] Creating virtual environment and installing dependencies..."
python -m venv .venv

# Activate venv (cross-platform)
if [[ "${OS:-}" == "Windows_NT" ]]; then
  source .venv/Scripts/activate
else
  source .venv/bin/activate
fi

python -m pip install --upgrade pip
pip install -r requirements.txt
STEP1="[x]"

echo "[2/4] Downloading spaCy model en_core_web_sm..."
python -m spacy download en_core_web_sm
STEP2="[x]"

echo "[3/4] Creating .env template..."
if [ ! -f .env ]; then
  cat > .env <<'EOF'
OPENAI_API_KEY=your_openai_api_key_here
HF_TOKEN=your_huggingface_token_here
EOF
fi
STEP3="[x]"

echo "[4/4] Ensuring runtime directories exist..."
mkdir -p embeddings models logs evaluation
STEP4="[x]"

echo ""
echo "Setup checklist:"
echo "$STEP1 venv created, activated, and dependencies installed"
echo "$STEP2 spaCy model en_core_web_sm downloaded"
echo "$STEP3 .env template created (if missing)"
echo "$STEP4 embeddings/, models/, logs/, evaluation/ directories ensured"

echo "Setup completed successfully."
