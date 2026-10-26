#!/usr/bin/env bash
# The answer evaluation on a Kaggle GPU notebook (kaggle/answer-eval.ipynb runs this).
# Builds llama.cpp for the GPU, then every candidate answers the questions of SPLIT and the
# judge grades them; results go to /kaggle/working/results (the notebook's output). Model files
# live in /tmp and are deleted after use (the 37 GB of models do not all fit at once).
# Re-running resumes: answers and grades already written are kept.
# Usage: scripts/kaggle_run.sh [dev|test]
set -euo pipefail
split="${1:-dev}"
out=/kaggle/working/results
export MODELS_DIR=/tmp/models UV_LINK_MODE=copy
mkdir -p "$out" /tmp/llama
nvidia-smi --query-gpu=name,memory.total --format=csv

tag=$(uv run python -c "from ai_act_qa.config import load_config; print(load_config()['llama_cpp']['tag'])")
[ -x /tmp/llama/llama-server ] || bash scripts/build_llama.sh "$tag" /tmp/llama

# Earlier results of this split (from a previous run's output, if attached) are reused.
mkdir -p "reports/answers/$split"
if [ -d "$out/answers/$split" ]; then cp -rn "$out/answers/$split/." "reports/answers/$split/"; fi

run() { uv run python -m ai_act_qa.answer_eval "$@" --split "$split" --mode binary \
          --binary /tmp/llama/llama-server --delete-after; }
save() { mkdir -p "$out/answers"; cp -r "reports/answers/$split" "$out/answers/"; }
trap save EXIT   # whatever happens, keep what was produced

uv run python -m ai_act_qa.answer_eval retrieve --split "$split"
run generate
save
run judge
uv run python -m ai_act_qa.answer_eval report --split "$split"
cp "reports/answers_$split.json" "$out/"
git log -1 --format='%H %s' > "$out/commit.txt"
echo "done: results in $out"
