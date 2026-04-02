#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"
INPUT_FILE="${INPUT_FILE:-$ROOT_DIR/data/benchmarks/ambigqa.validation.goldfull.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-$ROOT_DIR/results/scores_qwen14b_goldfull_strict.jsonl}"

K_SAMPLES="${K_SAMPLES:-8}"
SEMANTIC_THRESHOLD="${SEMANTIC_THRESHOLD:-0.93}"
SEMANTIC_LINKAGE="${SEMANTIC_LINKAGE:-complete}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

python -m ambiguity_score.run \
  --env-file "$ENV_FILE" \
  --backend hf \
  --model-id "${MODEL_ID:-Qwen/Qwen2.5-14B-Instruct}" \
  --embedding-model-id "${EMBEDDING_MODEL_ID:-Qwen/Qwen2.5-14B-Instruct}" \
  --hf-device "${HF_DEVICE:-auto}" \
  --hf-torch-dtype "${HF_TORCH_DTYPE:-bfloat16}" \
  --hf-max-input-length "${HF_MAX_INPUT_LENGTH:-1024}" \
  --semantic-threshold "$SEMANTIC_THRESHOLD" \
  --semantic-linkage "$SEMANTIC_LINKAGE" \
  --input "$INPUT_FILE" \
  --output "$OUTPUT_FILE" \
  --measures infogain sample_rep semantic_entropy token_entropy \
  --k_samples "$K_SAMPLES"
