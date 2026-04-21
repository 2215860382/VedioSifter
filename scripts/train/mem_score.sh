#!/bin/bash
# 记忆打分：调大模型给每条记忆片段打分，直接输出 VERL 训练格式 parquet

set -e

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

MODEL_NAME=${MODEL_NAME:-"Qwen3-VL-32B-Instruct"}
API_URL=${API_URL:-"http://localhost:8000/v1"}
INPUT_FILE=${INPUT_FILE:-"${ROOT}/data/raw/dataset.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"${ROOT}/data/rl_data"}
PROMPT_CONFIG=${PROMPT_CONFIG:-"${ROOT}/configs/scorer_prompt.yaml"}
MAX_CONCURRENT=${MAX_CONCURRENT:-32}
TRAIN_RATIO=${TRAIN_RATIO:-0.9}

echo "========== Memory Scoring =========="
echo "Model:      ${MODEL_NAME}"
echo "Input:      ${INPUT_FILE}"
echo "Output dir: ${OUTPUT_DIR}"

python "${ROOT}/my_verl/memory_scorer.py" \
    --input_file "${INPUT_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_name "${MODEL_NAME}" \
    --api_url "${API_URL}" \
    --prompt_config "${PROMPT_CONFIG}" \
    --max_concurrent "${MAX_CONCURRENT}" \
    --train_ratio "${TRAIN_RATIO}"

echo "Done. Training data saved to ${OUTPUT_DIR}"
