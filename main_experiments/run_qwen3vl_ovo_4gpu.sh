#!/bin/bash
# OVO-Bench full evaluation: Qwen3-VL, recent_frames=4, 4-GPU parallel
# Default behavior: auto-pick the most free GPUs unless CUDA_VISIBLE_DEVICES is set.
# Release baseline only: no LoRA / no compression.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"

cd "${REPO_ROOT}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    CUDA_VISIBLE_DEVICES="$(
        nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t',' -k2,2nr \
        | head -n "${NUM_PROCESSES}" \
        | cut -d',' -f1 \
        | tr -d ' ' \
        | paste -sd, -
    )"
fi

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" \
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}" \
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}" \
${PYTHON_BIN} -m accelerate.commands.launch \
    --num_processes "${NUM_PROCESSES}" \
    --multi_gpu \
    --mixed_precision bf16 \
    main_experiments/eval_qwen3vl_ovo.py \
    --model_path "${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}" \
    --anno_path "${OVO_ANNO_PATH:-data/ovo_bench/ovo_bench_new.json}" \
    --chunked_dir "${OVO_CHUNKED_DIR:-data/ovo_bench/chunked_videos}" \
    --result_dir "${OVO_RESULT_DIR:-main_experiments/results/ovo_qwen3vl_recent4}" \
    --recent_frames_only "${RECENT_FRAMES_ONLY:-4}" \
    --chunk_duration "${CHUNK_DURATION:-1.0}" \
    --fps "${FPS:-1.0}" \
    --max_qa_tokens "${MAX_QA_TOKENS:-256}"
