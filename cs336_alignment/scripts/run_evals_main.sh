#!/bin/bash

set -euo pipefail

BASE_DIR="$1"

MODEL_PATH="Qwen/Qwen2.5-Math-1.5B"
SYSTEM_PROMPT_PATH="$BASE_DIR/assignment5-alignment/cs336_alignment/prompts/my_system_prompt.prompt"
FAST_EVAL=true
MAX_EVAL_SAMPLES=-1

RUN_GSM8K_EVAL=true
RUN_COUNTDOWN_EVAL=true
OUTPUT_PATH_PREFIX="$BASE_DIR/notes_and_writeups/assignment5_output/baseline_eval"

COUNTDOWN_TEST_DATASET_PATH="$BASE_DIR/data/Jiayi-Pan_Countdown-Tasks-3to4_test"


uv run cs336_alignment/evals_main.py \
    --model_path="$MODEL_PATH" \
    --run_gsm8k_evaluation="$RUN_GSM8K_EVAL" \
    --run_countdown_evaluation="$RUN_COUNTDOWN_EVAL" \
    --system_prompt_path="$SYSTEM_PROMPT_PATH" \
    --output_path_prefix="$OUTPUT_PATH_PREFIX" \
    --fast_eval="$FAST_EVAL" \
    --max_eval_samples="$MAX_EVAL_SAMPLES" \
    --countdown_test_dataset_path="$COUNTDOWN_TEST_DATASET_PATH"
    