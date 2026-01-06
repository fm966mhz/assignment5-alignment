#!/bin/bash

set -euo pipefail

BASE_DIR="$1"
EXP_NAME="$2"

MODEL_ID="Qwen/Qwen2.5-Math-1.5B"
PROMPT_TEMPLATE_PATH="$BASE_DIR/assignment5-alignment/cs336_alignment/prompts/my_system_prompt.prompt"
OUTPUT_DIR="$BASE_DIR/notes_and_writeups/assignment5_output/$EXP_NAME"
TASK_NAME="gsm8k"
WANDB_ENTITY="fm966hz"
WANDB_PROJECT="cs336-assignment5-alignment"
WANDB_RUN_NAME="$EXP_NAME"
SEED=42
N_GRPO_STEPS=200
LEARNING_RATE=6e-5
ADVANTAGE_EPSILON=1e-6
ROLLOUT_BATCH_SIZE=256
GROUP_SIZE=8
SAMPLING_TEMPERATURE=1.0
TOP_P=1.0
SAMPLING_MAX_TOKENS=1024
SAMPLING_MIN_TOKENS=4
SAMPLING_STOP="</answer>"
EPOCHS_PER_ROLLOUT_BATCH=1
TRAIN_BATCH_SIZE=256
EVALUATION_SAMPLE_SIZE=-1
GRADIENT_ACCUMULATION_STEPS=128
GPU_MEMORY_UTILIZATION=0.85
LOSS_TYPE="reinforce_with_baseline"
USE_STD_NORMALIZATION=True
ADAMW_WEIGHT_DECAY=0
ADAMW_BETA_1=0.9
ADAMW_BETA_2=0.95
GRADIENT_CLIP=1.0
CLIPRANGE=0.2
VALIDATION_EVERY_N_UPDATES=5
LOG_TRAINING_METRICS_EVERY_N_MICROBATCHES=10
CHECKPOINT_EVERY_N_GRPO_STEPS=1
MAX_NUM_CHECKPOINTS=4


uv run cs336_alignment/grpo_train_main.py \
    --model_id="$MODEL_ID" \
    --prompt_template_path="$PROMPT_TEMPLATE_PATH" \
    --output_dir="$OUTPUT_DIR" \
    --task_name="$TASK_NAME" \
    --wandb_entity="$WANDB_ENTITY" \
    --wandb_project="$WANDB_PROJECT" \
    --wandb_run_name="$WANDB_RUN_NAME" \
    --seed="$SEED" \
    --n_grpo_steps="$N_GRPO_STEPS" \
    --learning_rate="$LEARNING_RATE" \
    --advantage_epsilon="$ADVANTAGE_EPSILON" \
    --rollout_batch_size="$ROLLOUT_BATCH_SIZE" \
    --group_size="$GROUP_SIZE" \
    --sampling_temperature="$SAMPLING_TEMPERATURE" \
    --top_p="$TOP_P" \
    --sampling_max_tokens="$SAMPLING_MAX_TOKENS" \
    --sampling_min_tokens="$SAMPLING_MIN_TOKENS" \
    --sampling_stop="$SAMPLING_STOP" \
    --epochs_per_rollout_batch="$EPOCHS_PER_ROLLOUT_BATCH" \
    --train_batch_size="$TRAIN_BATCH_SIZE" \
    --evaluation_sample_size="$EVALUATION_SAMPLE_SIZE" \
    --gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
    --gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
    --loss_type="$LOSS_TYPE" \
    --use_std_normalization="$USE_STD_NORMALIZATION" \
    --adamw_weight_decay="$ADAMW_WEIGHT_DECAY" \
    --adamw_beta_1="$ADAMW_BETA_1" \
    --adamw_beta_2="$ADAMW_BETA_2" \
    --gradient_clip="$GRADIENT_CLIP" \
    --cliprange="$CLIPRANGE" \
    --validation_every_n_updates="$VALIDATION_EVERY_N_UPDATES" \
    --log_training_metrics_every_n_microbatches="$LOG_TRAINING_METRICS_EVERY_N_MICROBATCHES" \
    --checkpoint_every_n_grpo_steps="$CHECKPOINT_EVERY_N_GRPO_STEPS" \
    --max_num_checkpoints="$MAX_NUM_CHECKPOINTS"