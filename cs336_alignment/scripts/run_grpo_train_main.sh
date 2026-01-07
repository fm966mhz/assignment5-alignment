#!/bin/bash

set -euo pipefail

BASE_DIR="$1"
EXP_NAME="$2"

MODEL_ID_OR_PATH="Qwen/Qwen2.5-Math-1.5B"
# MODEL_ID_OR_PATH="$BASE_DIR/notes_and_writeups/assignment5_output/ei_on_a6000_run18/policy_model_checkpoint_4"
PROMPT_TEMPLATE_PATH="$BASE_DIR/assignment5-alignment/cs336_alignment/prompts/my_system_prompt.prompt"
OUTPUT_DIR="$BASE_DIR/notes_and_writeups/assignment5_output/$EXP_NAME"
TASK_NAME="countdown"
WANDB_ENTITY="fm966hz"
WANDB_PROJECT="cs336-assignment5-alignment"
WANDB_RUN_NAME="$EXP_NAME"
SEED=42
N_GRPO_STEPS=200
# For the GRPO-clipo loss, also tried 8e-5. Gradient exploded.
# For long training runs, stable training is more important than faster initial training. This can
# be monitored through the gradient norm.
LEARNING_RATE=2e-5
ADVANTAGE_EPSILON=1e-6
# Rollout batch size 512 consistently worse than 256.
ROLLOUT_BATCH_SIZE=256
GROUP_SIZE=8
SAMPLING_TEMPERATURE=1.0
TOP_P=1.0
SAMPLING_MAX_TOKENS=1024
SAMPLING_MIN_TOKENS=4
SAMPLING_STOP="</answer>"
# 1 epoch per rollout batch is significantly better than 2 or more when train batch size is the same
# as the rollout batch size.
# Maybe this should be set dynamically based on the current rollout batch's average reward. In
# the beginning of training, the reward is low, so no need to train for more epochs. This is 
# roughly along the lines of exploring more rather than exploiting rollouts that have low rewards.
# But for rollout batches with high rewards, more epochs of training may be beneficial.
EPOCHS_PER_ROLLOUT_BATCH=2
# Training on 128 effective batch size is significantly worse than 256.
TRAIN_BATCH_SIZE=128    
POLICY_MODEL_INFERENCE_BATCH_SIZE=16
EVALUATION_SAMPLE_SIZE=1024
GRADIENT_ACCUMULATION_STEPS=64
GPU_MEMORY_UTILIZATION=0.85
LOSS_TYPE="grpo_clip"
USE_LENGTH_NORMALIZATION=True
USE_STD_NORMALIZATION=True
ADAMW_WEIGHT_DECAY=0
ADAMW_BETA_1=0.9
ADAMW_BETA_2=0.95
GRADIENT_CLIP=1.0
CLIPRANGE=0.2
EARLY_STOP_KL_DIVERGENCE_THRESHOLD=0.2
VALIDATION_EVERY_N_UPDATES=10
LOG_TRAINING_METRICS_EVERY_N_MICROBATCHES=16
CHECKPOINT_EVERY_N_GRPO_STEPS=5
MAX_NUM_CHECKPOINTS=4


uv run cs336_alignment/grpo_train_main.py \
    --model_id_or_path="$MODEL_ID_OR_PATH" \
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
    --policy_model_inference_batch_size="$POLICY_MODEL_INFERENCE_BATCH_SIZE" \
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
    --early_stop_kl_divergence_threshold="$EARLY_STOP_KL_DIVERGENCE_THRESHOLD" \
    --validation_every_n_updates="$VALIDATION_EVERY_N_UPDATES" \
    --log_training_metrics_every_n_microbatches="$LOG_TRAINING_METRICS_EVERY_N_MICROBATCHES" \
    --checkpoint_every_n_grpo_steps="$CHECKPOINT_EVERY_N_GRPO_STEPS" \
    --max_num_checkpoints="$MAX_NUM_CHECKPOINTS"