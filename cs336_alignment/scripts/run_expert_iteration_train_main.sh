#!/bin/bash

set -euo pipefail

BASE_DIR="$1"
EXP_NAME="$2"

MODEL_ID="Qwen/Qwen2.5-Math-1.5B"
PROMPT_TEMPLATE_PATH="$BASE_DIR/assignment5-alignment/cs336_alignment/prompts/my_system_prompt.prompt"
OUTPUT_DIR="$BASE_DIR/notes_and_writeups/assignment5_output/$EXP_NAME"
WANDB_ENTITY="fm966hz"
WANDB_PROJECT="cs336-assignment5-alignment"
WANDB_RUN_NAME="$EXP_NAME"
LEARNING_RATE=2e-5
NUM_TRAINING_EXAMPLES=-1
NUM_VALIDATION_EXAMPLES=512
MAX_MODEL_RESPONSE_LENGTH=512
SAMPLE_BATCH_SIZE=4096
NUM_ROLLOUTS=4
TRAINING_BATCH_SIZE=8
NUM_EPOCHS=4
NUM_EXPERT_ITERATIONS=5
SEED=42
GRADIENT_ACCUMULATION_STEPS=8
GRADIENT_CLIP=1.0
CHECKPOINT_EVERY_N_EXPERT_ITERATIONS=1

uv run cs336_alignment/expert_iteration_train_main.py \
    --model_id="$MODEL_ID" \
    --prompt_template_path="$PROMPT_TEMPLATE_PATH" \
    --output_dir="$OUTPUT_DIR" \
    --wandb_entity="$WANDB_ENTITY" \
    --wandb_project="$WANDB_PROJECT" \
    --wandb_run_name="$WANDB_RUN_NAME" \
    --learning_rate="$LEARNING_RATE" \
    --num_training_examples="$NUM_TRAINING_EXAMPLES" \
    --num_validation_examples="$NUM_VALIDATION_EXAMPLES" \
    --max_model_response_length="$MAX_MODEL_RESPONSE_LENGTH" \
    --sample_batch_size="$SAMPLE_BATCH_SIZE" \
    --num_rollouts="$NUM_ROLLOUTS" \
    --training_batch_size="$TRAINING_BATCH_SIZE" \
    --num_epochs="$NUM_EPOCHS" \
    --num_expert_iterations="$NUM_EXPERT_ITERATIONS" \
    --seed="$SEED" \
    --gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
    --gradient_clip="$GRADIENT_CLIP" \
    --checkpoint_every_n_expert_iterations="$CHECKPOINT_EVERY_N_EXPERT_ITERATIONS"