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
LEARNING_RATE=1e-4
NUM_TRAINING_EXAMPLES=128
NUM_VALIDATION_EXAMPLES=16
BATCH_SIZE=8
NUM_EPOCHS=2
SEED=42
GRADIENT_ACCUMULATION_STEPS=4
VALIDATE_EVERY_N_UPDATES=4
GRADIENT_CLIP=1.0
LOG_TRAINING_DATA_EVERY_N_UPDATES=1

uv run cs336_alignment/sft_train_main.py \
    --model_id="$MODEL_ID" \
    --prompt_template_path="$PROMPT_TEMPLATE_PATH" \
    --output_dir="$OUTPUT_DIR" \
    --wandb_entity="$WANDB_ENTITY" \
    --wandb_project="$WANDB_PROJECT" \
    --wandb_run_name="$WANDB_RUN_NAME" \
    --learning_rate="$LEARNING_RATE" \
    --num_training_examples="$NUM_TRAINING_EXAMPLES" \
    --num_validation_examples="$NUM_VALIDATION_EXAMPLES" \
    --batch_size="$BATCH_SIZE" \
    --num_epochs="$NUM_EPOCHS" \
    --seed="$SEED" \
    --gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
    --gradient_clip="$GRADIENT_CLIP" \
    --log_training_data_every_n_updates="$LOG_TRAINING_DATA_EVERY_N_UPDATES" \
    --validate_every_n_updates="$VALIDATE_EVERY_N_UPDATES"