"""Main entry point for training GRPO."""

import os

from typing import Any

import datasets
import numpy as np
import torch
import transformers
import vllm
import wandb

from absl import app
from absl import flags
from absl import logging

from cs336_alignment import custom_grader
from cs336_alignment import data_utils
from cs336_alignment import grpo_train_config
from cs336_alignment import grpo_utils
from cs336_alignment import pretrained_model_checkpoint
from cs336_alignment import sft_helpers
from cs336_alignment import vllm_utils


_model_id = flags.DEFINE_string(
    "model_id",
    "",
    "The ID of the model to use for training.",
)
_prompt_template_path = flags.DEFINE_string(
    "prompt_template_path",
    "",
    "The path to the prompt template to use for training.",
)
_output_dir = flags.DEFINE_string(
    "output_dir",
    "",
    "The directory to save the output to.",
)
_wandb_entity = flags.DEFINE_string(
    "wandb_entity",
    "",
    "The entity to use for Weights and Biases.",
)
_wandb_project = flags.DEFINE_string(
    "wandb_project",
    "",
    "The project to use for Weights and Biases.",
)
_wandb_run_name = flags.DEFINE_string(
    "wandb_run_name",
    "",
    "The name of the run to use for Weights and Biases.",
)
_task_name = flags.DEFINE_enum(
    "task_name",
    "gsm8k",
    ["gsm8k", "countdown"],
    "The name of the task to use for training.",
)
_seed = flags.DEFINE_integer(
    "seed",
    42,
    "The seed to use for training.",
)
_checkpoint_every_n_grpo_steps = flags.DEFINE_integer(
    "checkpoint_every_n_grpo_steps",
    1,
    "The number of GRPO steps to use for checkpointing.",
)
_max_num_checkpoints = flags.DEFINE_integer(
    "max_num_checkpoints",
    4,
    "The maximum number of checkpoints to save.",
)


def _set_seed(seed: int) -> None:
    """Sets the seed for the random number generators."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _check_at_least_two_gpus() -> None:
    """Checks that there are at least two GPUs available."""
    if torch.cuda.device_count() < 2:
        raise app.UsageError("There must be at least two GPUs available.")


def _load_prompt_template(path: str) -> str:
    """Loads the prompt template from the given path."""
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def _get_vllm_model_and_sampling_params(
    policy_model: transformers.PreTrainedModel,
    train_config: grpo_train_config.GrpoTrainConfig,
) -> tuple[vllm.LLM, vllm.SamplingParams, vllm.SamplingParams]:
    """Gets the vLLM model and sampling parameters.

    Returns:
        tuple[vllm.LLM, vllm.SamplingParams, vllm.SamplingParams]:
            The vLLM model, training sampling parameters, and evaluation sampling parameters.
    """
    vllm_model = vllm_utils.init_vllm(
        model_id=_model_id.value,
        device="cuda:1",
        seed=_seed.value,
        gpu_memory_utilization=train_config.gpu_memory_utilization,
    )
    vllm_utils.load_policy_into_vllm_instance(
        policy=policy_model,
        vllm_instance=vllm_model,
    )
    training_sampling_params = vllm.SamplingParams(
        temperature=train_config.sampling_temperature,
        top_p=train_config.top_p,
        max_tokens=train_config.sampling_max_tokens,
        min_tokens=train_config.sampling_min_tokens,
        n=train_config.group_size,
        stop=train_config.sampling_stop,
        include_stop_str_in_output=True,
        logprobs=1,
    )
    evaluation_sampling_params = vllm.SamplingParams(
        temperature=train_config.sampling_temperature,
        top_p=train_config.top_p,
        max_tokens=train_config.sampling_max_tokens,
        min_tokens=train_config.sampling_min_tokens,
        n=1,
        stop=train_config.sampling_stop,
        include_stop_str_in_output=True,
        logprobs=None,
    )
    return vllm_model, training_sampling_params, evaluation_sampling_params


def _get_policy_model_and_tokenizer(
    device: str,
) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizerBase]:
    """Gets the policy model and tokenizer."""
    policy_model = transformers.AutoModelForCausalLM.from_pretrained(
        _model_id.value,
        dtype=torch.bfloat16,
    ).to(
        device  # pyright: ignore[reportArgumentType]
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        _model_id.value,
    )
    return policy_model, tokenizer  # pyright: ignore[reportReturnType]


def _get_train_eval_datasets(
    task_name: str,
) -> tuple[datasets.Dataset, datasets.Dataset]:
    """Gets the training and evaluation datasets."""
    if task_name == "gsm8k":
        train_ds = datasets.load_dataset(
            "openai/gsm8k",
            "main",
            split="train",
        )
        eval_ds = datasets.load_dataset(
            "openai/gsm8k",
            "main",
            split="test",
        )
        return train_ds, eval_ds
    elif task_name == "countdown":
        countdown_ds = datasets.load_dataset(
            "Jiayi-Pan/Countdown-Tasks-3to4",
            split="train",
        )
        split_dataset_dict = countdown_ds.train_test_split(
            test_size=0.15, shuffle=False, seed=_seed.value
        )
        train_ds = split_dataset_dict["train"]
        eval_ds = split_dataset_dict["test"]
        return train_ds, eval_ds
    else:
        raise ValueError(f"Invalid task name: {task_name}")


def _get_optimizer(
    policy_model: transformers.PreTrainedModel,
    train_config: grpo_train_config.GrpoTrainConfig,
) -> torch.optim.Optimizer:
    """Gets the optimizer for the policy model."""
    return torch.optim.AdamW(
        policy_model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.adamw_weight_decay,
        betas=(train_config.adamw_beta_1, train_config.adamw_beta_2),
    )


def _init_wandb_run(
    wandb_entity: str,
    wandb_project: str,
    wandb_run_name: str,
    model_id: str,
    prompt_template: str,
    seed: int,
    config: grpo_train_config.GrpoTrainConfig,
) -> Any:
    """Initializes the WandB run."""
    return wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        name=wandb_run_name,
        config={
            "model_id": model_id,
            "prompt_template": prompt_template,
            "seed": seed,
            "n_grpo_steps": config.n_grpo_steps,
            "learning_rate": config.learning_rate,
            "advantage_epsilon": config.advantage_epsilon,
            "rollout_batch_size": config.rollout_batch_size,
            "group_size": config.group_size,
            "sampling_temperature": config.sampling_temperature,
            "top_p": config.top_p,
            "sampling_max_tokens": config.sampling_max_tokens,
            "sampling_min_tokens": config.sampling_min_tokens,
            "sampling_stop": config.sampling_stop,
            "epochs_per_rollout_batch": config.epochs_per_rollout_batch,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "loss_type": config.loss_type,
            "use_std_normalization": config.use_std_normalization,
            "adamw_weight_decay": config.adamw_weight_decay,
            "adamw_beta_1": config.adamw_beta_1,
            "adamw_beta_2": config.adamw_beta_2,
            "gradient_clip": config.gradient_clip,
            "cliprange": config.cliprange,
            "validation_every_n_updates": config.validation_every_n_updates,
            "n_microbatches_per_rollout_batch": config.n_microbatches_per_rollout_batch,
            "microbatch_size": config.microbatch_size,
        },
    )


def _get_pretrained_model_checkpoint_manager(
    output_dir: str,
    output_name_prefix: str,
    max_num_checkpoints: int,
) -> pretrained_model_checkpoint.PretrainedModelCheckpointManager:
    """Gets the pretrained model checkpoint manager."""
    return pretrained_model_checkpoint.PretrainedModelCheckpointManager(
        output_dir=output_dir,
        output_name_prefix=output_name_prefix,
        max_num_checkpoints=max_num_checkpoints,
    )


def _get_grpo_train_one_epoch_data_for_gsm8k(
    vllm_old_model: vllm.LLM,
    training_sampling_params: vllm.SamplingParams,
    tokenizer: transformers.PreTrainedTokenizerBase,
    train_ds_batch: datasets.Dataset,
    train_config: grpo_train_config.GrpoTrainConfig,
    prompt_template: str,
):
    """Gets the data for the GRPO train one epoch function for GSM8K."""
    (
        repeated_model_input_prompts,
        model_responses,
        repeated_ground_truth_answers,
        old_log_probs,
    ) = grpo_utils.sample_grpo_rollouts(
        model=vllm_old_model,
        sampling_params=training_sampling_params,
        questions=train_ds_batch["question"],
        ground_truth_answers=train_ds_batch["answer"],
        prompt_fn=lambda questions: data_utils.generate_gsm8k_prompt_from_question_list(
            prompt_template=prompt_template,
            questions=questions,
        ),
    )
    tokenized_input_dict = sft_helpers.tokenize_prompt_and_output(
        prompt_strs=repeated_model_input_prompts,
        output_strs=model_responses,
        tokenizer=tokenizer,
    )
    group_normalized_rewards, raw_rewards, rewards_metadata = (
        grpo_utils.compute_group_normalized_rewards(
            reward_fn=custom_grader.gsm8k_reward_fn,
            rollout_responses=model_responses,
            repeated_ground_truths=repeated_ground_truth_answers,
            group_size=train_config.group_size,
            advantage_eps=train_config.advantage_epsilon,
            normalize_by_std=train_config.use_std_normalization,
        )
    )
    return (
        tokenized_input_dict,
        old_log_probs,
        raw_rewards,
        rewards_metadata,
        group_normalized_rewards,
    )


def _get_grpo_train_one_epoch_data_for_countdown(
    vllm_old_model: vllm.LLM,
    training_sampling_params: vllm.SamplingParams,
    tokenizer: transformers.PreTrainedTokenizerBase,
    train_ds_batch: datasets.Dataset,
    train_config: grpo_train_config.GrpoTrainConfig,
    prompt_template: str,
):
    """Gets the data for the GRPO train one epoch function for Countdown."""
    input_prompts = data_utils.generate_countdown_prompt_from_nums_target_lists(
        prompt_template=prompt_template,
        nums_list=train_ds_batch["nums"],
        target_list=train_ds_batch["target"],
    )
    repeated_model_input_prompts, model_responses, _, old_log_probs = (
        grpo_utils.sample_grpo_rollouts(
            model=vllm_old_model,
            sampling_params=training_sampling_params,
            questions=input_prompts,
            ground_truth_answers=train_ds_batch["target"],
            prompt_fn=lambda questions: questions,
        )
    )
    tokenized_input_dict = sft_helpers.tokenize_prompt_and_output(
        prompt_strs=repeated_model_input_prompts,
        output_strs=model_responses,
        tokenizer=tokenizer,
    )
    repeated_nums_list = []
    repeated_target_list = []
    for nums, target in zip(train_ds_batch["nums"], train_ds_batch["target"]):
        repeated_nums_list.extend([nums] * train_config.group_size)
        repeated_target_list.extend([target] * train_config.group_size)
    group_normalized_rewards, raw_rewards, rewards_metadata = (
        grpo_utils.compute_group_normalized_rewards_for_countdown(
            reward_fn=custom_grader.countdown_reward_fn,
            rollout_responses=model_responses,
            repeated_nums_list=repeated_nums_list,
            repeated_target_list=repeated_target_list,
            group_size=train_config.group_size,
            advantage_eps=train_config.advantage_epsilon,
            normalize_by_std=train_config.use_std_normalization,
        )
    )
    return (
        tokenized_input_dict,
        old_log_probs,
        raw_rewards,
        rewards_metadata,
        group_normalized_rewards,
    )


def main(argv):
    """Main function for GRPO training."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    _check_at_least_two_gpus()
    _set_seed(_seed.value)
    train_config = grpo_train_config.get_grpo_train_config()

    policy_model, tokenizer = _get_policy_model_and_tokenizer(device="cuda:0")
    vllm_old_model, training_sampling_params, evaluation_sampling_params = (
        _get_vllm_model_and_sampling_params(policy_model, train_config)
    )
    prompt_template = _load_prompt_template(_prompt_template_path.value)
    train_ds, eval_ds = _get_train_eval_datasets(
        task_name=_task_name.value,
    )
    num_datapoints_per_grpo_batch = (
        train_config.rollout_batch_size // train_config.group_size
    )
    wandb_run = _init_wandb_run(
        wandb_entity=_wandb_entity.value,
        wandb_project=_wandb_project.value,
        wandb_run_name=_wandb_run_name.value,
        model_id=_model_id.value,
        prompt_template=prompt_template,
        seed=_seed.value,
        config=train_config,
    )
    logging.info(f"GRPO training configuration: {train_config}")
    optimizer = _get_optimizer(
        policy_model=policy_model,
        train_config=train_config,
    )
    pretrained_model_checkpoint_manager = _get_pretrained_model_checkpoint_manager(
        output_dir=_output_dir.value,
        output_name_prefix="grpo_model",
        max_num_checkpoints=_max_num_checkpoints.value,
    )
    global_update_count = 0
    for grpo_step in range(train_config.n_grpo_steps):
        logging.info(f"Starting GRPO step {grpo_step}...")
        train_ds_batch = grpo_utils.randomly_sample_batch(
            ds=train_ds, num_datapoints=num_datapoints_per_grpo_batch
        )
        if grpo_step > 0:
            vllm_utils.load_policy_into_vllm_instance(
                policy=policy_model,
                vllm_instance=vllm_old_model,
            )
        logging.info(
            f"Sampling {train_config.rollout_batch_size} rollouts from "
            f"{len(train_ds_batch)} datapoints and group size {train_config.group_size} "
            f"for GRPO step {grpo_step}..."
        )
        if _task_name.value == "gsm8k":
            (
                tokenized_input_dict,
                old_log_probs,
                raw_rewards,
                rewards_metadata,
                group_normalized_rewards,
            ) = _get_grpo_train_one_epoch_data_for_gsm8k(
                vllm_old_model=vllm_old_model,
                training_sampling_params=training_sampling_params,
                tokenizer=tokenizer,
                train_ds_batch=train_ds_batch,
                train_config=train_config,
                prompt_template=prompt_template,
            )
        elif _task_name.value == "countdown":
            (
                tokenized_input_dict,
                old_log_probs,
                raw_rewards,
                rewards_metadata,
                group_normalized_rewards,
            ) = _get_grpo_train_one_epoch_data_for_countdown(
                vllm_old_model=vllm_old_model,
                training_sampling_params=training_sampling_params,
                tokenizer=tokenizer,
                train_ds_batch=train_ds_batch,
                train_config=train_config,
                prompt_template=prompt_template,
            )
        else:
            raise ValueError(f"Invalid task name: {_task_name.value}")
        for epoch in range(train_config.epochs_per_rollout_batch):
            global_update_count = grpo_utils.grpo_train_one_epoch(
                policy_model=policy_model,
                optimizer=optimizer,
                task_name=_task_name.value,
                all_input_ids=tokenized_input_dict["input_ids"],
                all_labels=tokenized_input_dict["labels"],
                all_response_mask=tokenized_input_dict["response_mask"],
                all_old_log_probs=old_log_probs,
                all_raw_rewards=raw_rewards,
                all_format_rewards=rewards_metadata["format_rewards"],
                all_answer_rewards=rewards_metadata["answer_rewards"],
                all_advantages=group_normalized_rewards,
                vllm_old_model=vllm_old_model,
                evaluation_sampling_params=evaluation_sampling_params,
                prompt_template=prompt_template,
                train_config=train_config,
                eval_ds=eval_ds,
                wandb_run=wandb_run,
                grpo_step=grpo_step,
                epoch=epoch,
                global_update_count=global_update_count,
            )
        if grpo_step % _checkpoint_every_n_grpo_steps.value == 0:
            logging.info(
                f"Saving policy model and tokenizer for GRPO step {grpo_step}..."
            )
            checkpoint_dir_name = pretrained_model_checkpoint_manager.save_checkpoint(
                model=policy_model,
                tokenizer=tokenizer,
                step=grpo_step,
            )
            logging.info(
                f"Policy model and tokenizer saved successfully to "
                f"{os.path.join(_output_dir.value, checkpoint_dir_name)}."
            )


if __name__ == "__main__":
    app.run(main)
