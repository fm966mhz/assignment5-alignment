"""Utility functions for GRPO training."""

from typing import Any, Callable, Literal

import datasets
import numpy as np
import torch
import tqdm
import transformers
import vllm

from absl import logging
from jaxtyping import Bool, Float, Int

from cs336_alignment import custom_grader
from cs336_alignment import data_utils
from cs336_alignment import eval_utils
from cs336_alignment import grpo_train_config
from cs336_alignment import sft_helpers
from cs336_alignment import vllm_utils


def compute_group_normalized_rewards(  # pylint: disable=too-many-arguments
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    *,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[
    Float[torch.Tensor, "rollout_batch_size 1"],
    Float[torch.Tensor, "rollout_batch_size 1"],
    dict[str, Float[torch.Tensor, "rollout_batch_size 1"]],
]:
    """Computes group-normalized rewards for a batch of rollout responses.

    Args:
        reward_fn: Callable[[str, str], dict[str, float]],
            scores the rollout responses against the ground truths,
            producing a dict with keys
            "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str], rollouts from the policy.
            The length of this list is
            `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
        repeated_ground_truths: list[list], the ground truths for the examples.
            The length of this list is `rollout_batch_size`,
            because the ground truth for each example is repeated `group_size` times.
        group_size: int, number of rollouts per group.
        advantage_eps: float, epsilon to avoid division by zero
            during group normalization.
        normalize_by_std: bool, whether to normalize the rewards by
            std(rewards).

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            torch.Tensor of shape (rollout_batch_size,):
                group-normalized rewards for each rollout response.
            torch.Tensor of shape (rollout_batch_size,):
                raw rewards for each rollout response.
            dict[str, Any]: metadata for the rewards of the rollout batch.
    """
    assert len(rollout_responses) == len(repeated_ground_truths), (
        f"Rolloutput response length {len(rollout_responses)} doesn't match ground truth "
        f"{len(repeated_ground_truths)}"
    )
    assert len(rollout_responses) % group_size == 0, (
        f"Rollout responses length {len(rollout_responses)} cannot be divided by group size "
        f"{group_size}"
    )
    all_raw_rewards = []
    all_format_rewards = []
    all_answer_rewards = []
    raw_rewards_per_question = []
    for rollout_response, ground_truth in zip(
        rollout_responses, repeated_ground_truths
    ):
        reward_dict = reward_fn(rollout_response, ground_truth)
        raw_rewards_per_question.append(reward_dict.get("reward", 0.0))
        all_format_rewards.append(reward_dict.get("format_reward", 0.0))
        all_answer_rewards.append(reward_dict.get("answer_reward", 0.0))
        if len(raw_rewards_per_question) % group_size == 0:
            all_raw_rewards.append(raw_rewards_per_question)
            raw_rewards_per_question = []
    all_format_rewards = torch.tensor(all_format_rewards)
    all_answer_rewards = torch.tensor(all_answer_rewards)
    all_raw_rewards = torch.tensor(all_raw_rewards)
    group_normalized_rewards = all_raw_rewards - torch.mean(
        all_raw_rewards, dim=-1, keepdim=True
    )
    if normalize_by_std:
        group_normalized_rewards = group_normalized_rewards / (
            torch.std(all_raw_rewards, dim=-1, keepdim=True) + advantage_eps
        )
    return (
        group_normalized_rewards.reshape(-1, 1),
        all_raw_rewards.reshape(-1, 1),
        {
            "format_rewards": all_format_rewards.reshape(-1, 1),
            "answer_rewards": all_answer_rewards.reshape(-1, 1),
        },
    )


def compute_group_normalized_rewards_for_countdown(
    reward_fn: Callable[[str, list[int], int], dict[str, float]],
    rollout_responses: list[str],
    repeated_nums_list: list[list[int]],
    repeated_target_list: list[int],
    group_size: int,
    *,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[
    Float[torch.Tensor, "rollout_batch_size 1"],
    Float[torch.Tensor, "rollout_batch_size 1"],
    dict[str, Float[torch.Tensor, "rollout_batch_size 1"]],
]:
    """Computes group-normalized rewards for a batch of rollout responses for Countdown."""
    assert (
        len(rollout_responses) == len(repeated_nums_list) == len(repeated_target_list)
    ), (
        f"Rolloutput response length {len(rollout_responses)} doesn't match nums list length "
        f"{len(repeated_nums_list)} or target list length {len(repeated_target_list)}"
    )
    assert len(rollout_responses) % group_size == 0, (
        f"Rollout responses length {len(rollout_responses)} cannot be divided by group size "
        f"{group_size}"
    )
    all_raw_rewards = []
    all_format_rewards = []
    all_answer_rewards = []
    raw_rewards_per_question = []
    for rollout_response, nums, target in zip(
        rollout_responses, repeated_nums_list, repeated_target_list
    ):
        reward_dict = reward_fn(rollout_response, nums, target)
        raw_rewards_per_question.append(reward_dict.get("reward", 0.0))
        all_format_rewards.append(reward_dict.get("format_reward", 0.0))
        all_answer_rewards.append(reward_dict.get("answer_reward", 0.0))
        if len(raw_rewards_per_question) % group_size == 0:
            all_raw_rewards.append(raw_rewards_per_question)
            raw_rewards_per_question = []
    all_format_rewards = torch.tensor(all_format_rewards)
    all_answer_rewards = torch.tensor(all_answer_rewards)
    all_raw_rewards = torch.tensor(all_raw_rewards)
    group_normalized_rewards = all_raw_rewards - torch.mean(
        all_raw_rewards, dim=-1, keepdim=True
    )
    if normalize_by_std:
        group_normalized_rewards = group_normalized_rewards / (
            torch.std(all_raw_rewards, dim=-1, keepdim=True) + advantage_eps
        )
    return (
        group_normalized_rewards.reshape(-1, 1),
        all_raw_rewards.reshape(-1, 1),
        {
            "format_rewards": all_format_rewards.reshape(-1, 1),
            "answer_rewards": all_answer_rewards.reshape(-1, 1),
        },
    )


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: Float[torch.Tensor, "rollout_batch_size 1"],
    policy_log_probs: Float[torch.Tensor, "rollout_batch_size seq_len"],
) -> Float[torch.Tensor, "rollout_batch_size seq_len"]:
    """Compute the naive policy gradient loss.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (rollout_batch_size, 1):
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (rollout_batch_size, sequence_length):
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (rollout_batch_size, sequence_length):
            the naive policy gradient loss.
    """
    assert len(raw_rewards_or_advantages.shape) == len(policy_log_probs.shape), (
        f"Raw rewards or advantages shape {raw_rewards_or_advantages.shape} doesn't match "
        f"policy log probs shape {policy_log_probs.shape}"
    )
    assert raw_rewards_or_advantages.shape[0] == policy_log_probs.shape[0], (
        f"Raw rewards or advantages length {raw_rewards_or_advantages.shape[0]} doesn't match "
        f"policy log probs length {policy_log_probs.shape[0]}"
    )
    assert raw_rewards_or_advantages.shape[1] == 1, (
        f"Raw rewards or advantages shape {raw_rewards_or_advantages.shape} doesn't have "
        f"shape (rollout_batch_size, 1)"
    )
    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: Float[torch.Tensor, "rollout_batch_size 1"],
    policy_log_probs: Float[torch.Tensor, "rollout_batch_size seq_len"],
    old_log_probs: Float[torch.Tensor, "rollout_batch_size seq_len"],
    cliprange: float,
) -> tuple[Float[torch.Tensor, "rollout_batch_size seq_len"], dict[str, Any]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (rollout_batch_size, 1):
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (rollout_batch_size, sequence_length):
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (rollout_batch_size, sequence_length):
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        torch.Tensor of shape (rollout_batch_size, sequence_length):
            the GRPO-Clip per-token loss.
        dict[str, Any]: metadata for the GRPO-Clip loss
            "is_token_clipped": torch.Tensor of shape (rollout_batch_size, sequence_length):
                whether the token is clipped.
            "approx_kl_divergence": torch.Tensor of shape (rollout_batch_size, sequence_length):
                the approximate KL divergence between the policy and the old policy.
    """
    log_ratio = policy_log_probs - old_log_probs
    rho = torch.exp(log_ratio)
    first_term = advantages * rho
    second_term = advantages * torch.clip(rho, min=1 - cliprange, max=1 + cliprange)
    grpo_per_token_clipped_loss = -torch.minimum(first_term, second_term)
    with torch.no_grad():
        is_token_clipped: Bool[torch.Tensor, "rollout_batch_size seq_len"] = (
            second_term < first_term
        )
        approx_kl_divergence: Float[torch.Tensor, "rollout_batch_size seq_len"] = (
            rho - 1 - log_ratio
        )
    return (
        grpo_per_token_clipped_loss,
        {
            "is_token_clipped": is_token_clipped,
            "approx_kl_divergence": approx_kl_divergence,
        },
    )


def compute_policy_gradient_loss(
    policy_log_probs: Float[torch.Tensor, "rollout_batch_size seq_len"],
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: Float[torch.Tensor, "rollout_batch_size 1"] | None = None,
    advantages: Float[torch.Tensor, "rollout_batch_size 1"] | None = None,
    old_log_probs: Float[torch.Tensor, "rollout_batch_size seq_len"] | None = None,
    cliprange: float | None = None,
) -> tuple[Float[torch.Tensor, "rollout_batch_size seq_len"], dict[str, Any]]:
    """Compute the policy gradient loss.

    Args:
        policy_log_probs: torch.Tensor of shape (rollout_batch_size, sequence_length):
            the log-probs of the policy.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
            the type of loss function to use.
        raw_rewards: torch.Tensor of shape (rollout_batch_size, 1):
            the raw rewards for each rollout response.
        advantages: torch.Tensor of shape (rollout_batch_size, 1):
            the advantages for each rollout response.
        old_log_probs: torch.Tensor of shape (rollout_batch_size, sequence_length):
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        torch.Tensor of shape (rollout_batch_size, sequence_length):
            the policy gradient loss.
        dict[str, Any]: metadata for the policy gradient loss
    """
    assert loss_type in {
        "no_baseline",
        "reinforce_with_baseline",
        "grpo_clip",
    }, f"Invalid loss type: {loss_type}"
    if loss_type == "no_baseline":
        assert (
            raw_rewards is not None
        ), "Raw rewards are required when loss type is 'no_baseline'."
        return (
            compute_naive_policy_gradient_loss(
                raw_rewards_or_advantages=raw_rewards, policy_log_probs=policy_log_probs
            ),
            {},
        )
    elif loss_type == "reinforce_with_baseline":
        assert (
            advantages is not None
        ), "Advantages are required when loss type is 'reinforce_with_baseline'."
        return (
            compute_naive_policy_gradient_loss(
                raw_rewards_or_advantages=advantages, policy_log_probs=policy_log_probs
            ),
            {},
        )
    assert (
        advantages is not None
    ), "Advantages are required when loss type is 'grpo_clip'."
    assert (
        old_log_probs is not None
    ), "Old log probs are required when loss type is 'grpo_clip'."
    assert (
        cliprange is not None
    ), "Clip ranage is required when loss type is 'grpo_clip'."
    return compute_grpo_clip_loss(
        advantages=advantages,
        policy_log_probs=policy_log_probs,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )


def masked_mean(
    tensor: Float[torch.Tensor, "..."],
    mask: Float[torch.Tensor, "..."],
    dim: int | None = None,
) -> Float[torch.Tensor, "..."]:
    """Computes the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """
    assert tensor.shape == mask.shape, "Tensor and mask must have the same shape"
    return torch.sum(tensor * mask, dim=dim) / torch.sum(mask, dim=dim)


def grpo_microbatch_train_step(
    policy_log_probs: Float[torch.Tensor, "batch_size seq_len"],
    response_mask: Float[torch.Tensor, "batch_size seq_len"],
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    use_length_normalization: bool,
    max_sampling_length: int,
    raw_rewards: Float[torch.Tensor, "batch_size 1"] | None = None,
    advantages: Float[torch.Tensor, "batch_size 1"] | None = None,
    old_log_probs: Float[torch.Tensor, "batch_size seq_len"] | None = None,
    cliprange: float | None = None,
) -> tuple[Float[torch.Tensor, ""], dict[str, Any]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    This function computes the policy gradient loss and backprop its gradients
    for a microbatch.
    `loss.backward()` is called inside this function.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length):
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
            the type of loss function to use.
        use_length_normalization: bool, whether to use length normalization for rewards.
        max_sampling_length: int, the maximum sampling length. This is for normalizing per-token
            loss when `use_length_normalization` is False.
        raw_rewards: torch.Tensor of shape (batch_size, 1):
            the raw rewards for each rollout response.
        advantages: torch.Tensor of shape (batch_size, 1):
            the advantages for each rollout response.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, Any]]:
            the policy gradient loss and its metadata.
    """
    batch_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    if use_length_normalization:
        loss = (
            masked_mean(tensor=batch_loss, mask=response_mask)
            / gradient_accumulation_steps
        )
    else:
        loss = (
            torch.mean(
                sft_helpers.masked_normalize(
                    tensor=batch_loss,
                    mask=response_mask,
                    normalize_constant=max_sampling_length,
                    dim=-1,
                )
            )
            / gradient_accumulation_steps
        )
    loss.backward()
    return loss, metadata


def sample_grpo_rollouts(
    *,
    model: vllm.LLM,
    sampling_params: vllm.SamplingParams,
    questions: list[str],
    ground_truth_answers: list[str],
    prompt_fn: Callable[[list[str]], list[str]],
) -> tuple[list[str], list[str], list[str]]:
    """Samples GRPO rollouts.

    Args:
        model: vllm.LLM, the language model.
        sampling_params: vllm.SamplingParams, the sampling parameters.
        questions: list[str], the questions.
        ground_truth_answers: list[str], the ground truth answers.
        prompt_fn: Callable[[str], str], the prompt function.

    Returns:
        tuple[list[str], list[str], list[str]]:
            list[str]:
                The repeated model input prompts.
            list[str]:
                The model responses.
            list[str]:
                The repeated ground truth answers.
            list[Float[torch.Tensor, "seq_len"]]:
                The log probabilities.
    """
    assert len(questions) == len(ground_truth_answers), (
        f"Questions length {len(questions)} doesn't match ground truth answers length "
        f"{len(ground_truth_answers)}"
    )
    input_prompts = prompt_fn(questions)
    raw_model_responses = model.generate(input_prompts, sampling_params)
    repeated_model_input_prompts = []
    model_responses = []
    repeated_ground_truth_answers = []
    for model_response, ground_truth_answer in zip(
        raw_model_responses, ground_truth_answers
    ):
        for rollout in model_response.outputs:
            repeated_model_input_prompts.append(model_response.prompt)
            model_responses.append(rollout.text)
            repeated_ground_truth_answers.append(ground_truth_answer)
    return (
        repeated_model_input_prompts,
        model_responses,
        repeated_ground_truth_answers,
    )


def randomly_sample_batch(
    ds: datasets.Dataset,
    num_datapoints: int,
) -> datasets.Dataset:
    """Randomly samples a batch of datapoints from the dataset."""
    if num_datapoints == -1:
        return ds
    return ds.select(
        np.random.choice(range(len(ds)), size=num_datapoints, replace=False)
    )


def run_evaluation(
    *,
    vllm_old_model: vllm.LLM,
    evaluation_sampling_params: vllm.SamplingParams,
    policy_model: transformers.PreTrainedModel,
    train_config: grpo_train_config.GrpoTrainConfig,
    eval_ds: datasets.Dataset,
    task_name: str,
    prompt_template: str,
    grpo_step: int,
    epoch: int,
    microbatch_idx: int,
    wandb_run: Any,
) -> None:
    """Run evaluation on the model."""
    logging.info(
        f"Evaluating policy model on validation set at GRPO step "
        f"{grpo_step}/{train_config.n_grpo_steps}, "
        f"epoch {epoch}/{train_config.epochs_per_rollout_batch}, "
        f"microbatch {microbatch_idx + 1}/{train_config.n_microbatches_per_rollout_batch}..."
    )
    eval_ds_batch = randomly_sample_batch(
        ds=eval_ds,
        num_datapoints=train_config.evaluation_sample_size,
    )
    vllm_utils.load_policy_into_vllm_instance(
        policy=policy_model,
        vllm_instance=vllm_old_model,
    )
    if task_name == "gsm8k":
        eval_result = eval_utils.evaluate_on_gsm8k(
            vllm_model=vllm_old_model,
            reward_fn=custom_grader.gsm8k_reward_fn,
            model_inputs=data_utils.generate_gsm8k_prompt_from_question_list(
                prompt_template=prompt_template,
                questions=eval_ds_batch["question"],
            ),
            ground_truth_answers=eval_ds_batch["answer"],
            eval_sampling_params=evaluation_sampling_params,
        )
    elif task_name == "countdown":
        eval_result = eval_utils.evaluate_on_countdown(
            vllm_model=vllm_old_model,
            reward_fn=custom_grader.countdown_reward_fn,
            model_inputs=data_utils.generate_countdown_prompt_from_nums_target_lists(
                prompt_template=prompt_template,
                nums_list=eval_ds_batch["nums"],
                target_list=eval_ds_batch["target"],
            ),
            nums_list=eval_ds_batch["nums"],
            target_list=eval_ds_batch["target"],
            eval_sampling_params=evaluation_sampling_params,
        )
    else:
        raise ValueError(f"Invalid task name: {task_name}")
    logging.info(
        f"Evaluation score: {eval_result.score} at GRPO step "
        f"{grpo_step + 1}/{train_config.n_grpo_steps}, "
        f"epoch {epoch + 1}/{train_config.epochs_per_rollout_batch}, "
        f"microbatch {microbatch_idx + 1}/{train_config.n_microbatches_per_rollout_batch}"
    )
    wandb_run.log(
        {
            "eval/reward": eval_result.score,
            "eval/prompt_gt_correct_sample": eval_utils.get_sample_eval_result_table(
                eval_result=eval_result,
                max_num_samples=10,
                correct_samples=True,
                incorrect_samples=False,
            ),
            "eval/prompt_gt_incorrect_sample": eval_utils.get_sample_eval_result_table(
                eval_result=eval_result,
                max_num_samples=10,
                correct_samples=False,
                incorrect_samples=True,
            ),
        }
    )


def get_microbatch_and_move_to_device(
    input_tensor: Float[torch.Tensor, "rollout_batch_size seq_len"],
    microbatch_idx: int,
    microbatch_size: int,
    device: str | None = None,
) -> torch.Tensor:
    """Get a microbatch and move it to the device."""
    microbatch = input_tensor[
        microbatch_idx * microbatch_size : (microbatch_idx + 1) * microbatch_size
    ]
    if device is not None and device.startswith("cuda"):
        microbatch = microbatch.pin_memory().to(device=device, non_blocking=True)
    return microbatch


def grpo_train_one_epoch(
    *,
    policy_model: transformers.PreTrainedModel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    task_name: str,
    all_input_ids: Int[torch.Tensor, "rollout_batch_size seq_len"],
    all_labels: Int[torch.Tensor, "rollout_batch_size seq_len"],
    all_response_mask: Int[torch.Tensor, "rollout_batch_size seq_len"],
    all_old_log_probs: Float[torch.Tensor, "rollout_batch_size seq_len"],
    all_raw_rewards: Float[torch.Tensor, "rollout_batch_size"],
    all_format_rewards: Float[torch.Tensor, "rollout_batch_size"],
    all_answer_rewards: Float[torch.Tensor, "rollout_batch_size"],
    all_advantages: Float[torch.Tensor, "rollout_batch_size"],
    vllm_old_model: vllm.LLM,
    evaluation_sampling_params: vllm.SamplingParams,
    prompt_template: str,
    train_config: grpo_train_config.GrpoTrainConfig,
    eval_ds: datasets.Dataset,
    wandb_run: Any,
    grpo_step: int,
    epoch: int,
    global_update_count: int,
) -> tuple[int, bool]:
    """Train the policy model for one epoch.

    Returns:
        int: The global update count.
        bool: whether to early stop the GRPO step.
    """
    for microbatch_idx in tqdm.tqdm(
        range(train_config.n_microbatches_per_rollout_batch),
        desc=(
            f"Training microbatches in GRPO step {grpo_step}/{train_config.n_grpo_steps} "
            f"epoch {epoch}/{train_config.epochs_per_rollout_batch}"
        ),
    ):
        microbatch_input_ids = get_microbatch_and_move_to_device(
            input_tensor=all_input_ids,
            microbatch_idx=microbatch_idx,
            microbatch_size=train_config.microbatch_size,
            device="cuda:0",
        )
        microbatch_labels = get_microbatch_and_move_to_device(
            input_tensor=all_labels,
            microbatch_idx=microbatch_idx,
            microbatch_size=train_config.microbatch_size,
            device="cuda:0",
        )
        microbatch_response_mask = get_microbatch_and_move_to_device(
            input_tensor=all_response_mask,
            microbatch_idx=microbatch_idx,
            microbatch_size=train_config.microbatch_size,
            device="cuda:0",
        )
        microbatch_old_log_probs = get_microbatch_and_move_to_device(
            input_tensor=all_old_log_probs,
            microbatch_idx=microbatch_idx,
            microbatch_size=train_config.microbatch_size,
            device="cuda:0",
        )
        microbatch_raw_rewards = get_microbatch_and_move_to_device(
            input_tensor=all_raw_rewards,
            microbatch_idx=microbatch_idx,
            microbatch_size=train_config.microbatch_size,
            device="cuda:0",
        )
        microbatch_format_rewards = get_microbatch_and_move_to_device(
            input_tensor=all_format_rewards,
            microbatch_idx=microbatch_idx,
            microbatch_size=train_config.microbatch_size,
        )
        microbatch_answer_rewards = get_microbatch_and_move_to_device(
            input_tensor=all_answer_rewards,
            microbatch_idx=microbatch_idx,
            microbatch_size=train_config.microbatch_size,
        )
        microbatch_advantages = get_microbatch_and_move_to_device(
            input_tensor=all_advantages,
            microbatch_idx=microbatch_idx,
            microbatch_size=train_config.microbatch_size,
            device="cuda:0",
        )
        policy_log_probs_dict = sft_helpers.get_response_log_probs(
            model=policy_model,
            input_ids=microbatch_input_ids,
            labels=microbatch_labels,
            return_token_entropy=True,
        )
        loss, metadata = grpo_microbatch_train_step(
            policy_log_probs=policy_log_probs_dict["log_probs"],
            response_mask=microbatch_response_mask,
            gradient_accumulation_steps=train_config.gradient_accumulation_steps,
            loss_type=train_config.loss_type,
            use_length_normalization=train_config.use_length_normalization,
            max_sampling_length=train_config.sampling_max_tokens,
            raw_rewards=microbatch_raw_rewards,
            advantages=microbatch_advantages,
            old_log_probs=microbatch_old_log_probs,
            cliprange=train_config.cliprange,
        )
        if microbatch_idx % train_config.log_training_metrics_every_n_microbatches == 0:
            gradient_norm_before_clipping = torch.nn.utils.clip_grad_norm_(
                parameters=policy_model.parameters(),
                max_norm=train_config.gradient_clip,
            )
            log_dict = {
                "train/loss": loss.detach().item(),
                "train/average_token_entropy": masked_mean(
                    tensor=policy_log_probs_dict["token_entropy"],
                    mask=microbatch_response_mask,
                )
                .detach()
                .item(),
                "train/average_response_length": microbatch_response_mask.sum(dim=-1)
                .to(torch.float32)
                .mean()
                .detach()
                .item(),
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "train/gradient_norm": gradient_norm_before_clipping.detach().item(),
                "train/format_reward": microbatch_raw_rewards.mean().detach().item(),
                "train/answer_reward": microbatch_raw_rewards.mean().detach().item(),
                "train/reward": microbatch_raw_rewards.mean().detach().item(),
                "train/advantage": microbatch_advantages.mean().detach().item(),
            }
            del gradient_norm_before_clipping
            if train_config.loss_type == "grpo_clip":
                with torch.no_grad():
                    log_dict["train/clip_fraction"] = (
                        masked_mean(
                            tensor=metadata["is_token_clipped"],
                            mask=microbatch_response_mask,
                        )
                        .detach()
                        .item()
                    )
                    log_dict["train/approx_kl_divergence"] = (
                        masked_mean(
                            tensor=metadata["approx_kl_divergence"],
                            mask=microbatch_response_mask,
                        )
                        .detach()
                        .item()
                    )
            wandb_run.log(log_dict)
            del log_dict

        if (microbatch_idx + 1) % train_config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                parameters=policy_model.parameters(),
                max_norm=train_config.gradient_clip,
            )
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_update_count += 1
            if global_update_count % train_config.validation_every_n_updates == 0:
                run_evaluation(
                    vllm_old_model=vllm_old_model,
                    evaluation_sampling_params=evaluation_sampling_params,
                    policy_model=policy_model,
                    train_config=train_config,
                    prompt_template=prompt_template,
                    eval_ds=eval_ds,
                    task_name=task_name,
                    grpo_step=grpo_step,
                    epoch=epoch,
                    microbatch_idx=microbatch_idx,
                    wandb_run=wandb_run,
                )
        if (
            train_config.loss_type == "grpo_clip"
            and train_config.early_stop_kl_divergence_threshold < float("inf")
        ):
            with torch.no_grad():
                microbatch_average_kl_divergence = (
                    masked_mean(
                        tensor=metadata["approx_kl_divergence"],
                        mask=microbatch_response_mask,
                    )
                    .detach()
                    .item()
                )
            if (
                microbatch_average_kl_divergence
                > train_config.early_stop_kl_divergence_threshold
            ):
                logging.info(
                    f"Early stopping because microbatch KL divergence "
                    f"{microbatch_average_kl_divergence} is greater than threshold "
                    f"{train_config.early_stop_kl_divergence_threshold}"
                )
                del (
                    microbatch_input_ids,
                    microbatch_labels,
                    microbatch_response_mask,
                    microbatch_old_log_probs,
                    microbatch_raw_rewards,
                    microbatch_format_rewards,
                    microbatch_answer_rewards,
                    microbatch_advantages,
                    policy_log_probs_dict,
                    loss,
                    metadata,
                    microbatch_average_kl_divergence,
                )
                optimizer.zero_grad()
                return global_update_count, True
        del (
            microbatch_input_ids,
            microbatch_labels,
            microbatch_response_mask,
            microbatch_old_log_probs,
            microbatch_raw_rewards,
            microbatch_format_rewards,
            microbatch_answer_rewards,
            microbatch_advantages,
            policy_log_probs_dict,
            loss,
            metadata,
        )
    return global_update_count, False
