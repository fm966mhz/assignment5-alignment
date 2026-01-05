"""Utility functions for GRPO training."""

from typing import Any, Callable, Literal

import datasets
import torch
import tqdm
import transformers
import vllm

from jaxtyping import Float, Int

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
    """
    rho = torch.exp(policy_log_probs - old_log_probs)
    first_term = advantages * rho
    second_term = advantages * torch.clip(rho, min=1 - cliprange, max=1 + cliprange)
    grpo_per_token_clipped_loss = -torch.minimum(first_term, second_term)
    with torch.no_grad():
        is_token_clipped = second_term > first_term
    return (grpo_per_token_clipped_loss, {"is_token_clipped": is_token_clipped})


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
    loss = (
        masked_mean(tensor=batch_loss, mask=response_mask) / gradient_accumulation_steps
    )
    loss.backward()
    return loss, metadata


def _get_log_probs_from_vllm_responses(
    completion_output: vllm.CompletionOutput,
) -> Float[torch.Tensor, "seq_len"]:
    """Get the log probabilities from a vLLM completion output."""
    log_probs = []
    for token_id, log_probs_dict in zip(
        completion_output.token_ids,
        completion_output.logprobs,
    ):
        log_probs.append(log_probs_dict[token_id].logprob)
    return torch.tensor(log_probs)


def sample_grpo_rollouts(
    *,
    model: vllm.LLM,
    sampling_params: vllm.SamplingParams,
    questions: list[str],
    ground_truth_answers: list[str],
    prompt_fn: Callable[[list[str]], list[str]],
) -> tuple[list[str], list[str], list[str], list[Float[torch.Tensor, "seq_len"]]]:
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
    assert (
        sampling_params.logprobs == 1
    ), "Log probabilities are required for GRPO rollouts"
    input_prompts = prompt_fn(questions)
    raw_model_responses = model.generate(input_prompts, sampling_params)
    repeated_model_input_prompts = []
    model_responses = []
    repeated_ground_truth_answers = []
    log_probs = []
    for model_response, ground_truth_answer in zip(
        raw_model_responses, ground_truth_answers
    ):
        for rollout in model_response.outputs:
            repeated_model_input_prompts.append(model_response.prompt)
            model_responses.append(rollout.text)
            repeated_ground_truth_answers.append(ground_truth_answer)
            log_probs.append(
                _get_log_probs_from_vllm_responses(completion_output=rollout)
            )
    return (
        repeated_model_input_prompts,
        model_responses,
        repeated_ground_truth_answers,
        log_probs,
    )


def get_microbatch_and_move_to_device(
    input_tensor: torch.Tensor | list[Float[torch.Tensor, "seq_len"]],
    microbatch_idx: int,
    microbatch_size: int,
    device: str | None = None,
) -> torch.Tensor:
    """Get a microbatch and move it to the device."""
    microbatch = input_tensor[
        microbatch_idx * microbatch_size : (microbatch_idx + 1) * microbatch_size
    ]
    if isinstance(microbatch, list):
        microbatch = torch.nn.utils.rnn.pad_sequence(
            sequences=microbatch,
            batch_first=True,
            padding_value=0,
            padding_side="right",
        )
    if device is not None and device.startswith("cuda"):
        microbatch = microbatch.pin_memory().to(device=device, non_blocking=True)
    return microbatch


def grpo_train_one_epoch(
    *,
    policy_model: transformers.PreTrainedModel,
    optimizer: torch.optim.Optimizer,
    all_input_ids: Int[torch.Tensor, "rollout_batch_size seq_len"],
    all_labels: Int[torch.Tensor, "rollout_batch_size seq_len"],
    all_response_mask: Int[torch.Tensor, "rollout_batch_size seq_len"],
    all_old_log_probs: list[Float[torch.Tensor, "seq_len"]],
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
) -> None:
    """Train the policy model for one epoch."""
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
                log_dict["train/clip_fraction"] = (
                    masked_mean(
                        tensor=metadata["is_token_clipped"],
                        mask=microbatch_response_mask,
                    )
                    .detach()
                    .item()
                )
            wandb_run.log(log_dict)

        if (microbatch_idx + 1) % train_config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                parameters=policy_model.parameters(),
                max_norm=train_config.gradient_clip,
            )
            optimizer.step()
            optimizer.zero_grad()

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
        if (microbatch_idx + 1) % (
            train_config.validation_every_n_updates
            * train_config.gradient_accumulation_steps
        ) == 0:
            vllm_utils.load_policy_into_vllm_instance(
                policy=policy_model,
                vllm_instance=vllm_old_model,
            )
            eval_result = eval_utils.evaluate_on_gsm8k(
                vllm_model=vllm_old_model,
                reward_fn=custom_grader.gsm8k_reward_fn,
                model_inputs=data_utils.generate_gsm8k_prompt_from_question_list(
                    prompt_template=prompt_template,
                    questions=eval_ds["question"],
                ),
                ground_truth_answers=eval_ds["answer"],
                eval_sampling_params=evaluation_sampling_params,
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
