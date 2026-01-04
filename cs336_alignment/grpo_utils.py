"""Utility functions for GRPO training."""

from typing import Any, Callable, Literal

import torch

from jaxtyping import Float


def compute_group_normalized_rewards(  # pylint: disable=too-many-arguments
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    *,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[
    Float[torch.Tensor, "rollout_batch_size"],
    Float[torch.Tensor, "rollout_batch_size"],
    dict[str, Any],
]:
    """Compute group-normalized rewards for a batch of rollout responses.

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
            dict[str, float]: metadata for the rewards of the rollout batch.
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
    raw_rewards_per_question = []
    for rollout_response, ground_truth in zip(
        rollout_responses, repeated_ground_truths
    ):
        reward_dict = reward_fn(rollout_response, ground_truth)
        raw_rewards_per_question.append(reward_dict.get("reward", 0.0))
        if len(raw_rewards_per_question) % group_size == 0:
            all_raw_rewards.append(raw_rewards_per_question)
            raw_rewards_per_question = []

    all_raw_rewards = torch.tensor(all_raw_rewards)
    group_normalized_rewards = all_raw_rewards - torch.mean(
        all_raw_rewards, dim=-1, keepdim=True
    )
    if normalize_by_std:
        group_normalized_rewards = group_normalized_rewards / (
            torch.std(all_raw_rewards, dim=-1, keepdim=True) + advantage_eps
        )
    return (group_normalized_rewards.flatten(), all_raw_rewards.flatten(), {})


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
    """
    rho = torch.exp(policy_log_probs - old_log_probs)
    first_term = advantages * rho
    second_term = advantages * torch.clip(rho, min=1 - cliprange, max=1 + cliprange)
    grpo_per_token_clipped_loss = -torch.minimum(first_term, second_term)
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
