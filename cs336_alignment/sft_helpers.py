"""Helpers for SFT and RL."""

import dataclasses
import itertools
import math

from typing import Callable

import einops
import torch
import torch.nn.functional as F
import transformers
import vllm

from jaxtyping import Float, Int
from torch import optim


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """Tokenizes prompts and outputs using the provided tokenizer.

    Args:
        prompt_strs: List of prompt strings.
        output_strs: List of output strings.
        tokenizer: The tokenizer to use.

    Returns:
        A dictionary with tokenized 'input_ids', 'labels', and 'response_mask'.
    """
    assert len(prompt_strs) == len(
        output_strs
    ), f"Numbers of prompt strs and output strs mismatch: {len(prompt_strs)} vs {len(output_strs)}"
    all_sequences = []
    all_response_masks = []
    for prompt, output in zip(prompt_strs, output_strs):
        encoded_sequence = tokenizer(prompt, text_target=output, padding=False)
        all_sequences.append(
            torch.tensor(encoded_sequence["input_ids"] + encoded_sequence["labels"])
        )
        all_response_masks.append(
            torch.tensor(
                [
                    0,
                ]
                * len(encoded_sequence["input_ids"])
                + [
                    1,
                ]
                * len(encoded_sequence["labels"])
            )
        )
    all_sequences = torch.nn.utils.rnn.pad_sequence(
        sequences=all_sequences,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
        padding_side="right",
    )
    all_response_masks = torch.nn.utils.rnn.pad_sequence(
        sequences=all_response_masks,
        batch_first=True,
        padding_value=0,
        padding_side="right",
    ).to(torch.int8)
    return {
        "input_ids": all_sequences[:, :-1],
        "labels": all_sequences[:, 1:],
        "response_mask": all_response_masks[:, 1:],
    }


def compute_entropy(
    logits: Float[torch.Tensor, "batch_size seq_len vocab_size"],
) -> Float[torch.Tensor, "batch_size seq_len"]:
    """Computes the per-token entropy.

    Args:
        logits: Logits tensor of shape (batch_size, seq_len, vocab_size).

    Returns:
        Entropy tensor of shape (batch_size, seq_len).
    """
    return torch.logsumexp(logits, dim=-1) - einops.einsum(
        logits,
        F.softmax(logits, dim=-1),
        "B T V, B T V -> B T",
    )


def get_response_log_probs(
    model: transformers.PreTrainedModel,
    input_ids: Int[torch.Tensor, "batch_size seq_len"],
    labels: Int[torch.Tensor, "batch_size seq_len"],
    return_token_entropy: bool = False,
) -> dict[str, Float[torch.Tensor, "batch_size seq_len"]]:
    """Gets the log probabilities of the response tokens.

    Args:
        model: The language model.
        input_ids: Input IDs tensor of shape (batch_size, seq_len).
        labels: Labels tensor of shape (batch_size, seq_len).
        return_token_entropy: Whether to return per-token entropy.

    Returns:
        A dictionary with 'log_probs' and optionally 'token_entropy'.
    """
    assert (
        input_ids.shape == labels.shape
    ), f"Input ids shape {input_ids.shape} different from labels shape {labels.shape}."
    logits: Float[torch.Tensor, "batch_size seq_len vocab_size"] = model(input_ids)[
        "logits"
    ]
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(dim=-1)).squeeze(
        dim=-1
    )
    output = {"log_probs": log_probs}
    if return_token_entropy:
        # Save memory by not saving intermediate tensor for entropy, which is not used in the
        # backward pass.
        with torch.no_grad():
            output["token_entropy"] = compute_entropy(logits=logits)
    return output


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """Runs masked normalize on tensor."""
    return (
        torch.sum(
            torch.where(
                mask == 1,
                tensor,
                0.0,
            ),
            dim=dim,
        )
        / normalize_constant
    )


def sft_microbatch_train_step(
    policy_log_probs: Float[torch.Tensor, "batch_size seq_len"],
    response_mask: Int[torch.Tensor, "batch_size seq_len"],
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Runs one microbatch step in SFT training.

    `loss.backward()` is called.
    """
    loss = (
        torch.mean(
            masked_normalize(
                tensor=-policy_log_probs,
                mask=response_mask,
                normalize_constant=normalize_constant,
                dim=-1,
            )
        )
        / gradient_accumulation_steps
    )
    loss.backward()
    return (loss, {})


def stack_logits(
    model_output_logits: tuple[Float[torch.Tensor, "batch_size vocab_size"]],
) -> Float[torch.Tensor, "batch_size seq_len vocab_size"]:
    """Stacks logits."""
    output = torch.stack(model_output_logits, dim=1)
    return output


@dataclasses.dataclass
class ModelLog:
    """Model log for one input prompt."""

    input_prompt: str
    ground_truth_answer: str
    model_response: list[str]
    reward: list[dict[str, float]]
    average_token_entropy: float
    average_response_length: float
    average_correct_response_length: float
    average_incorrect_response_length: float
    num_correct_responses: int
    num_incorrect_responses: int


def generate_model_log(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizerBase,
    generation_config: transformers.GenerationConfig,
    input_prompts: list[str],
    ground_truth_answers: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    num_trials: int,
    device: str,
    batch_size: int | None = None,
) -> list[ModelLog]:
    """Generates mode logs."""

    def _process_batch(
        prompts_batch: list[str],
        ground_truth_answer_batch: list[str],
    ) -> list[ModelLog]:
        input_dict = tokenizer(prompts_batch, padding=True, return_tensors="pt").to(
            device
        )
        output = [
            ModelLog(
                input_prompt=input_prompt,
                ground_truth_answer=ground_truth_answer,
                model_response=[],
                reward=[],
                average_token_entropy=0.0,
                average_response_length=0.0,
                average_correct_response_length=0.0,
                average_incorrect_response_length=0.0,
                num_correct_responses=0,
                num_incorrect_responses=0,
            )
            for input_prompt, ground_truth_answer in zip(
                prompts_batch, ground_truth_answer_batch
            )
        ]
        with torch.inference_mode():
            for _ in range(num_trials):
                model_output_dict = model.generate(
                    **input_dict,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    do_sample=True,
                    output_logits=True,
                    tokenizer=tokenizer,
                )  # pyright: ignore[reportCallIssue]
                input_and_output_sequences_str = tokenizer.batch_decode(
                    **model_output_dict, skip_special_tokens=True
                )
                model_output_sequences_str = [
                    input_and_output_sequence_str[len(input_sequence_str) :]
                    for input_and_output_sequence_str, input_sequence_str in zip(
                        input_and_output_sequences_str, prompts_batch
                    )
                ]
                input_ids: Int[torch.Tensor, "batch_size seq_len"] = input_dict[
                    "input_ids"
                ]  # pyright: ignore[reportAssignmentType]
                model_response_token_ids = model_output_dict["sequences"][
                    :, input_ids.shape[1] :
                ]
                valid_position_mask: Int[torch.Tensor, "batch_size seq_len"] = (
                    model_response_token_ids != tokenizer.pad_token_id
                ).to(torch.int32)
                response_lengths = torch.sum(valid_position_mask, dim=-1)
                response_per_token_entropies: Float[
                    torch.Tensor, "batch_size seq_len"
                ] = compute_entropy(logits=stack_logits(model_output_dict["logits"]))
                response_average_entropies: list[float] = (
                    torch.sum(
                        response_per_token_entropies * valid_position_mask, dim=-1
                    )
                    / response_lengths
                ).tolist()
                response_lengths = response_lengths.tolist()
                for i, (model_output_str, ground_truth_str) in enumerate(
                    zip(model_output_sequences_str, ground_truth_answer_batch)
                ):
                    reward = reward_fn(model_output_str, ground_truth_str)
                    model_log = output[i]
                    model_log.model_response.append(model_output_str)
                    model_log.reward.append(reward)
                    model_log.average_token_entropy += response_average_entropies[i]
                    model_log.average_response_length += response_lengths[i]
                    if reward["answer_reward"] == 1.0:
                        model_log.average_correct_response_length += response_lengths[i]
                        model_log.num_correct_responses += 1
                    else:
                        model_log.average_incorrect_response_length += response_lengths[
                            i
                        ]
                        model_log.num_incorrect_responses += 1
        for model_log in output:
            model_log.average_token_entropy /= num_trials
            model_log.average_response_length /= num_trials
            if model_log.num_correct_responses == 0:
                model_log.average_correct_response_length = 0
            else:
                model_log.average_correct_response_length /= (
                    model_log.num_correct_responses
                )
            if model_log.num_incorrect_responses == 0:
                model_log.average_incorrect_response_length = 0
            else:
                model_log.average_incorrect_response_length /= (
                    model_log.num_incorrect_responses
                )
        return output

    if batch_size is None:
        return _process_batch(
            prompts_batch=input_prompts,
            ground_truth_answer_batch=ground_truth_answers,
        )
    output = []
    for prompts_batch, ground_truth_answer_batch in zip(
        itertools.batched(input_prompts, n=batch_size),
        itertools.batched(ground_truth_answers, n=batch_size),
    ):
        output.extend(
            _process_batch(
                prompts_batch=list(prompts_batch),
                ground_truth_answer_batch=list(ground_truth_answer_batch),
            )
        )
    return output


def sample_expert_rollouts(
    model: vllm.LLM,
    sampling_params: vllm.SamplingParams,
    input_prompts: list[str],
    ground_truth_answers: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
) -> tuple[list[str], list[str]]:
    """Samples expert rollouts.

    Args:
        model: The language model.
        sampling_params: The sampling parameters.
        input_prompts: The input prompts.
        ground_truth_answers: The ground truth answers. This is from the dataset directly before any
            formatting.
        reward_fn: The reward function.

    Returns:
        A tuple of lists of selected prompts and correct model responses.
    """
    selected_prompts, correct_model_responses = [], []
    model_responses = model.generate(
        input_prompts,
        sampling_params,
    )
    for model_response, ground_truth_answer in zip(
        model_responses, ground_truth_answers
    ):
        for rollout in model_response.outputs:
            reward = reward_fn(rollout.text, ground_truth_answer)
            if reward["reward"] == 1.0:
                selected_prompts.append(model_response.prompt)
                correct_model_responses.append(rollout.text)
    return selected_prompts, correct_model_responses


def get_cosine_lr(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    if it <= cosine_cycle_iters:
        return (
            0.5
            * (
                1
                + math.cos(
                    (it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi
                )
            )
            * (max_learning_rate - min_learning_rate)
            + min_learning_rate
        )
    return min_learning_rate


class CosineLrScheduler(optim.lr_scheduler.LRScheduler):
    """The cosine LR scheduler."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
        last_epoch: int = -1,
    ):
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters
        super().__init__(optimizer=optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:
        return [
            get_cosine_lr(
                it=self.last_epoch,
                max_learning_rate=self.max_learning_rate,
                min_learning_rate=self.min_learning_rate,
                warmup_iters=self.warmup_iters,
                cosine_cycle_iters=self.cosine_cycle_iters,
            )
            for _ in self.optimizer.param_groups
        ]
