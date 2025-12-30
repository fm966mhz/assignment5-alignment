"""Helpers for SFT and RL."""

import einops
import torch
import torch.nn.functional as F
import transformers

from jaxtyping import Float, Int


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
    """Computes the per-token entropy."""
    return torch.logsumexp(logits, dim=-1) - einops.einsum(
        logits,
        F.softmax(logits, dim=-1),
        "B T V, B T V -> B T",
    )


def get_response_log_probs(
    model: transformers.PreTrainedModel,
    input_ids: Int[torch.Tensor, "batch_size seq_len"],
    labels: Int[torch.Tensor, "batch_size seq_len"],
    return_toek_entropy: bool = False,
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
    if return_toek_entropy:
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
