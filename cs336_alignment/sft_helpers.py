"""Helpers for SFT and RL."""

import torch
import transformers


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
                    False,
                ]
                * len(encoded_sequence["input_ids"])
                + [
                    True,
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
        padding_value=False,
        padding_side="right",
    ).to(torch.bool)
    return {
        "input_ids": all_sequences[:, :-1],
        "labels": all_sequences[:, 1:],
        "response_mask": all_response_masks[:, 1:],
    }
