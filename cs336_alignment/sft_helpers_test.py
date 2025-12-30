"""Unit tests for helpers for SFT and RL."""

import transformers

from cs336_alignment import sft_helpers


def test_check_pad_token_id():
    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
    print(f"Pad token: [{tokenizer.pad_token}]")
    print(f"Pad token ID: [{tokenizer.pad_token_id}]")

    prompt_strs = [
        "A test of input prompt",
        "The capitol of the US is Washington D.C.",
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    ]
    target_strs = [
        " followed by another sequence.",
        " The capitol of China is Beijing.",
        " Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72",
    ]
    encoded_prompts = tokenizer(
        prompt_strs, text_target=target_strs, padding=True, return_tensors="pt"
    )
    print(encoded_prompts)
    print(
        tokenizer(
            prompt_strs[0],
            text_target=target_strs[0],
            padding=False,
            # return_tensors="pt",
        )
    )


def test_tokenize_prompt_and_output():
    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
    prompt_strs = [
        "A test of input prompt",
        "The capitol of the US is Washington D.C.",
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    ]
    target_strs = [
        " followed by another sequence.",
        " The capitol of China is Beijing.",
        " Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72",
    ]
    print(
        sft_helpers.tokenize_prompt_and_output(
            prompt_strs=prompt_strs, output_strs=target_strs, tokenizer=tokenizer
        )
    )
