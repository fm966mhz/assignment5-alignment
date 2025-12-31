"""Unit tests for helpers for SFT and RL."""

import torch
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


def test_generate_model_log():
    model = transformers.AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B"
    ).to("cuda")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B", dtype=torch.float16
    )
    generation_config = transformers.GenerationConfig.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B"
    )
    generation_config.max_length = 128
    generation_config.return_dict_in_generate = True
    generation_config.do_sample = True

    def _reward_fn(
        model_output: str,
        ground_truth_answer: str,
    ) -> dict[str, float]:
        del model_output
        int_ans = int(ground_truth_answer)
        return {
            "format_reward": 1.0,
            "answer_reward": int_ans % 2,
            "reward": int_ans % 2,
        }

    print(
        sft_helpers.generate_model_log(
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            input_prompts=[
                "A test of input prompt",
                "The capitol of the US is Washington D.C.",
                "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            ],
            ground_truth_answers=["1", "2", "3"],
            reward_fn=_reward_fn,
            num_trials=1,
            device="cuda",
        )
    )
