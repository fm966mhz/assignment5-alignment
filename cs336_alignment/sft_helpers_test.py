"""Unit tests for helpers for SFT and RL."""

import torch
import transformers
import vllm

from cs336_alignment import custom_grader
from cs336_alignment import data_utils
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


def test_sample_expert_rollouts():
    model = vllm.LLM(
        model="Qwen/Qwen2.5-Math-1.5B",
        device="cuda:0",
        seed=42,
        dtype="bfloat16",
        enable_prefix_caching=True,
        gpu_memory_utilization=0.85,
    )
    sampling_params = vllm.SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        min_tokens=10,
        stop=["</answer>"],
        n = 10,
        include_stop_str_in_output=True,
    )
    input_prompts = data_utils.generate_gsm8k_prompt_from_question_list(
        prompt_template="""
You are an helpful Assistant having a conversation with a User. The user asks you a question, and your job is to solve it.

You will first think about the reasoning process step by step and then provides the User with the final answer. Your reasoning process should be enclosed within the XML tags `<think>` and `</think>`, and the finnal answer within the XML tags `<answer>` and `</answer>`. In other words, format your response in the following way:

```
<think>
Your thinking process goes here
</think>
<answer> Your final answer goes here </answer>
```

Here is the Conversation:
User: {question}
Assistant:
<think>
        """,
        questions=["Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"],
    )
    ground_truth_answers = ["Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72"]
    print(
        sft_helpers.sample_expert_rollouts(
            model=model,
            sampling_params=sampling_params,
            input_prompts=input_prompts,
            ground_truth_answers=ground_truth_answers,
            reward_fn=custom_grader.gsm8k_reward_fn,
        )
    )