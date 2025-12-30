"""Unit tests for data_utils.py."""

import datasets
import torch

from cs336_alignment import data_utils


def test_generate_gsm8k_prompt_collate():
    """Tests GSM8K prompt collation."""
    prompt_template = "Q: {question}\nA:"

    test_ds = datasets.Dataset.from_dict(
        {
            "question": [
                "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
                "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
            ],
            "answer": [
                "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72",
                "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10",
            ],
        }
    )
    test_ds_loader = torch.utils.data.DataLoader(
        test_ds,  # pyright: ignore[reportArgumentType]
        batch_size=2,
        collate_fn=lambda batch: data_utils.generate_gsm8k_prompt_collate(
            prompt_template, batch  # pyright: ignore[reportArgumentType]
        ),
    )

    for batch in test_ds_loader:
        assert batch == {
            "prompt": [
                "Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? \n Please give the final answer as a single number.\nA:",
                "Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn? \n Please give the final answer as a single number.\nA:",
            ],
            "answer": [
                "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72",
                "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10",
            ],
        }


def test_generate_countdown_prompt_collate():
    """Tests Countdown prompt collation."""
    prompt_template = "Q: {question}\nA:"

    test_ds = datasets.Dataset.from_dict(
        {"target": [38, 13], "nums": [[22, 27, 11], [78, 61, 60, 5]]}
    )
    test_ds_loader = torch.utils.data.DataLoader(
        test_ds,  # pyright: ignore[reportArgumentType]
        batch_size=2,
        collate_fn=lambda batch: data_utils.generate_countdown_prompt_collate(
            prompt_template, batch  # pyright: ignore[reportArgumentType]
        ),
    )

    for batch in test_ds_loader:
        assert batch == {
            "prompts": [
                "Q: You're given the following numbers: [22, 27, 11]. Use addition ('+'), subtraction ('-'), multiplication ('*'), division ('/') and paraentheses ('()') to reach the target number: 38. You need to use each number exactly once. You can use the operations as many times as you want.\nGive your final answer as a single arithmetic expression.\nA:",
                "Q: You're given the following numbers: [78, 61, 60, 5]. Use addition ('+'), subtraction ('-'), multiplication ('*'), division ('/') and paraentheses ('()') to reach the target number: 13. You need to use each number exactly once. You can use the operations as many times as you want.\nGive your final answer as a single arithmetic expression.\nA:",
            ],
            "nums": [[22, 27, 11], [78, 61, 60, 5]],
            "target": [38, 13],
        }
