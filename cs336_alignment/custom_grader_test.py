"""Tests for custom grader functionality."""

from cs336_alignment import custom_grader


def test_extract_gsm8k_ground_truth_answer():
    """Tests extraction of GSM8K ground truth answers."""
    output = "Some explanation here. #### 42.0"
    answer = custom_grader.extract_gsm8k_ground_truth_answer(output)
    assert answer == 42.0

    output_invalid = "No delimiter here."
    try:
        custom_grader.extract_gsm8k_ground_truth_answer(output_invalid)
    except ValueError as e:
        assert str(e) == "Output does not contain expected '####' delimiter."

    output_non_float = "Explanation #### not_a_number"
    try:
        custom_grader.extract_gsm8k_ground_truth_answer(output_non_float)
    except ValueError as e:
        assert str(e) == "Extracted answer is not a valid float."


def test_gsm8k_reward_fn():
    """Tests GSM8K reward function."""
    reward = custom_grader.gsm8k_reward_fn(
        "Some reasoning... </think> <answer>42.0</answer>", "Explanation #### 42.0"
    )
    assert reward == {
        "format_reward": 1.0,
        "answer_reward": 1.0,
        "reward": 1.0,
    }

    reward = custom_grader.gsm8k_reward_fn(
        "Some reasoning... </think> <answer> Therefore, the final answer is 42. </answer>",
        "Explanation #### 42.0",
    )
    assert reward == {
        "format_reward": 1.0,
        "answer_reward": 1.0,
        "reward": 1.0,
    }

    reward = custom_grader.gsm8k_reward_fn(
        "Some reasoning... </think> <answer> Therefore, 42 is the final answer. </answer>",
        "Explanation #### 42.0",
    )
    assert reward == {
        "format_reward": 1.0,
        "answer_reward": 1.0,
        "reward": 1.0,
    }

    reward = custom_grader.gsm8k_reward_fn(
        "Some reasoning... </think> <answer> Therefore, 42 is the final answer. "
        "43 is not. </answer>",
        "Explanation #### 42.0",
    )
    assert reward != {
        "format_reward": 1.0,
        "answer_reward": 1.0,
        "reward": 1.0,
    }

    reward = custom_grader.gsm8k_reward_fn(
        "Some reasoning... </think> <answer>42.0</answer>",
        "Explanation #### 43.0",
    )
    assert reward == {
        "format_reward": 1.0,
        "answer_reward": 0.0,
        "reward": 0.0,
    }


def test_validate_countdown_answer():
    """Tests validation of Countdown task answers."""
    # Not all numbers are used
    assert not custom_grader.validate_countdown_answer(
        "25 + 75 = 100", [25, 75, 50, 10], 100
    )

    # Number used twice.
    assert not custom_grader.validate_countdown_answer(
        "25 + 75 + 75 = 175", [25, 75], 175
    )

    # Correct answer.
    assert custom_grader.validate_countdown_answer(
        "25 + 75 - 50 = 50", [25, 75, 50], 50
    )

    # 5 is not in the list of numbers.
    assert not custom_grader.validate_countdown_answer(
        "10 - (25 + 75) / 50 + 5 = 13", [25, 75, 50, 10], 13
    )

    # Expression does not evaluate to target.
    assert not custom_grader.validate_countdown_answer(
        "10 - (25 + 75) / 50 + 5 * 2 = 13", [25, 75, 50, 10, 5, 2], 13
    )

    # Correct expression and answer.
    assert custom_grader.validate_countdown_answer(
        "10 - (25 + 75) / 50 + 5 * 2 = 18", [25, 75, 50, 10, 5, 2], 18
    )

    # Invalid integer in answer part.
    assert not custom_grader.validate_countdown_answer(
        "30 + 40 = 80", [25, 75, 50, 10], 100
    )

    # Invalid integer in expression part.
    assert not custom_grader.validate_countdown_answer(
        "25 + abc = 100", [25, 75, 50, 10], 100
    )


def test_countdown_reward_fn():
    """Tests Countdown reward function."""
    reward = custom_grader.countdown_reward_fn(
        "Some reasoning... </think> <answer>25 + 75 = 100</answer>", [25, 75], 100
    )
    assert reward == {
        "format_reward": 1.0,
        "answer_reward": 1.0,
        "reward": 1.0,
    }

    reward = custom_grader.countdown_reward_fn(
        "Some reasoning... </think> <answer>25 + 75 + 75 = 175</answer>", [25, 75], 175
    )
    assert reward == {
        "format_reward": 1.0,
        "answer_reward": 0.0,
        "reward": 0.0,
    }

    reward = custom_grader.countdown_reward_fn(
        "Some reasoning... </think> <answer>25 + 75 + 75 = 175</answer>",
        [25, 75, 75],
        175,
    )
    assert reward == {
        "format_reward": 1.0,
        "answer_reward": 1.0,
        "reward": 1.0,
    }

    reward = custom_grader.countdown_reward_fn(
        "Some reasoning... </think> <answer>10 - (25 + 75) / 50 + 5 * 2 = 13</answer>",
        [25, 75, 50, 10, 5, 2],
        13,
    )
    assert reward == {
        "format_reward": 1.0,
        "answer_reward": 0.0,
        "reward": 0.0,
    }

    reward = custom_grader.countdown_reward_fn(
        "Some reasoning... </think> <answer>10 - (25 + 75) / 50 + 5 * 2 = 18</answer>",
        [25, 75, 50, 10, 5, 2],
        18,
    )
    assert reward == {
        "format_reward": 1.0,
        "answer_reward": 1.0,
        "reward": 1.0,
    }
