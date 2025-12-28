"""Custom grader for alignment assignments."""

import regex as re

from cs336_alignment import drgrpo_grader


def extract_gsm8k_ground_truth_answer(output: str) -> float:
    """Extracts answers from GSM8K model output.

    Args:
        output (str): The model output containing answers.

    Returns:
        float: The extracted answer as a float.
    """
    parts = output.strip().split("####")
    if len(parts) < 2:
        raise ValueError("Output does not contain expected '####' delimiter.")
    answer_part = parts[1].strip()
    try:
        answer = float(answer_part)
    except ValueError as e:
        raise ValueError("Extracted answer is not a valid float.") from e
    return answer


_MATCH_AND_EXTRACT_MODEL_ANSWER_PATTERN = re.compile(
    r"(?s).*?<\/think>\s*<answer>(.*?)<\/answer>"
)


def match_and_extract_model_answer(output: str) -> str | None:
    """Matches and extracts the model answer from GSM8K output.

    Args:
        output (str): The model output containing answers.

    Returns:
        str | None: The extracted model answer if found, otherwise None.
    """
    match = _MATCH_AND_EXTRACT_MODEL_ANSWER_PATTERN.match(output)
    if match:
        return match.group(1).strip()
    return None


def gsm8k_reward_fn(
    model_output: str, ground_truth: str, fast: bool = False
) -> dict[str, float]:
    """Computes reward for GSM8K model output.

    Args:
        model_output (str): The model output containing answers.
        ground_truth_answer (str): The ground truth answer to compare against.
        fast (bool): Whether to use fast grading.

    Returns:
        dict[str, float]: A dictionary with reward information.
    """
    try:
        ground_truth_answer_float = extract_gsm8k_ground_truth_answer(ground_truth)
    except ValueError as e:
        raise ValueError("Invalid ground truth answer format.") from e

    model_answer = match_and_extract_model_answer(model_output)
    if model_answer is None:
        return {
            "format_reward": 0.0,
            "anwer_reward": 0.0,
            "reward": 0.0,
        }

    if "\\boxed" in model_answer:
        model_answer = drgrpo_grader.extract_answer(model_answer)
        if model_answer is None:
            return {"format_reward": 1.0, "answer_reward": 0.0, "reward": 0.0}
    if drgrpo_grader.grade(model_answer, str(ground_truth_answer_float), fast=fast):
        return {"format_reward": 1.0, "answer_reward": 1.0, "reward": 1.0}
    else:
        return {"format_reward": 1.0, "answer_reward": 0.0, "reward": 0.0}


_COUNTDOWN_EXPR_SPLIT_PATTERN = r"\+|\-|\*|/|\(|\)"


def validate_countdown_answer(output: str, nums: list[int], target: int) -> bool:
    """Validates the answer for the Countdown task.

    Args:
        output (str): The model output containing the answer.
        target (int | None): The target answer to validate against.

    Returns:
        bool: True if the answer is correct, False otherwise.
    """
    if "=" in output:
        expr, answer_part = output.split("=")
        answer_part = answer_part.strip()
        try:
            answer = int(answer_part)
        except ValueError:
            return False
        if answer != target:
            return False
    else:
        expr = output
        answer_part = ""
    numbers_used = []
    for expr_token in re.split(_COUNTDOWN_EXPR_SPLIT_PATTERN, expr):
        expr_token = expr_token.strip()
        if not expr_token:
            continue
        try:
            number = int(expr_token)
        except ValueError:
            return False
        numbers_used.append(number)
    numbers_used.sort()
    nums_sorted = sorted(nums)
    if numbers_used != nums_sorted:
        return False
    expr_result = eval(expr)  # pylint: disable=eval-used
    return expr_result == target


def countdown_reward_fn(
    model_output: str, nums: list[int], target: int
) -> dict[str, float]:
    """Computes reward for Countdown model output.

    Args:
        model_output (str): The model output containing answers.
        nums (list[int]): The list of numbers provided for the task.
        target (int): The target number to reach.

    Returns:
        dict[str, float]: A dictionary with reward information.
    """
    model_answer = match_and_extract_model_answer(model_output)
    if model_answer is None:
        return {
            "format_reward": 0.0,
            "answer_reward": 0.0,
            "reward": 0.0,
        }

    if validate_countdown_answer(model_answer, nums, target):
        return {"format_reward": 1.0, "answer_reward": 1.0, "reward": 1.0}
    else:
        return {"format_reward": 1.0, "answer_reward": 0.0, "reward": 0.0}
