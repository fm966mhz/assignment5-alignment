"""Data utility functions for prompt generation and collation."""


def generate_gsm8k_prompt_collate(
    prompt_template: str, rows: list[dict[str, str]]
) -> dict[str, list[str]]:
    """Generates GSM8K prompts by collating provided rows.

    Args:
        prompt_template: The prompt template string with placeholders.
        rows: A list of dictionaries, each containing 'question' and 'answer' keys.

    Returns:
        A dictionary with generated prompts and answers.
    """
    prompts = []
    answers = []
    for row in rows:
        prompt = prompt_template.format(
            question=f"{row["question"]} \n Please give the final answer as a single number."
        )
        prompts.append(prompt)
        answers.append(row["answer"])
    return {
        "prompt": prompts,
        "answer": answers,
    }


def generate_gsm8k_prompt_from_question_list(
    prompt_template: str, questions: list[str]
) -> list[str]:
    """Generates GSM8K prompts from a list of questions.

    Args:
        prompt_template: The prompt template string with placeholders.
        questions: A list of question strings.

    Returns:
        A list of generated prompts.
    """
    prompts = []
    for question in questions:
        prompt = prompt_template.format(
            question=f"{question} \n Please give the final answer as a single number."
        )
        prompts.append(prompt)
    return prompts


def format_gsm8k_answer(gsm8k_answers: list[str]) -> list[str]:
    """Formats GSM8k answers to the one matching `prompts/my_system_prompt.prompt`."""
    output = []
    for gt_answer in gsm8k_answers:
        parts = gt_answer.strip().split("####")
        if len(parts) < 2:
            raise ValueError(f"Invalid GSM8K answer: {gt_answer}")
        answer_part = parts[-1].strip().replace(",", "")
        try:
            answer = float(answer_part)
        except ValueError as e:
            raise ValueError(f"Invalid GSM8K answer: {gt_answer}") from e
        reasoning_part = "####".join(parts[:-1])
        output.append(f"{reasoning_part} </think> <answer> {answer} </answer>")
    return output


def generate_countdown_prompt_collate(
    prompt_template: str, rows: list[dict[str, list[int] | int]]
) -> dict[str, list[str | int | list[int]]]:
    """Generates Countdown prompts by collating provided rows.

    Args:
        prompt_template: The prompt template string with placeholders.
        rows: A list of dictionaries, each containing 'nums' and 'target' keys.

    Returns:
        A list of generated prompts.
    """
    prompts, nums_list, target_list = [], [], []
    for row in rows:
        nums = row["nums"]  # pyright: ignore[reportGeneralTypeIssues]
        target = row["target"]  # pyright: ignore[reportGeneralTypeIssues
        question = (
            f"You're given the following numbers: {nums}. "
            f"Use addition ('+'), subtraction ('-'), multiplication ('*'), division ('/') and "
            f"paraentheses ('()') to reach the "
            f"target number: {target}. You need to use each number exactly once. You can use the "
            f"operations as many times as you want.\nGive your final answer as a "
            f"single arithmetic expression."
        )
        prompt = prompt_template.format(
            question=question,
        )
        prompts.append(prompt)
        nums_list.append(nums)
        target_list.append(target)
    return {
        "prompts": prompts,
        "nums": nums_list,
        "target": target_list,
    }  # pyright: ignore[reportReturnType]


def generate_countdown_prompt_from_nums_target_lists(
    prompt_template: str, nums_list: list[list[int]], target_list: list[int]
) -> list[str]:
    """Generates Countdown prompts from lists of numbers and targets.

    Args:
        prompt_template: The prompt template string with placeholders.
        nums_list: A list of lists of integers representing the numbers.
        target_list: A list of integers representing the target numbers.

    Returns:
        A list of generated prompts.
    """
    prompts = []
    for nums, target in zip(nums_list, target_list):
        question = (
            f"You're given the following numbers: {nums}. "
            f"Use addition ('+'), subtraction ('-'), multiplication ('*'), division ('/') and "
            f"paraentheses ('()') to reach the "
            f"target number: {target}. You need to use each number exactly once. You can use the "
            f"operations as many times as you want.\nGive your final answer as a "
            f"single arithmetic expression."
        )
        prompt = prompt_template.format(
            question=question,
        )
        prompts.append(prompt)
    return prompts
