"""Eval utilities for RLVR."""

import dataclasses

from typing import Callable

import vllm

from absl import logging


@dataclasses.dataclass
class EvalResultRow:
    """A single row of evaluation results."""

    input_prompt: str | None
    generated_output: str
    rewards: dict[str, float]
    ground_truth_answer: str | None = None


@dataclasses.dataclass
class EvalResult:
    """Evaluation result containing multiple rows."""

    rows: list[EvalResultRow]
    score: float


def evaluate_on_gsm8k(
    vllm_model: vllm.LLM,
    reward_fn: Callable[[str, str, bool], dict[str, float]],
    model_inputs: list[str],
    ground_truth_answers: list[str],
    eval_sampling_params: vllm.SamplingParams,
    fast_eval: bool = True,
) -> EvalResult:
    """Evaluates the model on GSM8K dataset using the provided reward function.

    Args:
        vllm_model (vllm.LLM): The vLLM model to evaluate.
        reward_fn (Callable[[str, str], dict[str, float]]): The reward function that
            takes model output and ground truth answer and returns reward dict.
        model_inputs (list[str]): The list of input prompts for the model.
        ground_truth_answers (list[str]): The list of ground truth answers corresponding
            to the model inputs.
        eval_sampling_params (vllm.SamplingParams): Sampling parameters for evaluation.
        fast_eval (bool): Whether to use fast evaluation mode.

    Returns:
        EvalResult: the evaluation results containing rows and accuracy.
    """
    inference_outputs = vllm_model.generate(model_inputs, eval_sampling_params)
    eval_rows = []
    correct_count = 0
    for inference_output, ground_truth_answer in zip(
        inference_outputs, ground_truth_answers
    ):
        generated_text = inference_output.outputs[0].text
        try:
            reward = reward_fn(generated_text, ground_truth_answer, fast_eval)
        except Exception as e:
            logging.warn(
                f"Error computing reward for output: {generated_text}. "
                f"Ground truth answer: {ground_truth_answer}. Error: {e}. "
                f"Set reward to empty dict."
            )
            reward = {}
        eval_rows.append(
            EvalResultRow(
                input_prompt=inference_output.prompt,
                generated_output=generated_text,
                rewards=reward,
                ground_truth_answer=ground_truth_answer,
            )
        )
        if reward.get("answer_reward", 0.0) == 1.0:
            correct_count += 1
    return EvalResult(
        rows=eval_rows,
        score=correct_count / len(eval_rows) if eval_rows else 0.0,
    )


def evaluate_on_countdown(
    vllm_model: vllm.LLM,
    reward_fn: Callable[[str, list[int], int], dict[str, float]],
    model_inputs: list[str],
    nums_list: list[list[int]],
    target_list: list[int],
    eval_sampling_params: vllm.SamplingParams,
):
    """Evaluates the model on Countdown dataset using the provided reward function.

    Args:
        vllm_model (vllm.LLM): The vLLM model to evaluate.
        reward_fn (Callable[[str, list[int], int], dict[str, float]]): The reward
            function that takes model output, numbers list, and target and returns
            reward dict.
        model_inputs (list[str]): The list of input prompts for the model.
        nums_list (list[list[int]]): The list of numbers lists for each input.
        target_list (list[int]): The list of target answers corresponding to the inputs.
        eval_sampling_params (vllm.SamplingParams): Sampling parameters for evaluation.

    Returns:
        EvalResult: the evaluation results containing rows and accuracy.
    """
    inference_outputs = vllm_model.generate(model_inputs, eval_sampling_params)
    eval_rows = []
    correct_count = 0
    for inference_output, nums, target in zip(
        inference_outputs, nums_list, target_list
    ):
        generated_text = inference_output.outputs[0].text
        try:
            reward = reward_fn(generated_text, nums, target)
        except Exception as e:
            logging.warn(
                f"Error computing reward for output: {generated_text}. "
                f"Nums: {nums}, Target: {target}. Error: {e}. Set reward to empty dict."
            )
            reward = {}
        eval_rows.append(
            EvalResultRow(
                input_prompt=inference_output.prompt,
                generated_output=generated_text,
                rewards=reward,
            )
        )
        if reward.get("answer_reward", 0.0) == 1.0:
            correct_count += 1
    return EvalResult(
        rows=eval_rows,
        score=correct_count / len(eval_rows) if eval_rows else 0.0,
    )
