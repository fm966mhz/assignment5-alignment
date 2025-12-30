"""Script for evaluating model on various tasks."""

import pickle

import datasets
import torch
import tqdm
import vllm

from absl import app
from absl import flags
from absl import logging

from cs336_alignment import custom_grader
from cs336_alignment import data_utils
from cs336_alignment import eval_utils


_model_path = flags.DEFINE_string(
    "model_path", "", "Path to the vLLM model to evaluate."
)
_run_gsm8k_evaluation = flags.DEFINE_bool(
    "run_gsm8k_evaluation", True, "Whether to run GSM8K evaluation."
)
_run_countdown_evaluation = flags.DEFINE_bool(
    "run_countdown_evaluation", True, "Whether to run Countdown evaluation."
)
_system_prompt_path = flags.DEFINE_string(
    "system_prompt_path", None, "Path to system prompt file."
)
_output_path_prefix = flags.DEFINE_string(
    "output_path_prefix", None, "Prefix for output path to write evaluation results."
)
_fast_eval = flags.DEFINE_bool(
    "fast_eval", False, "Whether to use fast evaluation mode."
)
_max_eval_samples = flags.DEFINE_integer(
    "max_eval_samples",
    -1,
    "Maximum number of evaluation samples to use from each dataset.",
)
_countdown_test_dataset_path = flags.DEFINE_string(
    "countdown_test_dataset_path",
    "",
    "Path to the Countdown test dataset.",
)


def _load_system_prompt(
    system_prompt_path: str | None,
) -> str:
    """Loads system prompt from file if provided.

    Args:
        system_prompt_path: Path to system prompt file.

    Returns:
        The system prompt string.
    """
    if system_prompt_path is None:
        return ""
    with open(system_prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def run_gsm8k_evaluation(
    vllm_model: vllm.LLM,
    sampling_params: vllm.SamplingParams,
    system_prompt: str,
    output_path_prefix: str,
    fast_eval: bool,
    max_eval_samples: int = -1,
) -> float:
    """Runs GSM8K evaluation.

    Args:
        vllm_model: The vLLM model to evaluate.
        sampling_params: Sampling parameters for evaluation.
        system_prompt: The system prompt string.
        output_path_prefix: Prefix for output path to write evaluation results.
        fast_eval: Whether to use fast evaluation mode.
        eval_batch_size: Batch size to use during evaluation.
        max_eval_samples: Maximum number of evaluation samples to use.

    Returns:
        The score on the GSM8K evaluation.
    """
    gsm8k_ds = datasets.load_dataset("openai/gsm8k", "main", split="test")
    if max_eval_samples > 0:
        gsm8k_ds = gsm8k_ds.select(range(max_eval_samples))
    logging.info(f"Run on {len(gsm8k_ds)} GSM8K samples.")
    eval_result = eval_utils.evaluate_on_gsm8k(
        vllm_model=vllm_model,
        reward_fn=custom_grader.gsm8k_reward_fn,
        model_inputs=data_utils.generate_gsmk8k_prompt_from_question_list(
            prompt_template=system_prompt,
            questions=gsm8k_ds["question"],
        ),
        ground_truth_answers=gsm8k_ds["answer"],
        eval_sampling_params=sampling_params,
        fast_eval=fast_eval,
    )
    pickle_output_path = f"{output_path_prefix}_gsm8k_eval_results.pkl"
    with open(pickle_output_path, "wb") as f:
        pickle.dump(eval_result, f)
    return eval_result.score


def run_countdown_evaluation(
    vllm_model: vllm.LLM,
    sampling_params: vllm.SamplingParams,
    system_prompt: str,
    test_ds_path: str,
    output_path_prefix: str,
    max_eval_samples: int = -1,
) -> float:
    """Runs Countdown evaluation.

    Args:
        vllm_model: The vLLM model to evaluate.
        sampling_params: Sampling parameters for evaluation.
        system_prompt: The system prompt string.
        test_ds_path: Path to the Countdown test dataset.
        output_path_prefix: Prefix for output path to write evaluation results.
        eval_batch_size: Batch size to use during evaluation.
        max_eval_samples: Maximum number of evaluation samples to use.

    Returns:
        The score on the Countdown evaluation.
    """
    countdown_ds = datasets.Dataset.load_from_disk(test_ds_path)
    if max_eval_samples > 0:
        countdown_ds = countdown_ds.select(range(max_eval_samples))
    logging.info(f"Run on {len(countdown_ds)} Countdown samples.")
    eval_result = eval_utils.evaluate_on_countdown(
        vllm_model=vllm_model,
        reward_fn=custom_grader.countdown_reward_fn,
        model_inputs=data_utils.generate_countdown_prompt_from_nums_target_lists(
            prompt_template=system_prompt,
            nums_list=countdown_ds["nums"],
            target_list=countdown_ds["target"],
        ),
        nums_list=countdown_ds["nums"],
        target_list=countdown_ds["target"],
        eval_sampling_params=sampling_params,
    )
    pickle_output_path = f"{output_path_prefix}_countdown_eval_results.pkl"
    with open(pickle_output_path, "wb") as f:
        pickle.dump(eval_result, f)
    return eval_result.score


def main(argv):
    """Runs evaluations based on command-line flags."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    system_prompt = _load_system_prompt(_system_prompt_path.value)
    vllm_model = vllm.LLM(model=_model_path.value)
    sampling_params = vllm.SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    if _run_gsm8k_evaluation.value:
        gsm8k_accuracy = run_gsm8k_evaluation(
            vllm_model=vllm_model,
            sampling_params=sampling_params,
            system_prompt=system_prompt,
            output_path_prefix=_output_path_prefix.value,  # pyright: ignore[reportArgumentType]
            fast_eval=_fast_eval.value,
            max_eval_samples=_max_eval_samples.value,  # pyright: ignore[reportArgumentType]
        )
        logging.info(f"GSM8K Evaluation Accuracy: {gsm8k_accuracy:.4f}")
    if _run_countdown_evaluation.value:
        countdown_accuracy = run_countdown_evaluation(
            vllm_model=vllm_model,
            sampling_params=sampling_params,
            system_prompt=system_prompt,
            test_ds_path=_countdown_test_dataset_path.value,
            output_path_prefix=_output_path_prefix.value,  # pyright: ignore[reportArgumentType]
            max_eval_samples=_max_eval_samples.value,  # pyright: ignore[reportArgumentType]
        )
        logging.info(f"Countdown Evaluation Accuracy: {countdown_accuracy:.4f}")


if __name__ == "__main__":
    app.run(main)
