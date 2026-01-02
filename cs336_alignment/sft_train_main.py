"""Main script for SFT training."""

from typing import Any

import datasets
import torch
import tqdm
import transformers
import wandb
import vllm

from absl import app
from absl import flags
from absl import logging

from cs336_alignment import custom_grader
from cs336_alignment import data_utils
from cs336_alignment import eval_utils
from cs336_alignment import sft_helpers
from cs336_alignment import vllm_utils

_model_id = flags.DEFINE_string(
    "model_id",
    "",
    "The ID of the model to use for training.",
)
_prompt_template_path = flags.DEFINE_string(
    "prompt_template_path",
    "",
    "The path to the prompt template to use for training.",
)
_output_dir = flags.DEFINE_string(
    "output_dir",
    "",
    "The directory to save the output to.",
)
_wandb_entity = flags.DEFINE_string(
    "wandb_entity",
    "",
    "The entity to use for Weights and Biases.",
)
_wandb_project = flags.DEFINE_string(
    "wandb_project",
    "",
    "The project to use for Weights and Biases.",
)
_wandb_run_name = flags.DEFINE_string(
    "wandb_run_name",
    "",
    "The name of the run to use for Weights and Biases.",
)
_learning_rate = flags.DEFINE_float(
    "learning_rate",
    1e-4,
    "The learning rate to use for training.",
)
_num_training_examples = flags.DEFINE_integer(
    "num_training_examples",
    -1,
    "The number of training examples to use for training. If -1, use all training examples.",
)
_num_validation_examples = flags.DEFINE_integer(
    "num_validation_examples",
    -1,
    "The number of validation examples to use for validation. If -1, use all validation examples.",
)
_batch_size = flags.DEFINE_integer(
    "batch_size",
    16,
    "The batch size to use for training.",
)
_num_epochs = flags.DEFINE_integer(
    "num_epochs",
    10,
    "The number of epochs to use for training.",
)
_seed = flags.DEFINE_integer(
    "seed",
    42,
    "The seed to use for training.",
)
_gradient_accumulation_steps = flags.DEFINE_integer(
    "gradient_accumulation_steps",
    1,
    "The number of gradient accumulation steps to use for training.",
)
_validate_every_n_updates = flags.DEFINE_integer(
    "validate_every_n_updates",
    5,
    "The number of updates to use for validation.",
)
_gradient_clip = flags.DEFINE_float(
    "gradient_clip",
    1.0,
    "The gradient clip to use for training.",
)


def _check_at_least_two_gpus() -> None:
    """Checks that there are at least two GPUs available."""
    if torch.cuda.device_count() < 2:
        raise app.UsageError("There must be at least two GPUs available.")


def _get_base_model(
    device: str,
) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizerBase]:
    """Gets the base model."""
    model = transformers.AutoModelForCausalLM.from_pretrained(
        _model_id.value,
        dtype=torch.bfloat16,
    )
    model = model.to(device)  # pyright: ignore[reportArgumentType]
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        _model_id.value,
    )
    return model, tokenizer


def _get_dataloader() -> tuple[torch.utils.data.DataLoader, torch.utils.data.Dataset]:
    """Gets the training and validation datasets.

    Returns:
        A tuple of the training data loader and validation datasets.
    """
    gsm8k_ds = datasets.load_dataset(
        "openai/gsm8k",
        "main",
        split="train",
    )
    split_dataset_dict = gsm8k_ds.train_test_split(test_size=0.15, shuffle=False)
    train_ds = split_dataset_dict["train"]
    if _num_training_examples.value > 0:
        train_ds = train_ds.select(range(_num_training_examples.value))
    train_dataloader = torch.utils.data.DataLoader(
        train_ds.with_format("torch"),  # type: ignore[reportArgumentType]
        batch_size=_batch_size.value,
        shuffle=True,
    )
    validation_ds = split_dataset_dict["test"]
    if _num_validation_examples.value > 0:
        validation_ds = validation_ds.select(range(_num_validation_examples.value))
    return train_dataloader, validation_ds  # type: ignore[reportReturnValueType]


def _load_prompt_template(prompt_template_path: str) -> str:
    """Loads the prompt template from the given path."""
    with open(prompt_template_path, "r") as f:
        return f.read()


def _init_wandb_run(
    wandb_entity: str,
    wandb_project: str,
    wandb_run_name: str,
    config: dict[str, Any],
) -> Any:
    """Initializes the WandB run."""
    output = wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        name=wandb_run_name,
        config=config,
    )
    output.define_metric("train_step")
    output.define_metric("eval_step")
    output.define_metric("train/*", step_metric="train_step")
    output.define_metric("eval/*", step_metric="eval_step")
    return output


def _get_optimizer(
    policy_model: transformers.PreTrainedModel,
    learning_rate: float,
) -> torch.optim.Optimizer:
    """Gets the optimizer for the policy model."""
    return torch.optim.AdamW(
        policy_model.parameters(),
        lr=learning_rate,
    )


def main(argv):
    """Main function for SFT training."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    _check_at_least_two_gpus()

    train_dataloader, validation_ds = _get_dataloader()
    prompt_template = _load_prompt_template(_prompt_template_path.value)

    policy_model, tokenizer = _get_base_model(device="cuda:0")
    optimizer = _get_optimizer(
        policy_model=policy_model,
        learning_rate=_learning_rate.value,
    )
    vllm_model = vllm_utils.init_vllm(
        model_id=_model_id.value,
        device="cuda:1",
        seed=_seed.value,
        gpu_memory_utilization=0.85,
    )
    eval_sampling_params = vllm.SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    wandb_run = _init_wandb_run(
        wandb_entity=_wandb_entity.value,
        wandb_project=_wandb_project.value,
        wandb_run_name=_wandb_run_name.value,
        config={
            "model_id": _model_id.value,
            "learning_rate": _learning_rate.value,
            "num_training_examples": _num_training_examples.value,
            "batch_size": _batch_size.value,
            "num_epochs": _num_epochs.value,
            "seed": _seed.value,
            "gradient_accumulation_steps": _gradient_accumulation_steps.value,
            "gradient_clip": _gradient_clip.value,
            "prompt_template": prompt_template,
        },
    )

    # Init evaluation results
    vllm_utils.load_policy_into_vllm_instance(
        policy=policy_model,
        vllm_instance=vllm_model,
    )
    eval_result = eval_utils.evaluate_on_gsm8k(
        vllm_model=vllm_model,
        reward_fn=custom_grader.gsm8k_reward_fn,
        model_inputs=data_utils.generate_gsm8k_prompt_from_question_list(
            prompt_template=prompt_template,
            questions=validation_ds["question"],
        ),
        ground_truth_answers=validation_ds["answer"],
        eval_sampling_params=eval_sampling_params,
    )
    wandb_run.log(
        {
            "eval_step": 0,
            "eval/gsm8k_score": eval_result.score,
        }
    )

    num_microbatches_used = 0
    for epoch in range(_num_epochs.value):
        logging.info(f"Starting epoch {epoch}...")
        for batch in tqdm.tqdm(train_dataloader, desc="Training"):
            input_questions = batch["question"]
            gt_answers = data_utils.format_gsm8k_answer(batch["answer"])
            input_prompts = data_utils.generate_gsm8k_prompt_from_question_list(
                prompt_template=prompt_template,
                questions=input_questions,
            )
            tokenized_input_dict = sft_helpers.tokenize_prompt_and_output(
                prompt_strs=input_prompts,
                output_strs=gt_answers,
                tokenizer=tokenizer,
            )
            tokenized_input_dict = {
                key: value.pin_memory().to(device="cuda:0", non_blocking=True)
                for key, value in tokenized_input_dict.items()
            }
            policy_log_probs_and_entropy = sft_helpers.get_response_log_probs(
                model=policy_model,
                input_ids=tokenized_input_dict["input_ids"],
                labels=tokenized_input_dict["labels"],
                return_token_entropy=True,
            )
            response_mask = tokenized_input_dict["response_mask"]
            loss, _ = sft_helpers.sft_microbatch_train_step(
                policy_log_probs=policy_log_probs_and_entropy["log_probs"],
                response_mask=response_mask,
                gradient_accumulation_steps=_gradient_accumulation_steps.value,
            )
            average_token_entropy = sft_helpers.masked_normalize(
                tensor=policy_log_probs_and_entropy["token_entropy"],
                mask=response_mask,
                normalize_constant=response_mask.sum().item(),
                dim=-1,
            )
            # Log training metrics and data.
            if num_microbatches_used % _gradient_accumulation_steps.value == 0:
                wandb_run.log(
                    {
                        "train_step": num_microbatches_used
                        // _gradient_accumulation_steps.value,
                        "train/loss": loss.detach().item(),
                        "train/average_token_entropy": average_token_entropy.mean()
                        .detach()
                        .item(),
                        "train/prompt_gt_sample": wandb.Table(
                            columns=["input_prompt", "ground_truth_answer"],
                            data=[
                                [input_prompt, gt_answer]
                                for input_prompt, gt_answer in zip(
                                    input_prompts, gt_answers
                                )
                            ],
                        ),
                    }
                )
            # Update parameters.
            num_microbatches_used += 1
            if num_microbatches_used % _gradient_accumulation_steps.value == 0:
                torch.nn.utils.clip_grad_norm_(
                    parameters=policy_model.parameters(),
                    max_norm=_gradient_clip.value,
                )
                optimizer.step()
                optimizer.zero_grad()
            # Run evals.
            if (
                num_microbatches_used
                % (_validate_every_n_updates.value * _gradient_accumulation_steps.value)
                == 0
            ):
                vllm_utils.load_policy_into_vllm_instance(
                    policy=policy_model,
                    vllm_instance=vllm_model,
                )
                eval_result = eval_utils.evaluate_on_gsm8k(
                    vllm_model=vllm_model,
                    reward_fn=custom_grader.gsm8k_reward_fn,
                    model_inputs=data_utils.generate_gsm8k_prompt_from_question_list(
                        prompt_template=prompt_template,
                        questions=validation_ds["question"],
                    ),
                    ground_truth_answers=validation_ds["answer"],
                    eval_sampling_params=eval_sampling_params,
                )
                wandb_run.log(
                    {
                        "eval_step": num_microbatches_used
                        // _gradient_accumulation_steps.value,
                        "eval/gsm8k_score": eval_result.score,
                        "eval/prompt_gt_correct_sample": eval_utils.get_sample_eval_result_table(
                            eval_result=eval_result,
                            max_num_samples=10,
                            correct_samples=True,
                            incorrect_samples=False,
                        ),
                        "eval/prompt_gt_incorrect_sample": eval_utils.get_sample_eval_result_table(
                            eval_result=eval_result,
                            max_num_samples=10,
                            correct_samples=False,
                            incorrect_samples=True,
                        ),
                    }
                )
            del (
                input_questions,
                gt_answers,
                input_prompts,
                tokenized_input_dict,
                policy_log_probs_and_entropy,
                response_mask,
                loss,
                average_token_entropy,
            )

        logging.info(f"Epoch {epoch} completed. Saving policy model and tokenizer...")
        output_dir = f"{_output_dir.value}/policy_model_checkpoint_{epoch}"
        policy_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logging.info(f"Policy model and tokenizer saved successfully to {output_dir}.")


if __name__ == "__main__":
    app.run(main)
