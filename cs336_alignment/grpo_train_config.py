"""Configuration for GRPO training."""

from dataclasses import dataclass
from typing import Literal, cast

from absl import flags

n_grpo_steps = flags.DEFINE_integer(
    "n_grpo_steps",
    10,
    "The number of GRPO steps to use for training.",
)
_learning_rate = flags.DEFINE_float(
    "learning_rate",
    1e-5,
    "The learning rate to use for training.",
)
_min_learning_rate = flags.DEFINE_float(
    "min_learning_rate",
    1e-6,
    "The minimum learning rate to use for training.",
)
_lr_warmup_grpo_steps = flags.DEFINE_integer(
    "lr_warmup_grpo_steps",
    15,
    "The ratio of learning rate warmup iterations to the total number of iterations.",
)
_lr_cosine_cycle_grpo_steps = flags.DEFINE_integer(
    "lr_cosine_cycle_grpo_steps",
    50,
    "The number of GRPO steps to use for the cosine cycle of the learning rate scheduler.",
)
_advantage_epsilon = flags.DEFINE_float(
    "advantage_epsilon",
    1e-6,
    "The advantage epsilon to use for training.",
)
_rollout_batch_size = flags.DEFINE_integer(
    "rollout_batch_size",
    256,
    "The number of rollouts to sample per GRPO step.",
)
_group_size = flags.DEFINE_integer(
    "group_size",
    8,
    "The number of rollouts to group together for reward computation.",
)
_sampling_temperature = flags.DEFINE_float(
    "sampling_temperature",
    1.0,
    "The temperature to use for sampling.",
)
_top_p = flags.DEFINE_float("top_p", 1.0, "The top-p to use for sampling.")
_sampling_max_tokens = flags.DEFINE_integer(
    "sampling_max_tokens",
    1024,
    "The maximum number of tokens to generate for each rollout.",
)
_sampling_min_tokens = flags.DEFINE_integer(
    "sampling_min_tokens",
    4,
    "The minimum number of tokens to generate for each rollout.",
)
_sampling_stop = flags.DEFINE_list(
    "sampling_stop",
    ["</answer>"],
    "The stop tokens to use for sampling.",
)
_epochs_per_rollout_batch = flags.DEFINE_integer(
    "epochs_per_rollout_batch",
    1,
    "The number of epochs to use for training per rollout batch.",
)
_train_batch_size = flags.DEFINE_integer(
    "train_batch_size",
    256,
    "The effective batch size to use for training.",
)
_policy_model_inference_batch_size = flags.DEFINE_integer(
    "policy_model_inference_batch_size",
    16,
    "The batch size to use for policy model inference. This is used for getting the old policy log "
    "probabilities.",
)
_evaluation_sample_size = flags.DEFINE_integer(
    "evaluation_sample_size",
    -1,
    "The number of samples to use for evaluation. If -1, use all samples.",
)
_gradient_accumulation_steps = flags.DEFINE_integer(
    "gradient_accumulation_steps",
    128,
    "The number of gradient accumulation steps to use for training.",
)
_gpu_memory_utilization = flags.DEFINE_float(
    "gpu_memory_utilization",
    0.85,
    "The GPU memory utilization to use for training.",
)
_loss_type = flags.DEFINE_enum(
    "loss_type",
    "reinforce_with_baseline",
    ["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    "The type of loss to use for training.",
)
_use_length_normalization = flags.DEFINE_boolean(
    "use_length_normalization",
    True,
    "Whether to use length normalization for rewards.",
)
_use_std_normalization = flags.DEFINE_boolean(
    "use_std_normalization",
    True,
    "Whether to use std normalization for rewards.",
)
_adamw_weight_decay = flags.DEFINE_float(
    "adamw_weight_decay",
    0,
    "The weight decay to use for AdamW optimizer.",
)
_adamw_beta_1 = flags.DEFINE_float(
    "adamw_beta_1",
    0.9,
    "The beta 1 to use for AdamW optimizer.",
)
_adamw_beta_2 = flags.DEFINE_float(
    "adamw_beta_2",
    0.95,
    "The beta 2 to use for AdamW optimizer.",
)
_gradient_clip = flags.DEFINE_float(
    "gradient_clip",
    1.0,
    "The gradient clip to use for training.",
)
_cliprange = flags.DEFINE_float(
    "cliprange",
    0.2,
    "The clip range for the ratio.",
)
_early_stop_kl_divergence_threshold = flags.DEFINE_float(
    "early_stop_kl_divergence_threshold",
    float("inf"),
    "The KL divergence threshold to use for early stopping.",
)
_validation_every_n_updates = flags.DEFINE_integer(
    "validation_every_n_updates",
    5,
    "Run validation every n updates/steps to the policy model.",
)
_log_training_metrics_every_n_microbatches = flags.DEFINE_integer(
    "log_training_metrics_every_n_microbatches",
    10,
    "Log training metrics every n microbatches.",
)


@dataclass
class GrpoTrainConfig:  # pylint: disable=too-many-instance-attributes
    """Configuration for GRPO training."""

    n_grpo_steps: int
    learning_rate: float
    min_learning_rate: float
    lr_warmup_grpo_steps: int
    lr_cosine_cycle_grpo_steps: int
    advantage_epsilon: float
    rollout_batch_size: int
    group_size: int
    sampling_temperature: float
    top_p: float
    sampling_max_tokens: int
    sampling_min_tokens: int
    sampling_stop: list[str]
    epochs_per_rollout_batch: int
    train_batch_size: int
    policy_model_inference_batch_size: int
    evaluation_sample_size: int
    gradient_accumulation_steps: int
    gpu_memory_utilization: float
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"]
    use_length_normalization: bool
    use_std_normalization: bool
    adamw_weight_decay: float
    adamw_beta_1: float
    adamw_beta_2: float
    gradient_clip: float
    cliprange: float
    early_stop_kl_divergence_threshold: float
    validation_every_n_updates: int
    log_training_metrics_every_n_microbatches: int
    n_microbatches_per_rollout_batch: int
    n_policy_model_inference_batches_per_rollout_batch: int
    microbatch_size: int


def get_grpo_train_config() -> GrpoTrainConfig:
    """Gets the GRPO training configuration."""
    assert _train_batch_size.value % _gradient_accumulation_steps.value == 0, (
        f"Train batch size {_train_batch_size.value} must be divisible by gradient accumulation "
        f"steps {_gradient_accumulation_steps.value}"
    )
    assert (
        _rollout_batch_size.value % _group_size.value == 0
    ), f"Rollout batch size {_rollout_batch_size.value} must be divisible by group size {_group_size.value}"
    assert _train_batch_size.value >= _group_size.value, (
        f"Train batch size {_train_batch_size.value} must be greater than or equal to group size "
        f"{_group_size.value}"
    )
    micro_batch_size = _train_batch_size.value // _gradient_accumulation_steps.value
    assert _rollout_batch_size.value % micro_batch_size == 0, (
        f"Rollout out batch size {_rollout_batch_size.value} must be divisible by micro batch size "
        f"{micro_batch_size}"
    )
    n_microbatches_per_rollout_batch = _rollout_batch_size.value // micro_batch_size
    assert _rollout_batch_size.value % _policy_model_inference_batch_size.value == 0, (
        f"Rollout batch size {_rollout_batch_size.value} must be divisible by policy model inference "
        f"batch size {_policy_model_inference_batch_size.value}"
    )
    n_policy_model_inference_batches_per_rollout_batch = (
        _rollout_batch_size.value // _policy_model_inference_batch_size.value
    )
    return GrpoTrainConfig(
        n_grpo_steps=n_grpo_steps.value,
        learning_rate=_learning_rate.value,
        min_learning_rate=_min_learning_rate.value,
        lr_warmup_grpo_steps=_lr_warmup_grpo_steps.value,
        lr_cosine_cycle_grpo_steps=_lr_cosine_cycle_grpo_steps.value,
        advantage_epsilon=_advantage_epsilon.value,
        rollout_batch_size=_rollout_batch_size.value,
        group_size=_group_size.value,
        sampling_temperature=_sampling_temperature.value,
        top_p=_top_p.value,
        sampling_max_tokens=_sampling_max_tokens.value,
        sampling_min_tokens=_sampling_min_tokens.value,
        sampling_stop=_sampling_stop.value,
        epochs_per_rollout_batch=_epochs_per_rollout_batch.value,
        train_batch_size=_train_batch_size.value,
        policy_model_inference_batch_size=_policy_model_inference_batch_size.value,
        evaluation_sample_size=_evaluation_sample_size.value,
        gradient_accumulation_steps=_gradient_accumulation_steps.value,
        gpu_memory_utilization=_gpu_memory_utilization.value,
        loss_type=cast(
            Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
            _loss_type.value,
        ),
        use_length_normalization=_use_length_normalization.value,
        use_std_normalization=_use_std_normalization.value,
        adamw_weight_decay=_adamw_weight_decay.value,
        adamw_beta_1=_adamw_beta_1.value,
        adamw_beta_2=_adamw_beta_2.value,
        gradient_clip=_gradient_clip.value,
        cliprange=_cliprange.value,
        early_stop_kl_divergence_threshold=_early_stop_kl_divergence_threshold.value,
        validation_every_n_updates=_validation_every_n_updates.value,
        log_training_metrics_every_n_microbatches=_log_training_metrics_every_n_microbatches.value,
        n_microbatches_per_rollout_batch=n_microbatches_per_rollout_batch,
        n_policy_model_inference_batches_per_rollout_batch=n_policy_model_inference_batches_per_rollout_batch,
        microbatch_size=micro_batch_size,
    )
