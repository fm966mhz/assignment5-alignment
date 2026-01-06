"""Utilities for vLLM."""

import os

import transformers
import vllm

from unittest.mock import patch


def init_vllm(
    model_id_or_path: str,
    device: str,
    seed: int,
    gpu_memory_utilization: float,
) -> vllm.LLM:
    """Initializes a vLLM model.

    Args:
        model_id_or_path: The ID or path of the model to initialize.
        device: The device to use for the model.
        seed: The seed to use for the model.
        gpu_memory_utilization: The GPU memory utilization to use for the model.
    """
    vllm.model_executor.set_random_seed(seed)
    # This is very much a hack, and only seems to work for vLLM 0.7.2.
    # This requires PyTorch version 2.5.1, which doesn't work with my Laptop GPU 5080.
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return vllm.LLM(
            model=model_id_or_path,
            device=device,
            dtype="bfloat16",
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(
    policy: transformers.PreTrainedModel,
    vllm_instance: vllm.LLM,
) -> None:
    """Loads a policy into a vLLM instance.

    Args:
        policy: The policy to load.
        vllm_instance: The vLLM instance to load the policy into.
    """
    state_dict = policy.state_dict()
    # Remove the _orig_mod. prefix from the state dict keys, which is added by torch.compile.
    state_dict = {
        name.replace("_orig_mod.", ""): value for name, value in state_dict.items()
    }
    llm_model = vllm_instance.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())
