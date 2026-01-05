"""Unit tests for grpo_utils.py."""

import torch
import vllm

from cs336_alignment import grpo_utils


def test_get_log_probs_from_vllm_responses():
    """Tests getting log probabilities from vLLM responses."""
    completion_output = vllm.CompletionOutput(
        index=0,
        text="Some text",
        token_ids=[1, 2, 3],
        logprobs=[
            {1: vllm.sequence.Logprob(logprob=0.1, rank=1)},
            {2: vllm.sequence.Logprob(logprob=0.2, rank=2)},
            {3: vllm.sequence.Logprob(logprob=0.3, rank=3)},
        ],
        cumulative_logprob=0.6,
    )
    log_probs = grpo_utils._get_log_probs_from_vllm_responses(
        completion_output=completion_output
    )
    torch.testing.assert_close(log_probs, torch.tensor([0.1, 0.2, 0.3]))
