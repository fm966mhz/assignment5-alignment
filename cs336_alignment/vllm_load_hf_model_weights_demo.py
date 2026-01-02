"""vLLM Load HF Model Weights Demo."""

import os

import torch
import transformers
import vllm

from cs336_alignment import vllm_utils


def main():
    """Runs a simple vLLM load HF model weights demo."""
    print(f"{vllm.__version__}")
    policy_model = transformers.AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B",
        dtype=torch.bfloat16,
    ).to("cuda:0")
    vllm_model = vllm_utils.init_vllm(
        model_id="Qwen/Qwen2.5-Math-1.5B",
        device="cuda:1",
        seed=42,
        gpu_memory_utilization=0.85,
    )
    print(f"Loading HF model weights into vLLM model...")
    vllm_utils.load_policy_into_vllm_instance(
        policy=policy_model,
        vllm_instance=vllm_model,
    )
    print(f"vLLM model weights loaded successfully.")
    print(f"Running inference...")
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "中国的首都是",
    ]
    sampling_params = vllm.SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["\n"],
    )
    outputs = vllm_model.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt}\nGenerated Text: {generated_text}\n")

if __name__ == "__main__":
    main()