"""vLLM Inference Demo."""

from vllm import LLM, SamplingParams


def main():
    """Runs a simple vLLM inference demo."""
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "中国的首都是",
    ]

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["\n"],
    )

    llm = LLM(model="Qwen/Qwen2.5-Math-1.5B")

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt}\nGenerated Text: {generated_text}\n")


if __name__ == "__main__":
    main()
