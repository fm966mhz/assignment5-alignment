"""A demo of using Hugging Face datasets."""

import datasets
import torch


def demo_countdown():
    """Runs a simple countdown dataset demo."""
    ds_builder = datasets.load_dataset_builder("Jiayi-Pan/Countdown-Tasks-3to4")
    print(f"Countdonw dataset description: {ds_builder.info.description}")
    print(f"Countdown dataset features: {ds_builder.info.features}")
    print(f"Countdown dataset splits: {ds_builder.info.splits}")
    print(
        f"Countdown dataset splits api: "
        f"{datasets.get_dataset_split_names('Jiayi-Pan/Countdown-Tasks-3to4')}"
    )

    dataset = datasets.load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    print(
        f"Countdown dataset length: {dataset.num_rows}"  # pylint: disable=line-too-long # pyright: ignore[reportAttributeAccessIssue]
    )
    print(f"First examples in countdown dataset: {dataset[:10]}")
    print(f"First examples in PyTorch format: {dataset.with_format('torch')[:10]}")


def demo_gsm8k():
    """Runs a simple GSM8K dataset demo."""
    ds = datasets.load_dataset("openai/gsm8k", "main", split="train")
    print(f"GSM8K dataset description: {ds.info.description}")
    dataloader = torch.utils.data.DataLoader(
        ds.with_format("torch"),  # pyright: ignore[reportArgumentType]
        batch_size=4,  # , shuffle=True
    )
    for i, batch in enumerate(dataloader):
        print(f"GSM8K batch {i}: {batch}")
        if i >= 2:
            break


def demo_tulu3():
    """Runs a simple Tulu-3 dataset demo."""
    ds = datasets.load_dataset(
        "allenai/tulu-3-sft-personas-math-filtered", split="train"
    )
    print(f"Tulu-3 dataset description: {ds.info.description}")
    dataloader = torch.utils.data.DataLoader(
        ds.with_format("torch"),  # pyright: ignore[reportArgumentType]
        batch_size=4,  # , shuffle=True
    )
    for i, batch in enumerate(dataloader):
        print(f"Tulu-3 batch {i}: {batch}")
        if i >= 2:
            break


def main():
    """Runs a simple Hugging Face datasets demo."""
    # demo_countdown()
    # demo_gsm8k()
    demo_tulu3()


if __name__ == "__main__":
    main()
