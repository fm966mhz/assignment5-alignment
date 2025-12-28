"""Randomly split datasets."""

import datasets

from absl import app
from absl import flags
from absl import logging

_output_dir = flags.DEFINE_string(
    "output_dir",
    None,
    "Directory to save the split datasets.",
)


def split_dataset(dataset_name: str, output_dir: str, test_size: float) -> str:
    """Randomly splits the Countdown dataset into train, validation, and test sets."""
    dataset = datasets.load_dataset(dataset_name, split="train")
    split_dataset_dict = dataset.train_test_split(test_size=test_size, seed=42)
    file_name_prefix = f"{output_dir}/{dataset_name.replace('/', '_')}"
    split_dataset_dict["train"].save_to_disk(f"{file_name_prefix}_train")
    split_dataset_dict["test"].save_to_disk(f"{file_name_prefix}_test")
    return file_name_prefix


def main(argv):
    """Main function to split datasets."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    for dataset_name in [
        "Jiayi-Pan/Countdown-Tasks-3to4",
        "allenai/tulu-3-sft-personas-math-filtered",
    ]:
        logging.info(f"Splitting dataset: {dataset_name}")
        output_file_prefix = split_dataset(
            dataset_name, output_dir=_output_dir.value, test_size=0.15
        )
        logging.info(
            f"Finished splitting dataset: {dataset_name} to {output_file_prefix}_train and "
            f"{output_file_prefix}_test"
        )


if __name__ == "__main__":
    app.run(main)
