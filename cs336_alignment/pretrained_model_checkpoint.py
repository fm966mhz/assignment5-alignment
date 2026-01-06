"""Pretrained model checkpoint."""

import os
import pickle
import shutil

from transformers import AutoModelForCausalLM, AutoTokenizer


class PretrainedModelCheckpointManager:
    """Manager for pretrained model checkpoints."""

    def __init__(
        self,
        output_dir: str,
        output_name_prefix: str,
        max_num_checkpoints: int,
    ):
        """Initializes the pretrained model checkpoint manager."""
        self.output_dir = output_dir
        self.output_name_prefix = output_name_prefix
        self.max_num_checkpoints = max_num_checkpoints
        self.metadata_file_path = os.path.join(
            self.output_dir, f"{self.output_name_prefix}_metadata.pkl"
        )
        if os.path.exists(self.metadata_file_path):
            with open(self.metadata_file_path, "rb") as f:
                self.metadata = pickle.load(f)
            assert len(self.metadata["step"]) == len(
                self.metadata["checkpoint_dir_names"]
            ), "The number of steps and checkpoint files must match."
        else:
            self.metadata = {
                "step": [],
                "checkpoint_dir_names": [],
            }
            os.makedirs(self.output_dir, exist_ok=True)
            with open(self.metadata_file_path, "wb") as f:
                pickle.dump(self.metadata, f)
        self._trim_checkpoints()

    def _trim_checkpoints(self):
        """Trims the checkpoints to the maximum number of checkpoints."""
        while len(self.metadata["step"]) > self.max_num_checkpoints:
            _ = self.metadata["step"].pop(0)
            checkpoint_dir_name_to_remove = self.metadata["checkpoint_dir_names"].pop(0)
            shutil.rmtree(os.path.join(self.output_dir, checkpoint_dir_name_to_remove))
        with open(self.metadata_file_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def save_checkpoint(
        self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, step: int
    ):
        """Saves the pretrained model checkpoint."""
        checkpoint_name = f"{self.output_name_prefix}_step_{step}"
        model.save_pretrained(os.path.join(self.output_dir, checkpoint_name))
        tokenizer.save_pretrained(os.path.join(self.output_dir, checkpoint_name))
        self.metadata["step"].append(step)
        self.metadata["checkpoint_dir_names"].append(checkpoint_name)
        with open(self.metadata_file_path, "wb") as f:
            pickle.dump(self.metadata, f)
        self._trim_checkpoints()
        return checkpoint_name
