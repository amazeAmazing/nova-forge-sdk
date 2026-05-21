# Copyright Amazon.com, Inc. or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for amzn_nova_forge.core.training_overrides."""

import unittest
from typing import get_type_hints

from amzn_nova_forge.core.training_overrides import TrainingOverrides


class TestTrainingOverrides(unittest.TestCase):
    def test_plain_dict_compatible(self):
        """A plain dict with known keys is a valid TrainingOverrides at runtime."""
        overrides: TrainingOverrides = {"lr": 0.001, "max_steps": 100}
        assert overrides["lr"] == 0.001
        assert overrides["max_steps"] == 100

    def test_subset_of_keys(self):
        """Only a subset of keys is needed (total=False)."""
        overrides: TrainingOverrides = {"global_batch_size": 8}
        assert overrides["global_batch_size"] == 8

    def test_all_keys_constructible(self):
        """All declared keys can be set."""
        overrides: TrainingOverrides = {
            "lr": 0.001,
            "loraplus_lr_ratio": 4.0,
            "lora_plus_lr_ratio": 4.0,
            "max_steps": 1000,
            "max_epochs": 3,
            "save_steps": 500,
            "warmup_steps": 100,
            "global_batch_size": 16,
            "max_length": 4096,
            "alpha": 32,
            "beta": 0.1,
            "reasoning_enabled": True,
            "reasoning_effort": "medium",
            "val_check_interval": 50,
            "validation_s3_path": "s3://bucket/val",
            "validation_data_s3_path": "s3://bucket/val_data",
            "top_logprobs": 5,
        }
        assert len(overrides) == 17

    def test_unknown_keys_no_runtime_error(self):
        """TypedDict does not reject unknown keys at runtime."""
        overrides: TrainingOverrides = {"lr": 0.001, "some_future_key": "value"}  # type: ignore[typeddict-unknown-key]
        assert "some_future_key" in overrides

    def test_empty_dict_valid(self):
        """Empty dict is valid since total=False."""
        overrides: TrainingOverrides = {}
        assert len(overrides) == 0

    def test_type_hints_available(self):
        """TypedDict exposes its annotations for introspection."""
        hints = get_type_hints(TrainingOverrides)
        assert "lr" in hints
        assert "max_steps" in hints
        assert hints["lr"] is float
        assert hints["max_steps"] is int


if __name__ == "__main__":
    unittest.main()
