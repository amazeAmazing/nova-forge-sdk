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
"""Unit tests for amzn_nova_forge.core.data_mixing_config (TypedDict version)."""

import unittest
from typing import get_type_hints
from unittest.mock import patch

from amzn_nova_forge.core.data_mixing_config import DataMixingConfig
from amzn_nova_forge.util.data_mixing import DataMixing


class TestDataMixingConfigTypedDict(unittest.TestCase):
    """Tests that DataMixingConfig is importable and behaves as a TypedDict."""

    def test_importable(self):
        """DataMixingConfig is importable from core.data_mixing_config."""
        self.assertIsNotNone(DataMixingConfig)

    def test_plain_dict_satisfies_type(self):
        """A plain dict with correct keys satisfies DataMixingConfig at runtime."""
        config: DataMixingConfig = {
            "customer_data_percent": 50.0,
            "nova_code_percent": 50.0,
            "nova_general_percent": 25.0,
            "nova_math_percent": 25.0,
        }
        self.assertEqual(config["customer_data_percent"], 50.0)
        self.assertEqual(config["nova_code_percent"], 50.0)

    def test_subset_of_keys_works(self):
        """TypedDict with total=False allows partial construction."""
        config: DataMixingConfig = {"customer_data_percent": 100.0}
        self.assertEqual(config["customer_data_percent"], 100.0)

    def test_all_declared_keys_constructible(self):
        """All declared keys can be set in a dict literal."""
        config: DataMixingConfig = {
            "customer_data_percent": 50.0,
            "nova_code_percent": 30.0,
            "nova_general_percent": 40.0,
            "nova_math_percent": 20.0,
            "nova_image_percent": 10.0,
        }
        self.assertEqual(config["nova_image_percent"], 10.0)

    def test_type_hints_available(self):
        """get_type_hints exposes the declared fields."""
        hints = get_type_hints(DataMixingConfig)
        self.assertIn("customer_data_percent", hints)
        self.assertIn("nova_code_percent", hints)
        self.assertIn("nova_general_percent", hints)
        self.assertIn("nova_math_percent", hints)
        self.assertIn("nova_image_percent", hints)


class TestDataMixingConfigIntegration(unittest.TestCase):
    """Integration tests: DataMixingConfig dicts flow through DataMixing.set_config()."""

    def _make_dm(self, default_fields=None):
        """Helper to create a DataMixing with default_nova_fields set."""
        dm = DataMixing()
        if default_fields is not None:
            dm._default_nova_fields = default_fields
        else:
            dm._default_nova_fields = {
                "nova_code_percent",
                "nova_general_percent",
                "nova_math_percent",
                "customer_data_percent",
            }
        return dm

    def test_valid_config_flows_through_set_config(self):
        """A valid DataMixingConfig dict is accepted by set_config."""
        dm = self._make_dm()
        config: DataMixingConfig = {
            "customer_data_percent": 50.0,
            "nova_code_percent": 50.0,
            "nova_general_percent": 25.0,
            "nova_math_percent": 25.0,
        }
        dm.set_config(config, normalize=False)
        self.assertEqual(dm.config["customer_data_percent"], 50.0)
        self.assertEqual(dm.config["nova_code_percent"], 50.0)

    def test_range_violation_raises_valueerror(self):
        """A field outside 0-100 raises ValueError with the field name."""
        dm = self._make_dm()
        config = {"customer_data_percent": 150.0, "nova_code_percent": 100.0}
        with self.assertRaises(ValueError) as ctx:
            dm.set_config(config, normalize=False)
        self.assertIn("customer_data_percent", str(ctx.exception))

    def test_negative_range_raises_valueerror(self):
        """A negative field raises ValueError with the field name."""
        dm = self._make_dm()
        config = {"nova_code_percent": -5.0}
        with self.assertRaises(ValueError) as ctx:
            dm.set_config(config, normalize=False)
        self.assertIn("nova_code_percent", str(ctx.exception))

    def test_sum_violation_raises_valueerror(self):
        """Nova fields not summing to 100 raises ValueError."""
        dm = self._make_dm()
        config = {
            "customer_data_percent": 50.0,
            "nova_code_percent": 20.0,
            "nova_general_percent": 20.0,
            "nova_math_percent": 20.0,
        }
        with self.assertRaises(ValueError) as ctx:
            dm.set_config(config, normalize=False)
        msg = str(ctx.exception)
        self.assertIn("sum", msg.lower())

    def test_customer_100_with_nova_nonzero_raises_valueerror(self):
        """customer=100 with nova fields > 0 raises ValueError."""
        dm = self._make_dm()
        config = {
            "customer_data_percent": 100.0,
            "nova_code_percent": 50.0,
            "nova_general_percent": 50.0,
        }
        with self.assertRaises(ValueError) as ctx:
            dm.set_config(config, normalize=False)
        msg = str(ctx.exception)
        self.assertIn("0", msg)

    def test_all_zero_nova_with_customer_below_100_raises_valueerror(self):
        """All nova fields = 0 with customer < 100 raises ValueError."""
        dm = self._make_dm()
        config = {
            "customer_data_percent": 50.0,
            "nova_code_percent": 0.0,
            "nova_general_percent": 0.0,
            "nova_math_percent": 0.0,
        }
        with self.assertRaises(ValueError) as ctx:
            dm.set_config(config, normalize=False)
        msg = str(ctx.exception)
        self.assertIn("100", msg)

    def test_extra_fields_accepted(self):
        """Unknown nova_*_percent fields are accepted."""
        dm = self._make_dm(
            default_fields={
                "nova_code_percent",
                "nova_general_percent",
                "nova_math_percent",
                "nova_video_percent",
                "customer_data_percent",
            }
        )
        config = {
            "customer_data_percent": 50.0,
            "nova_code_percent": 50.0,
            "nova_general_percent": 25.0,
            "nova_math_percent": 25.0,
            "nova_video_percent": 0.0,
        }
        dm.set_config(config, normalize=False)
        self.assertEqual(dm.config["nova_video_percent"], 0.0)

    def test_extra_fields_round_trip(self):
        """Extra nova fields survive the set_config round-trip."""
        dm = self._make_dm(
            default_fields={
                "nova_code_percent",
                "nova_general_percent",
                "nova_math_percent",
                "nova_video_percent",
                "customer_data_percent",
            }
        )
        config = {
            "customer_data_percent": 50.0,
            "nova_code_percent": 50.0,
            "nova_general_percent": 25.0,
            "nova_math_percent": 25.0,
            "nova_video_percent": 0.0,
        }
        dm.set_config(config, normalize=False)
        self.assertIn("nova_video_percent", dm.config)
        self.assertEqual(dm.config["nova_video_percent"], 0.0)

    def test_dataset_catalog_stripped_with_warning(self):
        """dataset_catalog key is stripped and a warning is logged."""
        dm = self._make_dm()
        config = {
            "nova_code_percent": 100.0,
            "customer_data_percent": 50.0,
            "dataset_catalog": "some_catalog",
        }
        with patch("amzn_nova_forge.util.data_mixing.logger") as mock_logger:
            dm.set_config(config, normalize=False)
            mock_logger.warning.assert_called_once()
        self.assertNotIn("dataset_catalog", dm.config)

    def test_none_values_filtered_out(self):
        """None values are filtered out of the stored config."""
        dm = self._make_dm()
        config = {
            "nova_code_percent": 100.0,
            "nova_general_percent": None,
            "customer_data_percent": 50.0,
        }
        dm.set_config(config, normalize=False)
        self.assertNotIn("nova_general_percent", dm.config)
        self.assertEqual(dm.config["nova_code_percent"], 100.0)

    def test_multiple_errors_surfaced(self):
        """Multiple validation errors appear in a single ValueError message."""
        dm = self._make_dm()
        config = {
            "customer_data_percent": 150.0,
            "nova_code_percent": -10.0,
        }
        with self.assertRaises(ValueError) as ctx:
            dm.set_config(config, normalize=False)
        msg = str(ctx.exception)
        self.assertIn("customer_data_percent", msg)
        self.assertIn("nova_code_percent", msg)

    def test_string_value_raises_valueerror(self):
        """String value for a percent field raises ValueError with type info."""
        dm = self._make_dm()
        config = {"customer_data_percent": "999"}
        with self.assertRaises(ValueError) as ctx:
            dm.set_config(config, normalize=False)
        msg = str(ctx.exception)
        self.assertIn("customer_data_percent", msg)
        self.assertIn("must be a number", msg)
        self.assertIn("str", msg)

    def test_float_tolerance(self):
        """Sum within +/-0.01 of 100 should pass."""
        dm = self._make_dm()
        config = {
            "customer_data_percent": 50.0,
            "nova_code_percent": 33.33,
            "nova_general_percent": 33.33,
            "nova_math_percent": 33.34,
        }
        # Should not raise
        dm.set_config(config, normalize=False)

    def test_customer_100_no_nova_valid(self):
        """customer_data_percent=100 with no nova fields is valid."""
        dm = self._make_dm()
        config = {"customer_data_percent": 100.0}
        dm.set_config(config, normalize=False)

    def test_integer_inputs_accepted(self):
        """Integer values are accepted for percent fields."""
        dm = self._make_dm()
        config = {
            "customer_data_percent": 50,
            "nova_code_percent": 50,
            "nova_general_percent": 25,
            "nova_math_percent": 25,
        }
        dm.set_config(config, normalize=False)

    def test_customer_zero_with_nova_sum_100(self):
        """customer_data_percent=0 with nova summing to 100 is valid."""
        dm = self._make_dm()
        config = {
            "customer_data_percent": 0.0,
            "nova_code_percent": 50.0,
            "nova_general_percent": 25.0,
            "nova_math_percent": 25.0,
        }
        dm.set_config(config, normalize=False)


if __name__ == "__main__":
    unittest.main()
