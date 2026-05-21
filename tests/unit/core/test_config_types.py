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
"""Unit tests for ConfigParameter and RecipeConfig."""

import unittest

from amzn_nova_forge.core.enums import Model, Platform, TrainingMethod
from amzn_nova_forge.core.types import ConfigParameter, RecipeConfig


class TestConfigParameter(unittest.TestCase):
    def test_frozen(self):
        param = ConfigParameter(name="lr", type="float", default=5e-6)
        with self.assertRaises(AttributeError):
            param.name = "other"

    def test_defaults(self):
        param = ConfigParameter(name="lr", type="float", default=5e-6)
        self.assertIsNone(param.description)
        self.assertIsNone(param.min)
        self.assertIsNone(param.max)
        self.assertIsNone(param.enum)
        self.assertFalse(param.required)

    def test_with_constraints(self):
        param = ConfigParameter(
            name="lr",
            type="float",
            default=5e-6,
            description="Learning rate",
            min=1e-6,
            max=1e-4,
        )
        self.assertEqual(param.min, 1e-6)
        self.assertEqual(param.max, 1e-4)
        self.assertEqual(param.description, "Learning rate")

    def test_with_enum(self):
        param = ConfigParameter(
            name="reasoning_effort",
            type="string",
            default="medium",
            enum=("low", "medium", "high"),
        )
        self.assertEqual(param.enum, ("low", "medium", "high"))

    def test_equality(self):
        a = ConfigParameter(name="lr", type="float", default=5e-6)
        b = ConfigParameter(name="lr", type="float", default=5e-6)
        self.assertEqual(a, b)

    def test_hashable(self):
        param = ConfigParameter(name="lr", type="float", default=5e-6)
        self.assertIsInstance(hash(param), int)


class TestRecipeConfig(unittest.TestCase):
    def setUp(self):
        self.params = (
            ConfigParameter(name="lr", type="float", default=5e-6, min=1e-6, max=1e-4),
            ConfigParameter(name="max_epochs", type="integer", default=2, min=1, max=5),
            ConfigParameter(
                name="reasoning_effort",
                type="string",
                default="medium",
                enum=("low", "medium", "high"),
            ),
        )
        self.config = RecipeConfig(
            model=Model.NOVA_LITE,
            method=TrainingMethod.SFT_LORA,
            platform=Platform.SMTJ,
            parameters=self.params,
        )

    def test_frozen(self):
        with self.assertRaises(AttributeError):
            self.config.model = Model.NOVA_PRO

    def test_to_dict(self):
        result = self.config.to_dict()
        self.assertEqual(result, {"lr": 5e-6, "max_epochs": 2, "reasoning_effort": "medium"})

    def test_repr_contains_model_and_params(self):
        text = repr(self.config)
        self.assertIn("NOVA_LITE", text)
        self.assertIn("SFT_LORA", text)
        self.assertIn("SMTJ", text)
        self.assertIn("lr: float = 5e-06", text)
        self.assertIn("[1e-06..0.0001]", text)
        self.assertIn("max_epochs: integer = 2", text)
        self.assertIn("enum=['low', 'medium', 'high']", text)

    def test_empty_parameters(self):
        config = RecipeConfig(
            model=Model.NOVA_LITE,
            method=TrainingMethod.SFT_LORA,
            platform=Platform.SMTJ,
            parameters=(),
        )
        self.assertEqual(config.to_dict(), {})
        self.assertIn("NOVA_LITE", repr(config))

    def test_equality(self):
        other = RecipeConfig(
            model=Model.NOVA_LITE,
            method=TrainingMethod.SFT_LORA,
            platform=Platform.SMTJ,
            parameters=self.params,
        )
        self.assertEqual(self.config, other)

    def test_hashable(self):
        self.assertIsInstance(hash(self.config), int)
