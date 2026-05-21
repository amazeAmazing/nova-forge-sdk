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
"""Typed configuration for data mixing percentages."""

from typing_extensions import TypedDict


class DataMixingConfig(TypedDict, total=False):
    """Typed configuration for data mixing between customer and Nova curated data.

    All percent fields must be 0-100. When customer_data_percent < 100, the
    nova_*_percent fields must sum to 100. Unrecognized nova_*_percent fields
    are accepted since TypedDict does not reject unknown keys at runtime.

    Can be passed directly to ``DataMixing.set_config()`` — plain dicts
    satisfy this annotation without construction.

    Example::

        config: DataMixingConfig = {
            "customer_data_percent": 50.0,
            "nova_code_percent": 30.0,
            "nova_general_percent": 70.0,
        }
    """

    customer_data_percent: float
    nova_code_percent: float
    nova_general_percent: float
    nova_math_percent: float
    nova_image_percent: float
