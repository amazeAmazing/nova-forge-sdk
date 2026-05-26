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
"""InspectLens evaluation configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from amzn_nova_forge.validation.inspect_lens_validator import validate_inspect_lens_config


@dataclass
class InspectLensConfig:
    """Configuration for an InspectLens job-based evaluation.

    Serialized to ``inspect_config.yaml`` and uploaded to S3 before the
    SageMaker Training Job is submitted.

    Decoding parameters (``temperature``, ``top_p``, ``top_k``, ``max_tokens``,
    ``max_connections``, ``max_retries``, ``timeout``) are passed via the
    ``overrides`` parameter on ``ForgeEvaluator.evaluate()``, consistent with
    how other evaluation tasks handle inference configuration.

    Args:
        benchmarks_path: ``s3://`` URI containing benchmark ``.py`` files with
            ``@task`` decorators.  Required — the container always needs a
            benchmarks location.  Use ``ForgeEvaluator.upload_benchmarks()``
            to upload a local directory to S3 first.
        tasks: List of task dicts, each with a required ``"name"`` key and
            optional ``"path"``, ``"limit"``, ``"epochs"``, and ``"task_args"``
            keys.  Empty list runs all ``@task`` functions in
            ``benchmarks_path`` (eval-set mode).
            Example: ``[{"name": "mmlu", "path": "mmlu/mmlu_sft.py", "limit": 100, "epochs": 1, "task_args": {"subject": "anatomy"}}]``
        output_s3_path: S3 prefix where InspectLens writes eval result JSON
            logs.  Defaults to the ``ForgeEvaluator`` output path.
        output_format: Output format for eval results. One of ``"eval"``,
            ``"csv"``, ``"jsonl"``, ``"json"``. Defaults to ``"eval"``.
        bedrock_model_id: Bedrock model ID or ARN for Bedrock inference mode.
            e.g. ``"us.amazon.nova-micro-v1:0"``.  Falls back to the
            ``ForgeEvaluator.model`` enum value when not set.
        endpoint_name: Existing SageMaker endpoint name to evaluate against.
            Mutually exclusive with ``model_s3_uri`` / ``inference_image_uri``.
        model_s3_uri: S3 URI of model artifacts for creating a new endpoint.
            Requires ``inference_image_uri``.
        inference_image_uri: ECR image URI for the new endpoint container.
            Requires ``model_s3_uri``.
        endpoint_instance_type: Instance type for the new endpoint.
        endpoint_instance_count: Number of instances for the new endpoint.
        endpoint_execution_role_arn: IAM role ARN for the new endpoint.
        context_length: Context length for the new endpoint (as string).
        max_concurrency: Max concurrency for the new endpoint (as string).
        enable_rai: Enable RAI guardrails on the endpoint. Default: True.
        cleanup_endpoint: Delete the endpoint after evaluation. Default: True.
        endpoint_prefix: Prefix for auto-created endpoint names. Default: ``"inspectlens"``.
        endpoint_environment: Extra environment variables for the inference
            endpoint container (e.g. ``{"HF_TOKEN": "hf_xxx"}``).
        extra_args: Additional CLI args forwarded to ``inspect eval``.
        environment: Arbitrary environment variables passed to the SageMaker Training Job
            container (e.g. ``{"HF_TOKEN": "hf_xxx", "HF_HUB_DOWNLOAD_TIMEOUT": "300"}``).
            Any key-value pairs are accepted.
    """

    benchmarks_path: Optional[str] = None
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    output_s3_path: Optional[str] = None
    output_format: Optional[str] = None
    bedrock_model_id: Optional[str] = None
    endpoint_name: Optional[str] = None
    model_s3_uri: Optional[str] = None
    inference_image_uri: Optional[str] = None
    endpoint_instance_type: Optional[str] = None
    endpoint_instance_count: int = 1
    endpoint_execution_role_arn: Optional[str] = None
    context_length: Optional[str] = None
    max_concurrency: Optional[str] = None
    enable_rai: bool = True
    cleanup_endpoint: bool = True
    endpoint_prefix: str = "inspectlens"
    endpoint_environment: Optional[Dict[str, str]] = None
    extra_args: Optional[List[str]] = None
    environment: Optional[Dict[str, str]] = None

    def __post_init__(self) -> None:
        new_endpoint_fields_set = any(
            getattr(self, f) is not None
            for f in ("model_s3_uri", "inference_image_uri", "endpoint_instance_type")
        )
        if self.endpoint_name is not None and new_endpoint_fields_set:
            raise ValueError(
                "Cannot combine endpoint_name with model_s3_uri / "
                "inference_image_uri / endpoint_instance_type. "
                "Set endpoint_name=None to create a new endpoint."
            )
        if self.endpoint_name is None and (
            bool(self.model_s3_uri) != bool(self.inference_image_uri)
        ):
            raise ValueError(
                "Creating a new endpoint requires both model_s3_uri and "
                "inference_image_uri. Provide both or neither."
            )
        self._validate()

    def _validate(self) -> None:
        """Validate InspectLensConfig fields and raise ValueError with actionable messages."""
        validate_inspect_lens_config(
            benchmarks_path=self.benchmarks_path,
            tasks=self.tasks,
            output_s3_path=self.output_s3_path,
            output_format=self.output_format,
            inference_image_uri=self.inference_image_uri,
            model_s3_uri=self.model_s3_uri,
            endpoint_execution_role_arn=self.endpoint_execution_role_arn,
            endpoint_instance_type=self.endpoint_instance_type,
            endpoint_instance_count=self.endpoint_instance_count,
            context_length=self.context_length,
            max_concurrency=self.max_concurrency,
        )

    def _infer_scenario(self) -> str:
        """Return the inference provider mode based on which fields are set."""
        if self.endpoint_name is not None:
            return "existing_endpoint"
        if self.model_s3_uri is not None:
            return "create_endpoint"
        return "bedrock"

    def to_yaml_dict(
        self,
        inference_provider: Dict[str, Any],
        output_s3_path: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Serialize to the dict structure expected by ``inspect_config.yaml``.

        Decoding and eval parameters are read from ``overrides`` with sensible
        defaults, consistent with how other SDK eval tasks handle configuration.
        """
        ov = overrides or {}
        benchmarks_section: Dict[str, Any] = {"tasks": self.tasks}
        if self.benchmarks_path is not None:
            benchmarks_section["s3_path"] = self.benchmarks_path

        decoding: Dict[str, Any] = {
            "temperature": ov.get("temperature", 0.0),
            "top_p": ov.get("top_p", 1.0),
            "top_k": ov.get("top_k", -1),
            "max_tokens": ov.get("max_tokens", 8192),
        }

        config: Dict[str, Any] = {
            "inference_provider": inference_provider,
            "benchmarks": benchmarks_section,
            "eval": {
                "max_connections": ov.get("max_connections", 16),
                "max_retries": ov.get("max_retries", 100),
                "timeout": ov.get("timeout", 600),
                "decoding": decoding,
            },
            "output": {
                "s3_path": output_s3_path,
            },
        }
        if self.output_format:
            config["output"]["output_format"] = self.output_format
        if self.extra_args:
            config["eval"]["extra_args"] = self.extra_args
        return config
