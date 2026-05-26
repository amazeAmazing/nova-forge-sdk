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
"""Unit tests for InspectLensConfig and inspect_lens_validator."""

import os
import tempfile
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch

from amzn_nova_forge.core.enums import Model
from amzn_nova_forge.evaluator.inspect_lens_config import InspectLensConfig
from amzn_nova_forge.validation.inspect_lens_validator import (
    _extract_task_names_from_dir,
    _extract_task_names_from_s3,
)

VALID_BENCHMARKS_PATH = "s3://bucket/benchmarks/"
VALID_TASKS = [{"name": "boolq_pt", "limit": 100}]
VALID_OUTPUT = "s3://bucket/output/"
VALID_ECR = "123456789012.dkr.ecr.us-east-1.amazonaws.com/my-repo:latest"
VALID_ROLE = "arn:aws:iam::123456789012:role/MyRole"


def _make_valid_config(**kwargs):
    """Return a minimal valid InspectLensConfig, bypassing __post_init__ validation."""
    defaults = dict(
        benchmarks_path=VALID_BENCHMARKS_PATH,
        tasks=list(VALID_TASKS),
    )
    defaults.update(kwargs)
    # Bypass __post_init__ to test validate_inspect_lens_config directly
    obj = object.__new__(InspectLensConfig)
    for field, default in InspectLensConfig.__dataclass_fields__.items():
        setattr(
            obj,
            field,
            defaults.get(
                field,
                default.default
                if default.default is not default.default_factory
                else default.default_factory(),
            ),
        )
    for k, v in defaults.items():
        setattr(obj, k, v)
    return obj


class TestInspectLensConfigPostInit(unittest.TestCase):
    """Tests for field-level mutual exclusion checks in __post_init__."""

    def test_valid_bedrock_config(self):
        """Minimal valid config with S3 benchmarks_path should not raise."""
        cfg = InspectLensConfig(
            benchmarks_path=VALID_BENCHMARKS_PATH,
            tasks=VALID_TASKS,
        )
        self.assertEqual(cfg.benchmarks_path, VALID_BENCHMARKS_PATH)

    def test_endpoint_name_with_model_s3_uri_raises(self):
        with self.assertRaises(ValueError) as ctx:
            InspectLensConfig(
                benchmarks_path=VALID_BENCHMARKS_PATH,
                tasks=VALID_TASKS,
                endpoint_name="my-endpoint",
                model_s3_uri="s3://bucket/model/",
                inference_image_uri=VALID_ECR,
            )
        self.assertIn("endpoint_name", str(ctx.exception))

    def test_model_s3_uri_without_inference_image_raises(self):
        with self.assertRaises(ValueError) as ctx:
            InspectLensConfig(
                benchmarks_path=VALID_BENCHMARKS_PATH,
                tasks=VALID_TASKS,
                model_s3_uri="s3://bucket/model/",
            )
        self.assertIn("inference_image_uri", str(ctx.exception))

    def test_inference_image_without_model_s3_uri_raises(self):
        with self.assertRaises(ValueError) as ctx:
            InspectLensConfig(
                benchmarks_path=VALID_BENCHMARKS_PATH,
                tasks=VALID_TASKS,
                inference_image_uri=VALID_ECR,
            )
        self.assertIn("model_s3_uri", str(ctx.exception))

    def test_both_new_endpoint_fields_valid(self):
        """Providing both model_s3_uri and inference_image_uri should not raise."""
        cfg = InspectLensConfig(
            benchmarks_path=VALID_BENCHMARKS_PATH,
            tasks=VALID_TASKS,
            model_s3_uri="s3://bucket/model/",
            inference_image_uri=VALID_ECR,
        )
        self.assertEqual(cfg.model_s3_uri, "s3://bucket/model/")


class TestValidateInspectLensConfig(unittest.TestCase):
    """Tests for validate_inspect_lens_config()."""

    def _assert_error(self, config, fragment: str):
        with self.assertRaises(ValueError) as ctx:
            config._validate()
        self.assertIn(fragment, str(ctx.exception))

    def test_missing_benchmarks_path_raises(self):
        cfg = _make_valid_config(benchmarks_path=None)
        self._assert_error(cfg, "benchmarks_path is required")

    def test_local_benchmarks_path_raises(self):
        cfg = _make_valid_config(benchmarks_path="./my_benchmarks/")
        self._assert_error(cfg, "must be an s3:// URI")

    def test_empty_benchmarks_path_raises(self):
        cfg = _make_valid_config(benchmarks_path="   ")
        self._assert_error(cfg, "must not be an empty string")

    def test_valid_s3_benchmarks_path_passes(self):
        cfg = _make_valid_config(benchmarks_path="s3://bucket/benchmarks/")
        cfg._validate()  # should not raise

    def test_tasks_not_a_list_raises(self):
        cfg = _make_valid_config(tasks="boolq_pt")
        self._assert_error(cfg, "tasks must be a list")

    def test_task_missing_name_raises(self):
        cfg = _make_valid_config(tasks=[{"limit": 10}])
        self._assert_error(cfg, "missing required key 'name'")

    def test_task_not_a_dict_raises(self):
        cfg = _make_valid_config(tasks=["boolq_pt"])
        self._assert_error(cfg, "must be a dict")

    def test_valid_tasks_pass(self):
        cfg = _make_valid_config(tasks=[{"name": "boolq_pt"}, {"name": "mmlu_pro_pt", "limit": 50}])
        cfg._validate()

    def test_empty_tasks_list_passes(self):
        cfg = _make_valid_config(tasks=[])
        cfg._validate()

    def test_invalid_output_s3_path_raises(self):
        cfg = _make_valid_config(output_s3_path="not-an-s3-path")
        self._assert_error(cfg, "output_s3_path must be an s3:// URI")

    def test_valid_output_s3_path_passes(self):
        cfg = _make_valid_config(output_s3_path=VALID_OUTPUT)
        cfg._validate()

    def test_none_output_s3_path_passes(self):
        cfg = _make_valid_config(output_s3_path=None)
        cfg._validate()

    def test_invalid_ecr_uri_raises(self):
        cfg = _make_valid_config(inference_image_uri="not-an-ecr-uri")
        self._assert_error(cfg, "inference_image_uri must be a valid ECR URI")

    def test_valid_ecr_uri_passes(self):
        cfg = _make_valid_config(inference_image_uri=VALID_ECR)
        cfg._validate()

    def test_invalid_model_s3_uri_raises(self):
        cfg = _make_valid_config(model_s3_uri="not-s3")
        self._assert_error(cfg, "model_s3_uri must be an s3:// URI")

    def test_valid_model_s3_uri_passes(self):
        cfg = _make_valid_config(model_s3_uri="s3://bucket/model/")
        cfg._validate()

    def test_invalid_role_arn_raises(self):
        cfg = _make_valid_config(endpoint_execution_role_arn="not-an-arn")
        self._assert_error(cfg, "endpoint_execution_role_arn must be a valid IAM role ARN")

    def test_valid_role_arn_passes(self):
        cfg = _make_valid_config(endpoint_execution_role_arn=VALID_ROLE)
        cfg._validate()

    def test_mlflow_fields_not_on_config(self):
        """mlflow and hf_token fields should not exist on InspectLensConfig."""
        cfg = InspectLensConfig(
            benchmarks_path=VALID_BENCHMARKS_PATH,
            tasks=VALID_TASKS,
        )
        self.assertFalse(hasattr(cfg, "mlflow_tracking_arn"))
        self.assertFalse(hasattr(cfg, "mlflow_experiment_name"))
        self.assertFalse(hasattr(cfg, "mlflow_tracing"))
        self.assertFalse(hasattr(cfg, "mlflow_log_artifacts"))
        self.assertFalse(hasattr(cfg, "hf_token"))

    def test_invalid_output_format_raises(self):
        cfg = _make_valid_config(output_format="xml")
        self._assert_error(cfg, "output_format must be one of")

    def test_valid_output_formats_pass(self):
        for fmt in ("eval", "csv", "jsonl", "json"):
            cfg = _make_valid_config(output_format=fmt)
            cfg._validate()

    def test_none_output_format_passes(self):
        cfg = _make_valid_config(output_format=None)
        cfg._validate()

    def test_invalid_endpoint_instance_type_raises(self):
        cfg = _make_valid_config(endpoint_instance_type="p5.48xlarge")
        self._assert_error(cfg, "must start with 'ml.'")

    def test_valid_endpoint_instance_type_passes(self):
        cfg = _make_valid_config(endpoint_instance_type="ml.p5.48xlarge")
        cfg._validate()

    def test_invalid_endpoint_instance_count_raises(self):
        cfg = _make_valid_config(endpoint_instance_count=0)
        self._assert_error(cfg, "endpoint_instance_count must be >= 1")

    def test_invalid_context_length_raises(self):
        cfg = _make_valid_config(context_length="abc")
        self._assert_error(cfg, "context_length must be a positive integer")

    def test_zero_context_length_raises(self):
        cfg = _make_valid_config(context_length="0")
        self._assert_error(cfg, "context_length must be a positive integer")

    def test_valid_context_length_passes(self):
        cfg = _make_valid_config(context_length="12000")
        cfg._validate()

    def test_invalid_max_concurrency_raises(self):
        cfg = _make_valid_config(max_concurrency="foo")
        self._assert_error(cfg, "max_concurrency must be a positive integer")

    def test_valid_max_concurrency_passes(self):
        cfg = _make_valid_config(max_concurrency="16")
        cfg._validate()

    def test_task_with_invalid_path_raises(self):
        cfg = _make_valid_config(tasks=[{"name": "mmlu", "path": "mmlu/mmlu_sft.txt"}])
        self._assert_error(cfg, "path must be a relative path ending in .py")

    def test_task_with_valid_path_passes(self):
        cfg = _make_valid_config(tasks=[{"name": "mmlu", "path": "mmlu/mmlu_sft.py"}])
        cfg._validate()

    def test_task_with_invalid_epochs_raises(self):
        cfg = _make_valid_config(tasks=[{"name": "mmlu", "epochs": 0}])
        self._assert_error(cfg, "epochs must be an int >= 1")

    def test_task_with_valid_epochs_passes(self):
        cfg = _make_valid_config(tasks=[{"name": "mmlu", "epochs": 3}])
        cfg._validate()

    def test_task_with_invalid_task_args_raises(self):
        cfg = _make_valid_config(tasks=[{"name": "mmlu", "task_args": "invalid"}])
        self._assert_error(cfg, "task_args must be a dict")

    def test_task_with_valid_task_args_passes(self):
        cfg = _make_valid_config(tasks=[{"name": "mmlu", "task_args": {"subject": "anatomy"}}])
        cfg._validate()

    def test_multiple_errors_raised_together(self):
        cfg = _make_valid_config(
            benchmarks_path=None,
            output_s3_path="bad-path",
            inference_image_uri="not-ecr",
        )
        with self.assertRaises(ValueError) as ctx:
            cfg._validate()
        msg = str(ctx.exception)
        self.assertIn("3 error(s)", msg)
        self.assertIn("benchmarks_path is required", msg)
        self.assertIn("output_s3_path", msg)
        self.assertIn("inference_image_uri", msg)


class TestBenchmarksPathMustBeS3(unittest.TestCase):
    """Tests that benchmarks_path must be an S3 URI."""

    def test_local_path_raises(self):
        cfg = _make_valid_config(benchmarks_path="/local/path/benchmarks")
        with self.assertRaises(ValueError) as ctx:
            cfg._validate()
        self.assertIn("must be an s3:// URI", str(ctx.exception))
        self.assertIn("upload_benchmarks", str(ctx.exception))

    def test_relative_path_raises(self):
        cfg = _make_valid_config(benchmarks_path="./my_benchmarks/")
        with self.assertRaises(ValueError) as ctx:
            cfg._validate()
        self.assertIn("must be an s3:// URI", str(ctx.exception))

    def test_s3_path_passes(self):
        cfg = _make_valid_config(benchmarks_path="s3://bucket/benchmarks/")
        cfg._validate()  # should not raise


class TestExtractTaskNamesFromDir(unittest.TestCase):
    def test_extracts_task_decorated_functions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = "@task\ndef boolq_pt():\n    pass\n\n@task\ndef mmlu_pro_pt():\n    pass\n"
            with open(os.path.join(tmpdir, "tasks.py"), "w") as f:
                f.write(source)
            result = _extract_task_names_from_dir(tmpdir)
            self.assertEqual(result, {"boolq_pt", "mmlu_pro_pt"})

    def test_ignores_non_task_functions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = "def helper():\n    pass\n\n@task\ndef real_task():\n    pass\n"
            with open(os.path.join(tmpdir, "tasks.py"), "w") as f:
                f.write(source)
            result = _extract_task_names_from_dir(tmpdir)
            self.assertEqual(result, {"real_task"})

    def test_ignores_non_py_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "README.md"), "w") as f:
                f.write("@task\ndef fake_task():\n    pass\n")
            result = _extract_task_names_from_dir(tmpdir)
            self.assertEqual(result, set())

    def test_nonexistent_dir_returns_empty(self):
        result = _extract_task_names_from_dir("/nonexistent/path")
        self.assertEqual(result, set())

    def test_multiple_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ("boolq_pt", "mmlu_pro_pt", "arc_c_pt"):
                with open(os.path.join(tmpdir, f"{name}.py"), "w") as f:
                    f.write(f"@task\ndef {name}():\n    pass\n")
            result = _extract_task_names_from_dir(tmpdir)
            self.assertEqual(result, {"boolq_pt", "mmlu_pro_pt", "arc_c_pt"})


class TestExtractTaskNamesFromS3(unittest.TestCase):
    @patch("boto3.client")
    def test_extracts_task_names_from_s3(self, mock_client):
        source = b"@task\ndef boolq_pt():\n    pass\n\n@task\ndef mmlu_pro_pt():\n    pass\n"
        mock_s3 = MagicMock()
        mock_client.return_value = mock_s3
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [{"Contents": [{"Key": "benchmarks/boolq_pt.py"}]}]
        mock_s3.get_object.return_value = {"Body": BytesIO(source)}

        result = _extract_task_names_from_s3("s3://bucket/benchmarks/")
        self.assertEqual(result, {"boolq_pt", "mmlu_pro_pt"})

    @patch("boto3.client")
    def test_s3_access_failure_returns_empty(self, mock_client):
        mock_client.side_effect = Exception("no credentials")
        result = _extract_task_names_from_s3("s3://bucket/benchmarks/")
        self.assertEqual(result, set())

    @patch("boto3.client")
    def test_skips_non_py_keys(self, mock_client):
        mock_s3 = MagicMock()
        mock_client.return_value = mock_s3
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": "benchmarks/README.md"}, {"Key": "benchmarks/config.yaml"}]}
        ]
        result = _extract_task_names_from_s3("s3://bucket/benchmarks/")
        self.assertEqual(result, set())
        mock_s3.get_object.assert_not_called()

    @patch("amzn_nova_forge.validation.inspect_lens_validator.boto3")
    def test_task_name_mismatch_via_s3_raises(self, mock_boto3):
        from amzn_nova_forge.validation.inspect_lens_validator import (
            validate_task_names_against_benchmarks,
        )

        source = b"@task\ndef boolq_pt():\n    pass\n"
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [{"Contents": [{"Key": "benchmarks/boolq_pt.py"}]}]
        mock_s3.get_object.return_value = {"Body": BytesIO(source)}

        errors = validate_task_names_against_benchmarks(
            tasks=[{"name": "mmlu_0_shot"}],
            benchmarks_path="s3://bucket/benchmarks/",
        )
        self.assertTrue(len(errors) > 0)
        self.assertTrue(any("mmlu_0_shot" in e for e in errors))
        self.assertTrue(any("boolq_pt" in e for e in errors))


class TestInferScenario(unittest.TestCase):
    def test_bedrock_scenario(self):
        cfg = InspectLensConfig(benchmarks_path=VALID_BENCHMARKS_PATH, tasks=VALID_TASKS)
        self.assertEqual(cfg._infer_scenario(), "bedrock")

    def test_existing_endpoint_scenario(self):
        cfg = InspectLensConfig(
            benchmarks_path=VALID_BENCHMARKS_PATH,
            tasks=VALID_TASKS,
            endpoint_name="my-endpoint",
        )
        self.assertEqual(cfg._infer_scenario(), "existing_endpoint")

    def test_create_endpoint_scenario(self):
        cfg = InspectLensConfig(
            benchmarks_path=VALID_BENCHMARKS_PATH,
            tasks=VALID_TASKS,
            model_s3_uri="s3://bucket/model/",
            inference_image_uri=VALID_ECR,
        )
        self.assertEqual(cfg._infer_scenario(), "create_endpoint")


class TestToYamlDict(unittest.TestCase):
    def _bedrock_provider(self):
        return {"bedrock": {"model_id": "us.amazon.nova-2-lite-v1:0", "region": "us-east-1"}}

    def test_benchmarks_s3_path_included_when_set(self):
        cfg = InspectLensConfig(
            benchmarks_path="s3://bucket/benchmarks/",
            tasks=[{"name": "boolq_pt"}],
        )
        result = cfg.to_yaml_dict(self._bedrock_provider(), "s3://bucket/output/")
        self.assertEqual(result["benchmarks"]["s3_path"], "s3://bucket/benchmarks/")

    def test_benchmarks_s3_path_omitted_when_none(self):
        cfg = object.__new__(InspectLensConfig)
        for field, default in InspectLensConfig.__dataclass_fields__.items():
            try:
                setattr(cfg, field, default.default)
            except Exception:
                setattr(cfg, field, default.default_factory())
        cfg.benchmarks_path = None
        cfg.tasks = [{"name": "boolq_pt"}]
        result = cfg.to_yaml_dict(self._bedrock_provider(), "s3://bucket/output/")
        self.assertNotIn("s3_path", result["benchmarks"])

    def test_decoding_overrides_applied(self):
        cfg = InspectLensConfig(
            benchmarks_path=VALID_BENCHMARKS_PATH,
            tasks=VALID_TASKS,
        )
        result = cfg.to_yaml_dict(
            self._bedrock_provider(),
            "s3://bucket/output/",
            overrides={"temperature": 0.5, "max_tokens": 2048},
        )
        self.assertEqual(result["eval"]["decoding"]["temperature"], 0.5)
        self.assertEqual(result["eval"]["decoding"]["max_tokens"], 2048)

    def test_mlflow_section_not_in_to_yaml_dict(self):
        """MLflow tracking is injected by ForgeEvaluator, not to_yaml_dict."""
        cfg = InspectLensConfig(
            benchmarks_path=VALID_BENCHMARKS_PATH,
            tasks=VALID_TASKS,
        )
        result = cfg.to_yaml_dict(self._bedrock_provider(), "s3://bucket/output/")
        self.assertNotIn("tracking", result)

    def test_mlflow_section_omitted_when_not_set(self):
        cfg = InspectLensConfig(
            benchmarks_path=VALID_BENCHMARKS_PATH,
            tasks=VALID_TASKS,
        )
        result = cfg.to_yaml_dict(self._bedrock_provider(), "s3://bucket/output/")
        self.assertNotIn("tracking", result)

    def test_output_s3_path_in_result(self):
        cfg = InspectLensConfig(
            benchmarks_path=VALID_BENCHMARKS_PATH,
            tasks=VALID_TASKS,
        )
        result = cfg.to_yaml_dict(self._bedrock_provider(), "s3://bucket/my-output/")
        self.assertEqual(result["output"]["s3_path"], "s3://bucket/my-output/")

    def test_extra_args_included_when_set(self):
        cfg = InspectLensConfig(
            benchmarks_path=VALID_BENCHMARKS_PATH,
            tasks=VALID_TASKS,
            extra_args=["-M", "completion_mode=True"],
        )
        result = cfg.to_yaml_dict(self._bedrock_provider(), "s3://bucket/output/")
        self.assertEqual(result["eval"]["extra_args"], ["-M", "completion_mode=True"])

    def test_extra_args_omitted_when_not_set(self):
        cfg = InspectLensConfig(
            benchmarks_path=VALID_BENCHMARKS_PATH,
            tasks=VALID_TASKS,
        )
        result = cfg.to_yaml_dict(self._bedrock_provider(), "s3://bucket/output/")
        self.assertNotIn("extra_args", result["eval"])

    def test_top_k_in_decoding(self):
        cfg = InspectLensConfig(
            benchmarks_path=VALID_BENCHMARKS_PATH,
            tasks=VALID_TASKS,
        )
        result = cfg.to_yaml_dict(self._bedrock_provider(), "s3://bucket/output/")
        self.assertEqual(result["eval"]["decoding"]["top_k"], -1)

    def test_top_k_override(self):
        cfg = InspectLensConfig(
            benchmarks_path=VALID_BENCHMARKS_PATH,
            tasks=VALID_TASKS,
        )
        result = cfg.to_yaml_dict(
            self._bedrock_provider(), "s3://bucket/output/", overrides={"top_k": 50}
        )
        self.assertEqual(result["eval"]["decoding"]["top_k"], 50)

    def test_output_format_included_when_set(self):
        cfg = InspectLensConfig(
            benchmarks_path=VALID_BENCHMARKS_PATH,
            tasks=VALID_TASKS,
            output_format="csv",
        )
        result = cfg.to_yaml_dict(self._bedrock_provider(), "s3://bucket/output/")
        self.assertEqual(result["output"]["output_format"], "csv")

    def test_output_format_omitted_when_none(self):
        cfg = InspectLensConfig(
            benchmarks_path=VALID_BENCHMARKS_PATH,
            tasks=VALID_TASKS,
        )
        result = cfg.to_yaml_dict(self._bedrock_provider(), "s3://bucket/output/")
        self.assertNotIn("output_format", result["output"])

    def test_default_eval_values_match_spec(self):
        cfg = InspectLensConfig(
            benchmarks_path=VALID_BENCHMARKS_PATH,
            tasks=VALID_TASKS,
        )
        result = cfg.to_yaml_dict(self._bedrock_provider(), "s3://bucket/output/")
        self.assertEqual(result["eval"]["max_connections"], 16)
        self.assertEqual(result["eval"]["max_retries"], 100)
        self.assertEqual(result["eval"]["timeout"], 600)
        self.assertEqual(result["eval"]["decoding"]["temperature"], 0.0)
        self.assertEqual(result["eval"]["decoding"]["top_p"], 1.0)
        self.assertEqual(result["eval"]["decoding"]["top_k"], -1)
        self.assertEqual(result["eval"]["decoding"]["max_tokens"], 8192)


class TestModelBedrockModelId(unittest.TestCase):
    """Tests for Model.bedrock_model_id — context-length suffix stripping."""

    def test_nova_micro_strips_suffix(self):
        # amazon.nova-micro-v1:0:128k → us.amazon.nova-micro-v1:0
        self.assertEqual(Model.NOVA_MICRO.bedrock_model_id, "us.amazon.nova-micro-v1:0")

    def test_nova_lite_strips_suffix(self):
        # amazon.nova-lite-v1:0:300k → us.amazon.nova-lite-v1:0
        self.assertEqual(Model.NOVA_LITE.bedrock_model_id, "us.amazon.nova-lite-v1:0")

    def test_nova_lite_2_strips_suffix(self):
        # amazon.nova-2-lite-v1:0:256k → us.amazon.nova-2-lite-v1:0
        self.assertEqual(Model.NOVA_LITE_2.bedrock_model_id, "us.amazon.nova-2-lite-v1:0")

    def test_nova_pro_strips_suffix(self):
        # amazon.nova-pro-v1:0:300k → us.amazon.nova-pro-v1:0
        self.assertEqual(Model.NOVA_PRO.bedrock_model_id, "us.amazon.nova-pro-v1:0")

    def test_all_models_start_with_us_prefix(self):
        for model in Model:
            self.assertTrue(
                model.bedrock_model_id.startswith("us."),
                f"{model.name}.bedrock_model_id should start with 'us.', got: {model.bedrock_model_id}",
            )

    def test_no_model_has_context_length_suffix(self):
        for model in Model:
            bid = model.bedrock_model_id
            # Should not end with :128k, :256k, :300k etc.
            self.assertNotRegex(
                bid,
                r":\d+k$",
                f"{model.name}.bedrock_model_id should not have context-length suffix, got: {bid}",
            )


if __name__ == "__main__":
    unittest.main()
