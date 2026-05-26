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
"""Validation for InspectLensConfig."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Set

import boto3

from amzn_nova_forge.util.logging import logger
from amzn_nova_forge.util.s3_utils import parse_s3_uri

_ECR_URI_REGEX = re.compile(r"^\d{12}\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com/[^:]+:[^:]+$")
_IAM_ROLE_ARN_REGEX = re.compile(r"^arn:aws[^:]*:iam::\d{12}:role/.+$")
_TASK_DECORATOR_REGEX = re.compile(r"@task\b[^\n]*\n(?:@[^\n]*\n)*def\s+(\w+)")


def _extract_task_names_from_dir(local_dir: str) -> Set[str]:
    """Return all @task-decorated function names found in .py files under local_dir."""
    task_names: Set[str] = set()
    expanded = os.path.expanduser(local_dir)
    if not os.path.isdir(expanded):
        return task_names
    for fname in os.listdir(expanded):
        if not fname.endswith(".py"):
            continue
        fpath = os.path.join(expanded, fname)
        try:
            with open(fpath) as f:
                source = f.read()
        except OSError:
            continue
        for match in _TASK_DECORATOR_REGEX.finditer(source):
            task_names.add(match.group(1))
    return task_names


def _extract_task_names_from_s3(s3_uri: str, region: Optional[str] = None) -> Set[str]:
    """Return all @task-decorated function names found in .py files at an S3 prefix.

    Returns an empty set silently if S3 access fails (permissions, network) —
    we don't block the user with a validation error they can't fix at config time.
    """
    task_names: Set[str] = set()
    bucket, prefix = parse_s3_uri(s3_uri)

    try:
        s3 = boto3.client("s3", region_name=region)
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith(".py"):
                    continue
                try:
                    response = s3.get_object(Bucket=bucket, Key=key)
                    source = response["Body"].read().decode("utf-8")
                    for match in _TASK_DECORATOR_REGEX.finditer(source):
                        task_names.add(match.group(1))
                except Exception as e:
                    logger.debug("Skipping S3 key '%s' during task extraction: %s", key, e)
                    continue
    except Exception as e:
        logger.debug("Could not list S3 prefix '%s' for task validation: %s", s3_uri, e)
    return task_names


def _validate_benchmarks_path(benchmarks_path: Optional[str], errors: List[str]) -> None:
    """Validate benchmarks_path value and append errors to the list."""
    if not benchmarks_path:
        errors.append(
            "benchmarks_path is required. Provide an s3:// URI containing benchmark .py files "
            "with @task decorators. Use evaluator.upload_benchmarks(local_dir, s3_path) to "
            "upload a local directory first.\n"
            "    Example: benchmarks_path='s3://your-bucket/benchmarks/my_benchmarks/'"
        )
    elif not benchmarks_path.strip():
        errors.append("benchmarks_path must not be an empty string.")
    elif not benchmarks_path.strip().startswith("s3://"):
        errors.append(
            f"benchmarks_path must be an s3:// URI, got: '{benchmarks_path}'. "
            f"Use evaluator.upload_benchmarks(local_dir, s3_path) to upload a local "
            f"directory first, then pass the returned S3 URI as benchmarks_path."
        )


_VALID_OUTPUT_FORMATS = ("eval", "csv", "jsonl", "json")


def _validate_tasks(
    tasks: Any,
    errors: List[str],
) -> None:
    """Validate tasks structure (format only — no S3 calls).

    S3-based task name matching is deferred to job submission time via
    validate_task_names_against_benchmarks() to avoid network calls in __post_init__.
    """
    if not isinstance(tasks, list):
        errors.append("tasks must be a list of dicts, e.g. [{'name': 'boolq_pt', 'limit': 100}].")
        return

    for i, t in enumerate(tasks):
        if not isinstance(t, dict):
            errors.append(f"tasks[{i}] must be a dict with a 'name' key, got: {t!r}")
            continue
        if "name" not in t:
            errors.append(
                f"tasks[{i}] is missing required key 'name'. "
                f"Each task must have at least {{'name': '<task_function_name>'}}."
            )
        if "path" in t and t["path"] is not None:
            path_val = t["path"]
            if not isinstance(path_val, str) or not path_val.endswith(".py"):
                errors.append(
                    f"tasks[{i}].path must be a relative path ending in .py, got: {path_val!r}"
                )
        if "limit" in t and t["limit"] is not None:
            limit_val = t["limit"]
            if not isinstance(limit_val, int) or limit_val < 1:
                errors.append(f"tasks[{i}].limit must be an int >= 1, got: {limit_val!r}")
        if "epochs" in t and t["epochs"] is not None:
            epochs_val = t["epochs"]
            if not isinstance(epochs_val, int) or epochs_val < 1:
                errors.append(f"tasks[{i}].epochs must be an int >= 1, got: {epochs_val!r}")
        if "task_args" in t and t["task_args"] is not None:
            task_args_val = t["task_args"]
            if not isinstance(task_args_val, dict):
                errors.append(
                    f"tasks[{i}].task_args must be a dict, got: {type(task_args_val).__name__}"
                )


def validate_task_names_against_benchmarks(
    tasks: Any,
    benchmarks_path: Optional[str],
    region: Optional[str] = None,
) -> List[str]:
    """Validate task names against @task-decorated functions in the benchmarks S3 path.

    This is intentionally separate from validate_inspect_lens_config() to avoid
    making S3 API calls during InspectLensConfig construction (__post_init__).
    Call this at job submission time when region and credentials are available.

    Returns a list of error strings (empty if all tasks match).
    """
    errors: List[str] = []

    if (
        not benchmarks_path
        or not tasks
        or not isinstance(tasks, list)
        or any(not isinstance(t, dict) or "name" not in t for t in tasks)
    ):
        return errors

    path = benchmarks_path.strip()
    if not path.startswith("s3://"):
        return errors

    known_tasks = _extract_task_names_from_s3(path, region=region)
    if not known_tasks:
        return errors

    for t in tasks:
        if not isinstance(t, dict) or "name" not in t:
            continue
        task_name = t["name"]
        if not any(task_name in k for k in known_tasks):
            errors.append(
                f"Task '{task_name}' was not found in any @task-decorated function "
                f"in '{path}'.\n"
                f"    Available @task functions: {sorted(known_tasks)}\n"
                f"    The container matches tasks by substring — make sure the task "
                f"name matches (or is a substring of) a @task function name in your "
                f"benchmark .py files."
            )
    return errors


def _validate_s3_and_arn_fields(
    output_s3_path: Optional[str],
    inference_image_uri: Optional[str],
    model_s3_uri: Optional[str],
    endpoint_execution_role_arn: Optional[str],
    errors: List[str],
) -> None:
    """Validate S3 URIs and ARN fields, appending errors to the list."""
    if output_s3_path is not None and not output_s3_path.startswith("s3://"):
        errors.append(f"output_s3_path must be an s3:// URI, got: '{output_s3_path}'")

    if inference_image_uri is not None and not _ECR_URI_REGEX.match(inference_image_uri):
        errors.append(
            f"inference_image_uri must be a valid ECR URI in the format "
            f"<account>.dkr.ecr.<region>.amazonaws.com/<repo>:<tag>. "
            f"Got: '{inference_image_uri}'"
        )

    if model_s3_uri is not None and not model_s3_uri.startswith("s3://"):
        errors.append(f"model_s3_uri must be an s3:// URI, got: '{model_s3_uri}'")

    if endpoint_execution_role_arn is not None and not _IAM_ROLE_ARN_REGEX.match(
        endpoint_execution_role_arn
    ):
        errors.append(
            f"endpoint_execution_role_arn must be a valid IAM role ARN "
            f"(arn:aws:iam::<account>:role/<name>). "
            f"Got: '{endpoint_execution_role_arn}'"
        )


def _validate_output_format(output_format: Optional[str], errors: List[str]) -> None:
    """Validate output_format value if provided."""
    if output_format is not None and output_format not in _VALID_OUTPUT_FORMATS:
        errors.append(
            f"output_format must be one of {_VALID_OUTPUT_FORMATS}, got: '{output_format}'"
        )


def _validate_endpoint_fields(
    endpoint_instance_type: Optional[str],
    endpoint_instance_count: int,
    context_length: Optional[str],
    max_concurrency: Optional[str],
    errors: List[str],
) -> None:
    """Validate endpoint-related numeric and string fields."""
    if endpoint_instance_type is not None and not endpoint_instance_type.startswith("ml."):
        errors.append(
            f"endpoint_instance_type must start with 'ml.', got: '{endpoint_instance_type}'"
        )
    if endpoint_instance_count < 1:
        errors.append(f"endpoint_instance_count must be >= 1, got: {endpoint_instance_count}")
    if context_length is not None:
        if not context_length.isdigit() or int(context_length) <= 0:
            errors.append(
                f"context_length must be a positive integer as string, got: '{context_length}'"
            )
    if max_concurrency is not None:
        if not max_concurrency.isdigit() or int(max_concurrency) <= 0:
            errors.append(
                f"max_concurrency must be a positive integer as string, got: '{max_concurrency}'"
            )


def validate_inspect_lens_config(
    benchmarks_path: Optional[str],
    tasks: Any,
    output_s3_path: Optional[str],
    output_format: Optional[str],
    inference_image_uri: Optional[str],
    model_s3_uri: Optional[str],
    endpoint_execution_role_arn: Optional[str],
    endpoint_instance_type: Optional[str] = None,
    endpoint_instance_count: int = 1,
    context_length: Optional[str] = None,
    max_concurrency: Optional[str] = None,
) -> None:
    """Validate InspectLensConfig fields and raise ValueError with actionable messages.

    Takes individual field values rather than the config object to avoid
    circular imports between the config and validator modules.

    Raises:
        ValueError: If any required fields are missing or malformed.
    """
    errors: List[str] = []
    _validate_benchmarks_path(benchmarks_path, errors)
    _validate_tasks(tasks, errors)
    _validate_output_format(output_format, errors)
    _validate_s3_and_arn_fields(
        output_s3_path, inference_image_uri, model_s3_uri, endpoint_execution_role_arn, errors
    )
    _validate_endpoint_fields(
        endpoint_instance_type, endpoint_instance_count, context_length, max_concurrency, errors
    )

    if errors:
        bullet_list = "\n".join(f"  - {e}" for e in errors)
        raise ValueError(
            f"InspectLensConfig validation failed with {len(errors)} error(s):\n{bullet_list}"
        )
