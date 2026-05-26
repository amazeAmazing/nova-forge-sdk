# InspectLens Evaluation Guide

InspectLens is a job-based evaluation framework in the Nova Forge SDK. It runs [Inspect AI](https://inspect.ai-safety-institute.org.uk/) benchmarks against your Nova model via a SageMaker Training Job — no GPU required, since inference is delegated to a Bedrock endpoint or a SageMaker endpoint.

## Overview

| Concept | Description |
|---------|-------------|
| **Orchestrator** | CPU-only SageMaker Training Job (e.g. `ml.m5.large`) |
| **Inference** | Bedrock, existing SageMaker endpoint, or SDK-created endpoint |
| **Benchmarks** | Python files with `@task` decorators uploaded to S3 |
| **Results** | JSON logs written to an S3 output path |

## Prerequisites

- AWS credentials configured
- S3 bucket for evaluation artifacts and results
- IAM execution role with SageMaker and S3 permissions
- Nova Forge SDK installed (`pip install .` from the repo root)

## Quick Start

```python
from amzn_nova_forge import *
from amzn_nova_forge.core.enums import EvaluationTask, Model
from amzn_nova_forge.evaluator import ForgeEvaluator, InspectLensConfig

# 1. Configure infrastructure (CPU-only — no GPU needed)
eval_infra = SMTJRuntimeManager(
    instance_type="ml.m5.large",
    instance_count=1,
    execution_role="arn:aws:iam::123456789012:role/YourSageMakerRole",
)

# 2. Initialize the evaluator
evaluator = ForgeEvaluator(
    model=Model.NOVA_LITE_2,
    infra=eval_infra,
    config=ForgeConfig(output_s3_path="s3://your-bucket/inspectlens/output"),
)

# 3. Upload benchmarks to S3
benchmarks_s3_uri = evaluator.upload_benchmarks(
    "./my_benchmarks",
    "s3://your-bucket/inspectlens/benchmarks/my_benchmarks/",
)

# 4. Configure and run evaluation
inspect_config = InspectLensConfig(
    benchmarks_path=benchmarks_s3_uri,
    tasks=[
        {"name": "boolq_pt", "limit": 100},
        {"name": "mmlu_pro_pt", "limit": 50},
    ],
    output_s3_path="s3://your-bucket/inspectlens/results/",
)

result = evaluator.evaluate(
    job_name="my-inspectlens-eval",
    eval_task=EvaluationTask.INSPECT_LENS,
    inspect_lens_config=inspect_config,
)

print(f"Job ID: {result.job_id}")
print(f"Results: {result.eval_output_path}")
```

## Writing Benchmarks

Benchmarks are Python files with `@task` decorators. They can wrap built-in `inspect-evals` tasks or define custom evaluation logic.

```python
# my_benchmarks/boolq_pt.py
from inspect_ai import task
from inspect_evals.boolq import boolq

@task
def boolq_pt():
    return boolq()
```

```python
# my_benchmarks/mmlu_pro_pt.py
from inspect_ai import task
from inspect_evals.mmlu_pro import mmlu_pro

@task
def mmlu_pro_pt():
    return mmlu_pro()
```

Upload benchmarks to S3 before starting a job:

```python
benchmarks_s3_uri = evaluator.upload_benchmarks("./my_benchmarks", "s3://bucket/benchmarks/")
```

Only `.py`, `pyproject.toml`, and `requirements.txt` files are uploaded.

## Inference Providers

InspectLens supports three inference modes configured via `InspectLensConfig`.

### Option A: Bedrock (default)

The simplest path — no endpoint management needed. If `bedrock_model_id` is omitted, the SDK uses the cross-region inference profile for the `model` passed to `ForgeEvaluator`.

```python
config = InspectLensConfig(
    benchmarks_path=benchmarks_s3_uri,
    # bedrock_model_id="us.amazon.nova-2-lite-v1:0",  # optional override
    tasks=[{"name": "boolq_pt", "limit": 100}],
    output_s3_path="s3://bucket/results/bedrock-eval/",
)
```

### Option B: Existing SageMaker Endpoint

Use when you already have a deployed SageMaker endpoint.

```python
config = InspectLensConfig(
    benchmarks_path=benchmarks_s3_uri,
    endpoint_name="my-deployed-nova-endpoint",
    tasks=[{"name": "boolq_pt", "limit": 100}],
    output_s3_path="s3://bucket/results/endpoint-eval/",
)
```

### Option C: Create a New SageMaker Endpoint

Use when you have model artifacts in S3 and want the SDK to spin up an endpoint, run the evaluation, and clean up afterwards.

```python
config = InspectLensConfig(
    benchmarks_path=benchmarks_s3_uri,
    model_s3_uri="s3://bucket/model-artifacts/",
    inference_image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/your-image:latest",
    endpoint_instance_type="ml.g5.12xlarge",
    endpoint_execution_role_arn="arn:aws:iam::123456789012:role/YourRole",
    cleanup_endpoint=True,  # delete endpoint after eval (default)
    tasks=[{"name": "boolq_pt", "limit": 100}],
    output_s3_path="s3://bucket/results/new-endpoint-eval/",
)
```

## InspectLensConfig Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `benchmarks_path` | `str` | Required | S3 URI containing benchmark `.py` files |
| `tasks` | `list[dict]` | `[]` | Task dicts with `name`, optional `limit` and `epochs`. Empty = run all tasks |
| `output_s3_path` | `str` | Auto | S3 prefix for result JSON logs |
| `output_format` | `str` | `"eval"` | Output format: `"eval"`, `"csv"`, `"jsonl"`, `"json"` |
| `bedrock_model_id` | `str` | Auto | Bedrock model ID/ARN (Bedrock mode) |
| `endpoint_name` | `str` | None | Existing SageMaker endpoint (endpoint mode) |
| `model_s3_uri` | `str` | None | Model artifacts S3 URI (create-endpoint mode) |
| `inference_image_uri` | `str` | None | ECR image for new endpoint (requires `model_s3_uri`) |
| `endpoint_instance_type` | `str` | None | Instance type for new endpoint |
| `endpoint_instance_count` | `int` | 1 | Instance count for new endpoint |
| `endpoint_execution_role_arn` | `str` | None | IAM role for new endpoint |
| `context_length` | `str` | None | Context length for new endpoint |
| `max_concurrency` | `str` | None | Max concurrency for new endpoint |
| `enable_rai` | `bool` | True | Enable RAI guardrails on the endpoint |
| `cleanup_endpoint` | `bool` | True | Delete endpoint after evaluation |
| `endpoint_prefix` | `str` | `"inspectlens"` | Prefix for auto-created endpoint names |
| `endpoint_environment` | `dict[str, str]` | None | Env vars for the inference endpoint container |
| `extra_args` | `list[str]` | None | Additional CLI args forwarded to `inspect eval` |
| `environment` | `dict[str, str]` | None | Env vars for the container (e.g. `{"HF_TOKEN": "..."}`) |

To override the InspectLens orchestrator container image, set `image_uri` on `ForgeConfig` (not on `InspectLensConfig`):

```python
config = ForgeConfig(
    output_s3_path="s3://bucket/output/",
    image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/sagemaker-inspect-ai",
)
```

The default image is `763104351884.dkr.ecr.{region}.amazonaws.com/sagemaker-inspect-ai`, resolved automatically from the evaluator's region.

## Decoding Overrides

Decoding parameters are passed via the `overrides` dict on `evaluator.evaluate()`:

```python
result = evaluator.evaluate(
    job_name="my-eval",
    eval_task=EvaluationTask.INSPECT_LENS,
    inspect_lens_config=config,
    overrides={
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 8192,
        "max_connections": 16,
        "max_retries": 100,
        "timeout": 600,
    },
)
```

| Override | Default | Description |
|---------|---------|-------------|
| `temperature` | 0.0 | Sampling temperature |
| `top_p` | 1.0 | Nucleus sampling |
| `max_tokens` | 8192 | Max output tokens |
| `max_connections` | 16 | Parallel inference connections |
| `max_retries` | 100 | Retry count for failed requests |
| `timeout` | 600 | Request timeout (seconds) |

## Dry Run Validation

Use `dry_run=True` to validate your configuration locally without submitting a job:

```python
evaluator.evaluate(
    job_name="my-eval",
    eval_task=EvaluationTask.INSPECT_LENS,
    inspect_lens_config=config,
    dry_run=True,
)
```

This validates config fields, resolves the inference provider, and saves the generated YAML locally.

## Monitoring and Results

### Check job status

```python
status, message = result.get_job_status()
print(f"Status: {status} — {message}")
```

### Stream CloudWatch logs

```python
evaluator.get_logs(job_result=result, limit=50, start_from_head=False)
```

### Retrieve results

Results are written as JSON logs to the configured `output_s3_path` after the job completes:

```python
print(f"Results location: {result.eval_output_path}")
```

## MLflow Tracking

Pass an `MLflowMonitor` to `ForgeConfig` and the SDK automatically populates the MLflow tracking section in the InspectLens config.

Both tracking server ARNs (`mlflow-tracking-server/...`) and app ARNs (`mlflow-app/...`) are supported.

```python
from amzn_nova_forge.monitor import MLflowMonitor

mlflow_monitor = MLflowMonitor(
    tracking_uri="arn:aws:sagemaker:us-east-1:123456789012:mlflow-app/my-app",
    experiment_name="nova-inspectlens-evals",
    run_name="inspectlens-run-1",
)

evaluator = ForgeEvaluator(
    model=Model.NOVA_LITE_2,
    infra=eval_infra,
    config=ForgeConfig(
        output_s3_path="s3://bucket/output",
        mlflow_monitor=mlflow_monitor,
    ),
)

result = evaluator.evaluate(
    job_name="inspectlens-mlflow-eval",
    eval_task=EvaluationTask.INSPECT_LENS,
    inspect_lens_config=InspectLensConfig(
        benchmarks_path=benchmarks_s3_uri,
        tasks=[{"name": "boolq_pt", "limit": 100}],
    ),
)
```

## Evaluating a Fine-Tuned Model

You can pass a `TrainingResult` from a previous training job to evaluate the trained checkpoint:

```python
# After training
training_result = trainer.train(job_name="my-sft-job")

# Evaluate the trained model via Bedrock or endpoint
eval_result = evaluator.evaluate(
    job_name="post-training-eval",
    eval_task=EvaluationTask.INSPECT_LENS,
    inspect_lens_config=config,
    job_result=training_result,
)
```

If the checkpoint is an S3 URI and Bedrock mode is selected, the SDK falls back to the base model and logs a warning. To evaluate an S3 checkpoint, use the endpoint modes (`endpoint_name` or `model_s3_uri` + `inference_image_uri`).

## Job Caching

When `enable_job_caching=True` in `ForgeConfig`, the SDK caches completed InspectLens results. Subsequent calls with identical parameters return the cached result without submitting a new job.

Cache keys include: `job_name`, `benchmarks_path`, `tasks`, inference scenario, endpoint name, Bedrock model ID, and overrides.

## Summary

- `InspectLensConfig` controls benchmarks, inference provider, and output location
- `benchmarks_path` must be an S3 URI — use `evaluator.upload_benchmarks()` to upload first
- Pass `eval_task=EvaluationTask.INSPECT_LENS` to `ForgeEvaluator.evaluate()`
- Use `dry_run=True` to validate without submitting
- Decoding params go in `overrides`, not `InspectLensConfig`
- Pass `MLflowMonitor` via `ForgeConfig` for experiment tracking
