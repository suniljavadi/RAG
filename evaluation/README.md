# Evaluation Guide

The public golden fixture is `golden-dataset-v1.jsonl`. It contains synthetic grounded, missing-evidence, injection, access, conflict, and provider-failure cases.

## Metrics

Track evidence recall, retrieval precision, grounded claims, citation correctness, stale-index rate, unauthorized retrieval, latency, and cost.

## Observability

Record request ID, index and document versions, chunk IDs and scores, filter outcomes, model version, latency, and safe quality outcome. Do not log confidential document text by default.

## Release Checks

Evaluate retrieval separately from generation and require a safe insufficient-evidence behavior before publishing a new index or prompt.
