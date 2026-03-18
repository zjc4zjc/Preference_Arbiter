# Persona Benchmark Result Format

## Output Layout

Each run writes to:

```text
Code/results/<benchmark>/<split>_<context_size>/<timestamp>/
```

Examples:

```text
Code/results/personamem/benchmark_32k/20260313_153000/
Code/results/personamem_v2/benchmark_128k/20260313_153500/
```

Each run directory contains:

- `config.json`
- `records.jsonl`
- `summary.json`

## `config.json`

Stores the resolved run configuration:

- benchmark name
- context size
- split
- sample range (`offset`, `limit`)
- retrieval and rerank parameters (`pre_top_k`, `top_k`)
- model paths
- device settings
- ranker limits
- whether chat was enabled
- whether full turns were saved
- run output directory

## `records.jsonl`

One JSON object per sample.

Common fields:

- `status`: `ok`, `error`, or `dry_run`
- `benchmark`
- `sample_id`
- `question`
- `ground_truth`
- `session_count`
- `metadata`

When `status=ok`, additional fields are written:

- `predicted_answer`
- `gold_session_indices`
- `gold_hit_in_embedding`
- `gold_hit_in_rerank`
- `embedding_candidates`
- `reranked_sessions`

### `embedding_candidates`

List of the embedding-stage retrieved sessions. Each item contains:

- `rank`
- `session_index`
- `date`
- `similarity_score`
- `num_turns`
- `preview`
- `turns`
  Only present when `--save-full-turns` is enabled.

### `reranked_sessions`

List of the final ranker output sessions. Each item contains:

- `rank`
- `session_index`
- `date`
- `num_turns`
- `preview`
- `turns`
  Only present when `--save-full-turns` is enabled.

### `metadata`

Benchmark-specific row metadata is preserved to support diagnosis.

For `personamem`, typical fields include:

- `persona_id`
- `question_id`
- `question_type`
- `topic`
- `context_length_in_tokens`
- `distance_to_ref_in_tokens`
- `shared_context_id`

For `personamem_v2`, typical fields include:

- `persona_id`
- `topic_query`
- `preference`
- `topic_preference`
- `conversation_scenario`
- `pref_type`
- `who`
- `updated`
- `prev_pref`
- `sensitive_info`
- `chat_history_path`
- `related_conversation_snippet`

## `summary.json`

Run-level summary with:

- `total_records`
- `success_records`
- `failed_records`
- `dry_run_records`
- `chat_records`
- `gold_session_match_available_records`
- `embedding_gold_hit_count`
- `rerank_gold_hit_count`
- `embedding_gold_hit_rate`
- `rerank_gold_hit_rate`
- `failed_sample_ids`

## Notes

- `gold_session_indices` are only available when the script can align an annotated conversation snippet to one or more segmented sessions.
- For `personamem_v2`, this alignment uses `related_conversation_snippet`.
- A single gold snippet can map to multiple sessions when the annotated snippet spans multiple user-assistant exchanges.
- For `personamem`, no gold session alignment is currently provided in the output.
