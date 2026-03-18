import argparse
import ast
import csv
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
MEMSIFTER_ROOT = REPO_ROOT / "Baseline" / "MemSifter"
if str(MEMSIFTER_ROOT) not in sys.path:
    sys.path.insert(0, str(MEMSIFTER_ROOT))

from memsifter.toolkit import LLMChat, SessionEmbedder, SessionRanker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MemSifter on PersonaMem or PersonaMem-v2 and save detailed results."
    )
    parser.add_argument(
        "--benchmark",
        choices=["personamem", "personamem_v2"],
        required=True,
    )
    parser.add_argument(
        "--context-size",
        choices=["32k", "128k", "1M"],
        required=True,
        help="PersonaMem supports 32k/128k/1M; PersonaMem-v2 text supports 32k/128k.",
    )
    parser.add_argument(
        "--split",
        default="benchmark",
        choices=["benchmark", "train", "val"],
        help="Only used by PersonaMem-v2.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--pre-top-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--drop-system-session",
        action="store_true",
        help="Drop the leading system persona/context message instead of keeping it as session 0.",
    )
    parser.add_argument(
        "--embed-model-path",
        default=str(MEMSIFTER_ROOT / "models" / "bge-m3"),
    )
    parser.add_argument(
        "--ranker-model-path",
        default=str(MEMSIFTER_ROOT / "models" / "zstanjj" / "MemSifter-4B-Thinking"),
    )
    parser.add_argument("--embed-device", default="cuda:0")
    parser.add_argument("--ranker-device", default="cuda:1")
    parser.add_argument("--ranker-max-model-len", type=int, default=32768)
    parser.add_argument("--ranker-max-output-tokens", type=int, default=256)
    parser.add_argument("--ranker-temperature", type=float, default=0.6)
    parser.add_argument("--ranker-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--skip-chat", action="store_true")
    parser.add_argument("--api-key", default=os.environ.get("API_KEY"))
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL"))
    parser.add_argument(
        "--chat-model",
        default=os.environ.get("CHAT_MODEL") or os.environ.get("MODEL_NAME"),
    )
    parser.add_argument(
        "--chat-prompt-config-path",
        default=str(MEMSIFTER_ROOT / "configs" / "chat_default_prompt.yaml"),
    )
    parser.add_argument("--chat-max-output-tokens", type=int, default=1024)
    parser.add_argument("--chat-temperature", type=float, default=0.6)
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "Code" / "results"),
    )
    parser.add_argument(
        "--save-full-turns",
        action="store_true",
        help="Store full turns for retrieved sessions in the result jsonl.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only parse dataset rows and write result skeletons without loading models.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if a sample fails instead of continuing.",
    )
    return parser.parse_args()


def ensure_supported_args(args: argparse.Namespace) -> None:
    if args.benchmark == "personamem_v2" and args.context_size not in {"32k", "128k"}:
        raise ValueError("PersonaMem-v2 only supports --context-size 32k or 128k.")
    if args.benchmark == "personamem" and args.split != "benchmark":
        raise ValueError("PersonaMem uses its own CSV files and does not support --split.")


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def normalize_content_for_match(text: str) -> str:
    return normalize_whitespace(text).lower()


def strip_role_prefix(role: str, content: str) -> str:
    prefixes = {
        "user": ["User:"],
        "assistant": ["Assistant:"],
        "system": ["System:"],
    }
    stripped = content.strip()
    for prefix in prefixes.get(role, []):
        if stripped.startswith(prefix):
            return stripped[len(prefix):].strip()
    return stripped


def normalize_message(message: Dict[str, Any]) -> Dict[str, str]:
    role = str(message["role"]).strip().lower()
    content = strip_role_prefix(role, str(message["content"]))
    return {
        "role": role,
        "content": content,
    }


def chunk_messages_into_sessions(
    messages: Sequence[Dict[str, Any]],
    keep_system_session: bool = True,
) -> List[List[Dict[str, str]]]:
    if not messages:
        return []

    normalized = [normalize_message(m) for m in messages]
    sessions: List[List[Dict[str, str]]] = []
    start_idx = 0

    if normalized[0]["role"] == "system":
        if keep_system_session:
            sessions.append([normalized[0]])
        start_idx = 1

    current: List[Dict[str, str]] = []
    for message in normalized[start_idx:]:
        current.append(message)
        if message["role"] == "assistant":
            sessions.append(current)
            current = []
    if current:
        sessions.append(current)

    return sessions


def session_preview(session_turns: Sequence[Dict[str, str]], max_chars: int = 240) -> str:
    preview = " ".join(
        f"{turn['role']}: {normalize_whitespace(turn['content'])}"
        for turn in session_turns
    )
    if len(preview) <= max_chars:
        return preview
    return preview[: max_chars - 3] + "..."


def safe_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_bool(value: Any) -> Optional[bool]:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def maybe_parse_literal(value: Any) -> Any:
    if value in (None, ""):
        return value
    if not isinstance(value, str):
        return value
    parsers = (json.loads, ast.literal_eval)
    for parser in parsers:
        try:
            return parser(value)
        except Exception:
            continue
    return value


def rows_with_slice(rows: List[Dict[str, str]], offset: int, limit: Optional[int]) -> List[Dict[str, str]]:
    if offset < 0:
        raise ValueError("--offset must be >= 0")
    if limit is None:
        return rows[offset:]
    if limit < 0:
        raise ValueError("--limit must be >= 0")
    return rows[offset: offset + limit]


def load_personamem_examples(
    context_size: str,
    offset: int,
    limit: Optional[int],
    keep_system_session: bool,
) -> List[Dict[str, Any]]:
    dataset_root = REPO_ROOT / "Dataset" / "PersonaMem"
    questions_path = dataset_root / f"questions_{context_size}.csv"
    contexts_path = dataset_root / f"shared_contexts_{context_size}.jsonl"

    with questions_path.open(newline="", encoding="utf-8") as f:
        rows = rows_with_slice(list(csv.DictReader(f)), offset, limit)

    wanted_context_ids = {row["shared_context_id"] for row in rows}
    context_map: Dict[str, List[Dict[str, Any]]] = {}
    with contexts_path.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            context_id, context_messages = next(iter(item.items()))
            if context_id in wanted_context_ids:
                context_map[context_id] = context_messages
                if len(context_map) == len(wanted_context_ids):
                    break

    examples: List[Dict[str, Any]] = []
    for row in rows:
        context_id = row["shared_context_id"]
        if context_id not in context_map:
            raise KeyError(f"Missing shared context {context_id} in {contexts_path}")
        end_index = safe_int(row["end_index_in_shared_context"])
        context_messages = context_map[context_id]
        history_messages = context_messages[:end_index] if end_index is not None else context_messages
        sessions = chunk_messages_into_sessions(
            history_messages,
            keep_system_session=keep_system_session,
        )
        metadata = {
            "persona_id": safe_int(row["persona_id"]),
            "question_id": row["question_id"],
            "question_type": row["question_type"],
            "topic": row["topic"],
            "context_length_in_tokens": safe_int(row["context_length_in_tokens"]),
            "distance_to_ref_in_blocks": safe_int(row["distance_to_ref_in_blocks"]),
            "distance_to_ref_in_tokens": safe_int(row["distance_to_ref_in_tokens"]),
            "num_irrelevant_tokens": safe_int(row["num_irrelevant_tokens"]),
            "distance_to_ref_proportion_in_context": row["distance_to_ref_proportion_in_context"],
            "shared_context_id": context_id,
            "end_index_in_shared_context": end_index,
            "all_options": maybe_parse_literal(row["all_options"]),
        }
        examples.append(
            {
                "benchmark": "personamem",
                "sample_id": row["question_id"],
                "question": row["user_question_or_message"],
                "ground_truth": row["correct_answer"],
                "sessions": sessions,
                "metadata": metadata,
                "gold_session_indices": [],
            }
        )

    return examples


def parse_personamem_v2_query(raw_query: str) -> Dict[str, str]:
    parsed = maybe_parse_literal(raw_query)
    if not isinstance(parsed, dict):
        raise ValueError(f"Unable to parse user_query: {raw_query[:200]}")
    if "role" not in parsed or "content" not in parsed:
        raise ValueError("Parsed user_query must contain role/content.")
    return {
        "role": str(parsed["role"]).lower(),
        "content": str(parsed["content"]),
    }


def find_gold_session_indices(
    sessions: Sequence[Sequence[Dict[str, str]]],
    related_snippet: Any,
) -> List[int]:
    snippet = maybe_parse_literal(related_snippet)
    if not isinstance(snippet, list) or not snippet:
        return []

    normalized_snippet_turns = [
        normalize_content_for_match(str(turn["content"]))
        for turn in snippet
        if isinstance(turn, dict) and "content" in turn
    ]
    if not normalized_snippet_turns:
        return []

    gold_indices: set[int] = set()
    normalized_sessions = [
        "\n".join(
            normalize_content_for_match(turn["content"])
            for turn in session
        )
        for session in sessions
    ]
    for snippet_turn in normalized_snippet_turns:
        for session_idx, normalized_session in enumerate(normalized_sessions):
            if snippet_turn in normalized_session:
                gold_indices.add(session_idx)
    return sorted(gold_indices)


def load_personamem_v2_examples(
    context_size: str,
    split: str,
    offset: int,
    limit: Optional[int],
    keep_system_session: bool,
) -> List[Dict[str, Any]]:
    dataset_root = REPO_ROOT / "Dataset" / "PersonaMem-v2"
    benchmark_path = dataset_root / "benchmark" / "text" / f"{split}.csv"

    with benchmark_path.open(newline="", encoding="utf-8") as f:
        rows = rows_with_slice(list(csv.DictReader(f)), offset, limit)

    examples: List[Dict[str, Any]] = []
    for row in rows:
        history_link_key = f"chat_history_{context_size}_link"
        history_rel_path = row.get(history_link_key)
        if not history_rel_path:
            raise KeyError(f"Missing {history_link_key} for row persona_id={row.get('persona_id')}")
        history_path = dataset_root / history_rel_path
        history_obj = json.loads(history_path.read_text(encoding="utf-8"))
        history_messages = history_obj["chat_history"]
        sessions = chunk_messages_into_sessions(
            history_messages,
            keep_system_session=keep_system_session,
        )
        query_message = parse_personamem_v2_query(row["user_query"])
        gold_session_indices = find_gold_session_indices(
            sessions=sessions,
            related_snippet=row.get("related_conversation_snippet"),
        )

        metadata = {
            "persona_id": safe_int(row["persona_id"]),
            "topic_query": row["topic_query"],
            "preference": row["preference"],
            "topic_preference": row["topic_preference"],
            "conversation_scenario": row["conversation_scenario"],
            "pref_type": row["pref_type"],
            "who": row["who"],
            "updated": safe_bool(row["updated"]),
            "prev_pref": row["prev_pref"],
            "sensitive_info": row["sensitive_info"],
            "chat_history_path": history_rel_path,
            "raw_persona_file": row["raw_persona_file"],
            "total_tokens_in_chat_history_32k": safe_int(row["total_tokens_in_chat_history_32k"]),
            "total_tokens_in_chat_history_128k": safe_int(row["total_tokens_in_chat_history_128k"]),
            "distance_from_related_snippet_to_query_32k": safe_int(
                row["distance_from_related_snippet_to_query_32k"]
            ),
            "distance_from_related_snippet_to_query_128k": safe_int(
                row["distance_from_related_snippet_to_query_128k"]
            ),
            "num_persona_relevant_tokens_32k": safe_int(row["num_persona_relevant_tokens_32k"]),
            "num_persona_irrelevant_tokens_32k": safe_int(row["num_persona_irrelevant_tokens_32k"]),
            "num_persona_relevant_tokens_128k": safe_int(row["num_persona_relevant_tokens_128k"]),
            "num_persona_irrelevant_tokens_128k": safe_int(row["num_persona_irrelevant_tokens_128k"]),
            "related_conversation_snippet": maybe_parse_literal(row["related_conversation_snippet"]),
        }

        examples.append(
            {
                "benchmark": "personamem_v2",
                "sample_id": f"{row['persona_id']}_{offset + len(examples)}",
                "question": query_message["content"],
                "ground_truth": row["correct_answer"],
                "sessions": sessions,
                "metadata": metadata,
                "gold_session_indices": gold_session_indices,
            }
        )

    return examples


def load_examples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    keep_system_session = not args.drop_system_session
    if args.benchmark == "personamem":
        return load_personamem_examples(
            context_size=args.context_size,
            offset=args.offset,
            limit=args.limit,
            keep_system_session=keep_system_session,
        )
    return load_personamem_v2_examples(
        context_size=args.context_size,
        split=args.split,
        offset=args.offset,
        limit=args.limit,
        keep_system_session=keep_system_session,
    )


def serialize_embedding_candidates(
    top_sessions: Sequence[Tuple[int, List[Dict[str, str]], Optional[str], float]],
    save_full_turns: bool,
) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for rank, (session_index, turns, date, score) in enumerate(top_sessions, start=1):
        item = {
            "rank": rank,
            "session_index": session_index,
            "date": date,
            "similarity_score": score,
            "num_turns": len(turns),
            "preview": session_preview(turns),
        }
        if save_full_turns:
            item["turns"] = turns
        serialized.append(item)
    return serialized


def serialize_reranked_sessions(
    ranked_sessions: Sequence[Tuple[int, List[Dict[str, str]], Optional[str]]],
    save_full_turns: bool,
) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for rank, (session_index, turns, date) in enumerate(ranked_sessions, start=1):
        item = {
            "rank": rank,
            "session_index": session_index,
            "date": date,
            "num_turns": len(turns),
            "preview": session_preview(turns),
        }
        if save_full_turns:
            item["turns"] = turns
        serialized.append(item)
    return serialized


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_run_dir(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root)
    benchmark_part = args.benchmark
    split_part = args.split if args.benchmark == "personamem_v2" else "benchmark"
    run_dir = output_root / benchmark_part / f"{split_part}_{args.context_size}" / now_stamp()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_config_payload(args: argparse.Namespace, example_count: int, chat_enabled: bool) -> Dict[str, Any]:
    return {
        "benchmark": args.benchmark,
        "context_size": args.context_size,
        "split": args.split,
        "offset": args.offset,
        "limit": args.limit,
        "loaded_examples": example_count,
        "pre_top_k": args.pre_top_k,
        "top_k": args.top_k,
        "drop_system_session": args.drop_system_session,
        "embed_model_path": str(Path(args.embed_model_path).resolve()),
        "ranker_model_path": str(Path(args.ranker_model_path).resolve()),
        "embed_device": args.embed_device,
        "ranker_device": args.ranker_device,
        "ranker_max_model_len": args.ranker_max_model_len,
        "ranker_max_output_tokens": args.ranker_max_output_tokens,
        "ranker_temperature": args.ranker_temperature,
        "ranker_gpu_memory_utilization": args.ranker_gpu_memory_utilization,
        "chat_enabled": chat_enabled,
        "chat_model": args.chat_model if chat_enabled else None,
        "chat_prompt_config_path": str(Path(args.chat_prompt_config_path).resolve()),
        "chat_max_output_tokens": args.chat_max_output_tokens,
        "chat_temperature": args.chat_temperature,
        "save_full_turns": args.save_full_turns,
        "dry_run": args.dry_run,
    }


def build_summary(
    config: Dict[str, Any],
    run_seconds: float,
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    success_records = [r for r in records if r["status"] == "ok"]
    failed_records = [r for r in records if r["status"] == "error"]
    dry_run_records = [r for r in records if r["status"] == "dry_run"]
    gold_available_records = [
        r for r in success_records
        if r.get("gold_session_indices")
    ]
    embed_gold_hits = sum(1 for r in gold_available_records if r.get("gold_hit_in_embedding"))
    rerank_gold_hits = sum(1 for r in gold_available_records if r.get("gold_hit_in_rerank"))
    chat_records = [r for r in success_records if r.get("predicted_answer") is not None]
    return {
        "config": config,
        "run_seconds": run_seconds,
        "total_records": len(records),
        "success_records": len(success_records),
        "failed_records": len(failed_records),
        "dry_run_records": len(dry_run_records),
        "chat_records": len(chat_records),
        "gold_session_match_available_records": len(gold_available_records),
        "embedding_gold_hit_count": embed_gold_hits,
        "rerank_gold_hit_count": rerank_gold_hits,
        "embedding_gold_hit_rate": (
            embed_gold_hits / len(gold_available_records)
            if gold_available_records else None
        ),
        "rerank_gold_hit_rate": (
            rerank_gold_hits / len(gold_available_records)
            if gold_available_records else None
        ),
        "failed_sample_ids": [r["sample_id"] for r in failed_records],
        "dry_run_sample_ids": [r["sample_id"] for r in dry_run_records],
    }


def run() -> None:
    args = parse_args()
    ensure_supported_args(args)

    examples = load_examples(args)
    chat_enabled = (
        not args.skip_chat
        and bool(args.api_key)
        and bool(args.base_url)
        and bool(args.chat_model)
    )
    run_dir = build_run_dir(args)
    config = build_config_payload(args, example_count=len(examples), chat_enabled=chat_enabled)
    config["run_dir"] = str(run_dir)
    config_path = run_dir / "config.json"
    records_path = run_dir / "records.jsonl"
    summary_path = run_dir / "summary.json"
    write_json(config_path, config)

    if args.dry_run:
        dry_records: List[Dict[str, Any]] = []
        for example in examples:
            row = {
                "status": "dry_run",
                "benchmark": example["benchmark"],
                "sample_id": example["sample_id"],
                "question": example["question"],
                "ground_truth": example["ground_truth"],
                "session_count": len(example["sessions"]),
                "gold_session_indices": example["gold_session_indices"],
                "metadata": example["metadata"],
            }
            append_jsonl(records_path, row)
            dry_records.append(row)
        summary = build_summary(config=config, run_seconds=0.0, records=dry_records)
        write_json(summary_path, summary)
        print(f"Dry run complete. Results written to: {run_dir}")
        return

    embedder = SessionEmbedder(
        model_path=str(Path(args.embed_model_path).resolve()),
        device=args.embed_device,
    )
    ranker = SessionRanker(
        model_path=str(Path(args.ranker_model_path).resolve()),
        device=args.ranker_device,
        max_model_len=args.ranker_max_model_len,
        max_output_tokens=args.ranker_max_output_tokens,
        temperature=args.ranker_temperature,
        gpu_memory_utilization=args.ranker_gpu_memory_utilization,
    )
    chat = None
    if chat_enabled:
        chat = LLMChat(
            api_key=args.api_key,
            base_url=args.base_url,
            model_name=args.chat_model,
            max_output_tokens=args.chat_max_output_tokens,
            temperature=args.chat_temperature,
            prompt_config_path=str(Path(args.chat_prompt_config_path).resolve()),
        )

    run_start = time.time()
    records: List[Dict[str, Any]] = []
    total = len(examples)
    for sample_no, example in enumerate(examples, start=1):
        print(f"[{sample_no}/{total}] {example['sample_id']}")
        try:
            top_sessions = embedder.get_top_sessions(
                question=example["question"],
                sessions=example["sessions"],
                dates=None,
                top_k=args.pre_top_k,
            )
            ranked_sessions = ranker.rerank(
                question=example["question"],
                pre_ranked_sessions=top_sessions,
                top_k=args.top_k,
            )
            predicted_answer = None
            if chat is not None:
                predicted_answer = chat.answer(
                    question=example["question"],
                    ranked_sessions=ranked_sessions,
                )

            embedding_session_indices = [item[0] for item in top_sessions]
            rerank_session_indices = [item[0] for item in ranked_sessions]
            gold_session_indices = example["gold_session_indices"]
            record = {
                "status": "ok",
                "benchmark": example["benchmark"],
                "sample_id": example["sample_id"],
                "question": example["question"],
                "ground_truth": example["ground_truth"],
                "predicted_answer": predicted_answer,
                "session_count": len(example["sessions"]),
                "gold_session_indices": gold_session_indices,
                "gold_hit_in_embedding": (
                    any(idx in embedding_session_indices for idx in gold_session_indices)
                    if gold_session_indices else None
                ),
                "gold_hit_in_rerank": (
                    any(idx in rerank_session_indices for idx in gold_session_indices)
                    if gold_session_indices else None
                ),
                "embedding_candidates": serialize_embedding_candidates(
                    top_sessions=top_sessions,
                    save_full_turns=args.save_full_turns,
                ),
                "reranked_sessions": serialize_reranked_sessions(
                    ranked_sessions=ranked_sessions,
                    save_full_turns=args.save_full_turns,
                ),
                "metadata": example["metadata"],
            }
        except Exception as exc:
            record = {
                "status": "error",
                "benchmark": example["benchmark"],
                "sample_id": example["sample_id"],
                "question": example["question"],
                "ground_truth": example["ground_truth"],
                "session_count": len(example["sessions"]),
                "metadata": example["metadata"],
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            if args.stop_on_error:
                append_jsonl(records_path, record)
                records.append(record)
                raise

        append_jsonl(records_path, record)
        records.append(record)

    summary = build_summary(
        config=config,
        run_seconds=time.time() - run_start,
        records=records,
    )
    write_json(summary_path, summary)
    print(f"Run complete. Results written to: {run_dir}")


if __name__ == "__main__":
    run()
