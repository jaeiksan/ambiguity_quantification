import argparse
import os
from pathlib import Path
from typing import Dict, List

from .cache import SQLiteCache
from .io_utils import read_jsonl, utc_now_iso, write_jsonl
from .llm_client import build_llm_client
from .measures import (
    compute_infogain,
    compute_mean_token_entropy,
    compute_sample_rep_uncertainty,
    compute_semantic_entropy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute ambiguity scores for query JSONL.")
    parser.add_argument(
        "--env-file",
        default=str(_project_root() / ".env"),
        help="Path to .env file. Defaults to ambiguity_test/.env",
    )
    parser.add_argument("--input", required=True, help="Path to input JSONL with {id, query}.")
    parser.add_argument("--output", required=True, help="Path to output JSONL.")
    parser.add_argument(
        "--measures",
        nargs="+",
        default=["infogain", "sample_rep", "semantic_entropy"],
        choices=["infogain", "sample_rep", "semantic_entropy", "token_entropy"],
    )
    parser.add_argument("--k_samples", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N queries after offset.")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N queries from the input.")
    parser.add_argument("--backend", choices=["openai", "hf", "fake"], default=os.environ.get("AMBIGUITY_SCORE_BACKEND", "hf"))
    parser.add_argument("--cache-path", default=".cache/ambiguity_score.sqlite")
    parser.add_argument("--model-id", default=os.environ.get("MODEL_ID", "Qwen/Qwen2.5-14B-Instruct"))
    parser.add_argument("--embedding-model-id", default=os.environ.get("EMBEDDING_MODEL_ID", "Qwen/Qwen2.5-14B-Instruct"))
    parser.add_argument("--semantic-threshold", type=float, default=float(os.environ.get("SEMANTIC_CLUSTER_THRESHOLD", "0.9")))
    parser.add_argument("--semantic-linkage", choices=["complete", "centroid"], default=os.environ.get("SEMANTIC_CLUSTER_LINKAGE", "complete"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--hf-device", default=os.environ.get("HF_DEVICE", "auto"))
    parser.add_argument("--hf-torch-dtype", default=os.environ.get("HF_TORCH_DTYPE", "bfloat16"))
    parser.add_argument("--hf-max-input-length", type=int, default=int(os.environ.get("HF_MAX_INPUT_LENGTH", "1024")))
    return parser.parse_args()


def main() -> None:
    bootstrap_args, _ = _bootstrap_parser().parse_known_args()
    load_dotenv(bootstrap_args.env_file)
    args = parse_args()
    cache = SQLiteCache(_resolve_path(args.cache_path))
    client = build_llm_client(
        backend=args.backend,
        cache=cache,
        model_id=args.model_id,
        embedding_model_id=args.embedding_model_id,
        base_url=args.base_url,
        api_key=args.api_key,
        hf_device=args.hf_device,
        hf_torch_dtype=args.hf_torch_dtype,
        hf_max_input_length=args.hf_max_input_length,
    )

    rows = read_jsonl(_resolve_path(args.input))
    rows = rows[args.offset :]
    if args.limit is not None:
        rows = rows[: args.limit]
    results: List[Dict] = []
    for row in rows:
        query = row["query"]
        result = {
            "id": row["id"],
            "query": query,
            "disambiguated_query": None,
            "measures": {},
            "meta": {
                "model": args.model_id,
                "backend": args.backend,
                "k_samples": args.k_samples,
                "timestamp": utc_now_iso(),
            },
        }

        if "infogain" in args.measures:
            gold_disambiguations = row.get("gold_disambiguated_queries")
            infogain = compute_infogain(
                client=client,
                cache=cache,
                query=query,
                model_id=args.model_id,
                disambiguated_queries=gold_disambiguations,
                disambiguation_source="gold" if gold_disambiguations else "generated",
            )
            result["disambiguated_query"] = infogain["disambiguated_query"]
            result["gold_disambiguated_queries"] = gold_disambiguations
            result["measures"]["infogain"] = infogain["infogain"]
            result["measures"]["question_entropy"] = infogain["question_entropy"]
            result["measures"]["disambiguated_entropy"] = infogain["disambiguated_entropy"]
            result["meta"]["infogain_disambiguation_source"] = infogain["disambiguation_source"]

        if "sample_rep" in args.measures:
            sample_rep = compute_sample_rep_uncertainty(
                client=client,
                cache=cache,
                query=query,
                model_id=args.model_id,
                k_samples=args.k_samples,
            )
            result["measures"]["sample_rep_uncertainty"] = sample_rep["sample_rep_uncertainty"]
            result["meta"]["sample_rep_samples"] = sample_rep["samples"]

        if "semantic_entropy" in args.measures:
            semantic_entropy = compute_semantic_entropy(
                client=client,
                cache=cache,
                query=query,
                model_id=args.model_id,
                k_samples=args.k_samples,
                embedding_model_id=args.embedding_model_id,
                cluster_threshold=args.semantic_threshold,
                cluster_linkage=args.semantic_linkage,
            )
            result["measures"]["semantic_entropy"] = semantic_entropy["semantic_entropy"]
            result["meta"]["semantic_clusters"] = semantic_entropy["clusters"]
            result["meta"]["semantic_entropy_embedding_model"] = semantic_entropy["embedding_model"]
            result["meta"]["semantic_cluster_threshold"] = semantic_entropy["cluster_threshold"]
            result["meta"]["semantic_cluster_linkage"] = semantic_entropy["cluster_linkage"]

        if "token_entropy" in args.measures:
            token_entropy = compute_mean_token_entropy(
                client=client,
                cache=cache,
                query=query,
                model_id=args.model_id,
            )
            result["measures"]["mean_token_entropy"] = token_entropy["mean_token_entropy"]

        results.append(result)

    write_jsonl(_resolve_path(args.output), results)


def _resolve_path(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def _bootstrap_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--env-file", default=str(_project_root() / ".env"))
    return parser


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_dotenv(path: str) -> None:
    env_path = Path(path).expanduser().resolve()
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


if __name__ == "__main__":
    main()
