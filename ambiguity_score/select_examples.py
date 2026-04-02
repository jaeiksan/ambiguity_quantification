import argparse
import json
from pathlib import Path
from typing import Dict, List

from .io_utils import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select high/low ambiguity examples from scored JSONL.")
    parser.add_argument("--input", required=True, help="Scored JSONL input path.")
    parser.add_argument("--high-output", required=True, help="Output path for top high-ambiguity rows.")
    parser.add_argument("--low-output", required=True, help="Output path for top low-ambiguity rows.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of examples for each side.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.input).expanduser().resolve())
    high_rows = sorted(rows, key=high_rank_key, reverse=True)[: args.top_n]
    low_rows = sorted(rows, key=low_rank_key)[: args.top_n]
    write_jsonl(str(Path(args.high_output).expanduser().resolve()), [summarize(row) for row in high_rows])
    write_jsonl(str(Path(args.low_output).expanduser().resolve()), [summarize(row) for row in low_rows])


def read_jsonl(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def high_rank_key(row: Dict) -> tuple:
    measures = row.get("measures", {})
    return (
        measures.get("semantic_entropy") or 0.0,
        measures.get("sample_rep_uncertainty") or 0.0,
        measures.get("mean_token_entropy") or 0.0,
    )


def low_rank_key(row: Dict) -> tuple:
    measures = row.get("measures", {})
    return (
        measures.get("semantic_entropy") or 0.0,
        measures.get("sample_rep_uncertainty") or 0.0,
        measures.get("mean_token_entropy") or 0.0,
    )


def summarize(row: Dict) -> Dict:
    measures = row.get("measures", {})
    meta = row.get("meta", {})
    return {
        "id": row.get("id"),
        "query": row.get("query"),
        "disambiguated_query": row.get("disambiguated_query"),
        "gold_disambiguated_queries": row.get("gold_disambiguated_queries"),
        "measures": {
            "semantic_entropy": measures.get("semantic_entropy"),
            "sample_rep_uncertainty": measures.get("sample_rep_uncertainty"),
            "mean_token_entropy": measures.get("mean_token_entropy"),
            "infogain": measures.get("infogain"),
        },
        "meta": {
            "infogain_disambiguation_source": meta.get("infogain_disambiguation_source"),
            "semantic_clusters": meta.get("semantic_clusters"),
            "sample_rep_samples": meta.get("sample_rep_samples"),
        },
    }


if __name__ == "__main__":
    main()
