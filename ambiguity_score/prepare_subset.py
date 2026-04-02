import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from .io_utils import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare filtered benchmark subsets.")
    parser.add_argument("--input", required=True, help="Input JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--limit", type=int, default=None, help="Number of rows to keep. Keep all rows if omitted.")
    parser.add_argument(
        "--require-gold-disambiguation",
        action="store_true",
        help="Keep only rows with non-empty gold_disambiguated_queries.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle filtered rows before truncation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used with --shuffle.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.input).expanduser().resolve())
    if args.require_gold_disambiguation:
        rows = [row for row in rows if row.get("gold_disambiguated_queries")]

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(rows)

    selected = rows[: args.limit] if args.limit is not None else rows
    write_jsonl(str(Path(args.output).expanduser().resolve()), selected)


def read_jsonl(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    main()
