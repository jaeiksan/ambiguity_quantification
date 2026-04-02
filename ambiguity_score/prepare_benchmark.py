import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

from .io_utils import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ambiguity benchmarks into {id, query} JSONL.")
    parser.add_argument("--dataset", required=True, choices=["ambigqa", "asqa", "situatedqa", "jsonl"])
    parser.add_argument("--input", help="Path to raw benchmark file.")
    parser.add_argument("--output", required=True, help="Path to output JSONL with {id, query}.")
    parser.add_argument(
        "--source",
        choices=["local", "hf"],
        default="local",
        help="Load benchmark from local file or Hugging Face datasets.",
    )
    parser.add_argument("--hf-dataset", help="Hugging Face dataset id. Example: sewon/ambig_qa")
    parser.add_argument("--hf-config", help="Hugging Face dataset config/subset. Example: light")
    parser.add_argument("--hf-split", default="validation", help="Hugging Face split name. Default: validation")
    parser.add_argument(
        "--query-field",
        default=None,
        help="Override query field for json/jsonl inputs. Examples: question, ambiguous_question, edited_question",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()

    if args.source == "hf":
        rows = load_from_huggingface(
            dataset=args.dataset,
            hf_dataset=args.hf_dataset,
            hf_config=args.hf_config,
            hf_split=args.hf_split,
            query_field=args.query_field,
        )
    else:
        if not args.input:
            raise ValueError("--input is required when --source local is used.")
        input_path = Path(args.input).expanduser().resolve()
        if args.dataset == "ambigqa":
            rows = convert_ambigqa(load_json(input_path))
        elif args.dataset == "asqa":
            rows = convert_asqa(load_json(input_path))
        elif args.dataset == "situatedqa":
            rows = convert_situatedqa(load_json(input_path), query_field=args.query_field or "question")
        else:
            rows = convert_jsonl(input_path, query_field=args.query_field or "query")

    write_jsonl(str(output_path), rows)


def load_from_huggingface(
    *,
    dataset: str,
    hf_dataset: str | None,
    hf_config: str | None,
    hf_split: str,
    query_field: str | None,
) -> List[Dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face loading requires the 'datasets' package. Install it first."
        ) from exc

    dataset_id = hf_dataset or default_hf_dataset_id(dataset)
    records = load_dataset(dataset_id, hf_config, split=hf_split)
    payload = [dict(item) for item in records]
    if dataset == "ambigqa":
        return convert_ambigqa(payload)
    if dataset == "asqa":
        return convert_asqa(payload)
    if dataset == "situatedqa":
        return convert_situatedqa(payload, query_field=query_field or "question")
    return convert_hf_records(payload, query_field=query_field or "query")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def convert_ambigqa(payload) -> List[Dict]:
    rows: List[Dict] = []
    for index, item in enumerate(_iter_records(payload)):
        gold_disambiguated_queries = extract_ambigqa_disambiguations(item)
        rows.append(
            {
                "id": item.get("id", f"ambigqa-{index}"),
                "query": item["question"],
                "gold_disambiguated_queries": gold_disambiguated_queries,
                "is_ambiguous": bool(gold_disambiguated_queries),
            }
        )
    return rows


def convert_asqa(payload) -> List[Dict]:
    rows: List[Dict] = []
    for index, item in enumerate(_iter_records(payload)):
        query = item.get("ambiguous_question") or item.get("question")
        if query is None:
            continue
        rows.append({"id": item.get("sample_id", item.get("id", f"asqa-{index}")), "query": query})
    return rows


def convert_situatedqa(payload, *, query_field: str) -> List[Dict]:
    rows: List[Dict] = []
    for index, item in enumerate(_iter_records(payload)):
        query = item.get(query_field)
        if query is None:
            continue
        rows.append({"id": item.get("id", f"situatedqa-{index}"), "query": query})
    return rows


def convert_jsonl(path: Path, *, query_field: str) -> List[Dict]:
    rows: List[Dict] = []
    for index, item in enumerate(load_jsonl(path)):
        query = item.get(query_field)
        if query is None:
            continue
        rows.append({"id": item.get("id", f"jsonl-{index}"), "query": query})
    return rows


def convert_hf_records(records: List[Dict], *, query_field: str) -> List[Dict]:
    rows: List[Dict] = []
    for index, item in enumerate(records):
        query = item.get(query_field)
        if query is None:
            continue
        rows.append({"id": item.get("id", f"hf-{index}"), "query": query})
    return rows


def default_hf_dataset_id(dataset: str) -> str:
    defaults = {
        "ambigqa": "sewon/ambig_qa",
        "asqa": "din0s/asqa",
        "situatedqa": "siyue/SituatedQA",
    }
    if dataset not in defaults:
        raise ValueError(f"No default HF dataset id registered for dataset={dataset}")
    return defaults[dataset]


def extract_ambigqa_disambiguations(item: Dict) -> List[str]:
    annotations = item.get("annotations") or []
    collected: List[str] = []

    if isinstance(annotations, dict):
        types = annotations.get("type") or []
        qa_pairs_by_annotation = annotations.get("qaPairs") or []
        for annotation_type, qa_pairs in zip(types, qa_pairs_by_annotation):
            if annotation_type != "multipleQAs":
                continue
            if isinstance(qa_pairs, list):
                for qa_pair in qa_pairs:
                    questions = _extract_questions_from_qa_pair(qa_pair)
                    collected.extend(questions)
            elif isinstance(qa_pairs, dict):
                questions = _extract_questions_from_qa_pair(qa_pairs)
                collected.extend(questions)
    elif isinstance(annotations, list):
        for annotation in annotations:
            if annotation.get("type") != "multipleQAs":
                continue
            for qa_pair in annotation.get("qaPairs", []):
                questions = _extract_questions_from_qa_pair(qa_pair)
                collected.extend(questions)

    deduped: List[str] = []
    seen = set()
    for question in collected:
        if question not in seen:
            seen.add(question)
            deduped.append(question)
    return deduped


def _extract_questions_from_qa_pair(qa_pair: Dict) -> List[str]:
    question = qa_pair.get("question")
    if isinstance(question, str):
        return [question]
    if isinstance(question, list):
        return [item for item in question if isinstance(item, str) and item.strip()]
    return []


def _iter_records(payload) -> Iterable[Dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "examples", "train", "dev", "validation", "test"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError("Unsupported benchmark payload shape.")


if __name__ == "__main__":
    main()
