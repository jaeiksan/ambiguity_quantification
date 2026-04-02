import json
from pathlib import Path
from typing import Any, Dict, List

from ..cache import SQLiteCache
from ..llm_client import BaseLLMClient, GeneratedText
from ..text_utils import stable_hash


def load_prompt(prompt_name: str) -> str:
    prompt_path = Path(__file__).resolve().parent.parent / "prompts" / prompt_name
    return prompt_path.read_text(encoding="utf-8")


def cached_generate(
    *,
    cache: SQLiteCache,
    client: BaseLLMClient,
    measure: str,
    query: str,
    model_id: str,
    prompt_version: str,
    prompt: str,
    n: int,
    temperature: float,
    max_tokens: int,
    logprobs: bool,
) -> List[GeneratedText]:
    cache_key = stable_hash(
        json.dumps(
            {
                "measure": measure,
                "query_hash": stable_hash(query),
                "model_id": model_id,
                "prompt_version": prompt_version,
                "n": n,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "logprobs": logprobs,
                "prompt": prompt,
            },
            sort_keys=True,
        )
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return [GeneratedText(**item) for item in cached]
    outputs = client.generate(
        prompt,
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
        logprobs=logprobs,
    )
    cache.set(cache_key, [item.__dict__ for item in outputs])
    return outputs


def cached_value(
    *,
    cache: SQLiteCache,
    measure: str,
    query: str,
    model_id: str,
    prompt_version: str,
    params: Dict[str, Any],
) -> Any:
    cache_key = stable_hash(
        json.dumps(
            {
                "measure": measure,
                "query_hash": stable_hash(query),
                "model_id": model_id,
                "prompt_version": prompt_version,
                "params": params,
            },
            sort_keys=True,
        )
    )
    return cache_key, cache.get(cache_key)

