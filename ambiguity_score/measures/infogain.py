from statistics import mean
from typing import Dict, List, Optional

from ..cache import SQLiteCache
from ..llm_client import BaseLLMClient, LLMError, UnsupportedFeatureError
from .common import cached_generate, cached_value, load_prompt


PROMPT_NAME = "disambiguate_v1.txt"
PROMPT_VERSION = "disambiguate_v1"
PUNCS = [".", ",", "!", "?", "~", "-", "_"]


def compute_infogain(
    *,
    client: BaseLLMClient,
    cache: SQLiteCache,
    query: str,
    model_id: str,
    disambiguated_queries: Optional[List[str]] = None,
    disambiguation_source: str = "generated",
) -> Dict[str, Optional[float]]:
    cache_key, cached = cached_value(
        cache=cache,
        measure="infogain",
        query=query,
        model_id=model_id,
        prompt_version=PROMPT_VERSION,
        params={
            "disambiguated_queries": disambiguated_queries or [],
            "disambiguation_source": disambiguation_source,
        },
    )
    if cached is not None:
        return cached

    if disambiguated_queries:
        candidates = [item.strip() for item in disambiguated_queries if item and item.strip()]
    else:
        prompt = load_prompt(PROMPT_NAME).format(query=query)
        candidates = [
            cached_generate(
                cache=cache,
                client=client,
                measure="disambiguation_generation",
                query=query,
                model_id=model_id,
                prompt_version=PROMPT_VERSION,
                prompt=prompt,
                n=1,
                temperature=0.0,
                max_tokens=128,
                logprobs=False,
            )[0].text
        ]

    try:
        question_entropy = client.score_text_entropy(clean_string_for_measure(query))
        disambiguation_entropies = [
            client.score_text_entropy(clean_string_for_measure(candidate))
            for candidate in candidates
        ]
    except (LLMError, UnsupportedFeatureError):
        question_entropy = None
        disambiguation_entropies = []

    filtered_entropies = [value for value in disambiguation_entropies if value is not None]
    disambiguation_entropy = mean(filtered_entropies) if filtered_entropies else None
    infogain = None
    if question_entropy is not None and disambiguation_entropy is not None:
        infogain = question_entropy - disambiguation_entropy

    result = {
        "disambiguated_query": candidates[0] if candidates else None,
        "disambiguated_queries": candidates,
        "question_entropy": question_entropy,
        "disambiguated_entropy": disambiguation_entropy,
        "infogain": infogain,
        "disambiguation_source": disambiguation_source if disambiguated_queries else "generated",
    }
    cache.set(cache_key, result)
    return result


def clean_string_for_measure(text: str) -> str:
    cleaned = text.strip().lower()
    for punc in PUNCS:
        cleaned = cleaned.replace(punc, "")
    return cleaned
