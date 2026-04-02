from typing import Dict, List

from ..cache import SQLiteCache
from ..llm_client import BaseLLMClient
from ..text_utils import pairwise_average_similarity
from .common import cached_generate, cached_value


PROMPT_VERSION = "answer_v1"


def compute_sample_rep_uncertainty(
    *,
    client: BaseLLMClient,
    cache: SQLiteCache,
    query: str,
    model_id: str,
    k_samples: int,
) -> Dict:
    cache_key, cached = cached_value(
        cache=cache,
        measure="sample_rep",
        query=query,
        model_id=model_id,
        prompt_version=PROMPT_VERSION,
        params={"k_samples": k_samples},
    )
    if cached is not None:
        return cached

    prompt = f"Answer the following question.\nQuestion: {query}\nAnswer:"
    outputs = cached_generate(
        cache=cache,
        client=client,
        measure="sample_rep_generation",
        query=query,
        model_id=model_id,
        prompt_version=PROMPT_VERSION,
        prompt=prompt,
        n=k_samples,
        temperature=0.8,
        max_tokens=128,
        logprobs=False,
    )
    answers: List[str] = [item.text for item in outputs]
    avg_similarity = pairwise_average_similarity(answers)
    result = {
        "samples": answers,
        "average_similarity": avg_similarity,
        "sample_rep_uncertainty": 1.0 - avg_similarity,
    }
    cache.set(cache_key, result)
    return result

