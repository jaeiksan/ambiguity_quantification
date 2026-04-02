from typing import Dict, Optional

from ..cache import SQLiteCache
from ..llm_client import BaseLLMClient, LLMError, UnsupportedFeatureError
from .common import cached_value


PROMPT_VERSION = "token_entropy_v1"


def compute_mean_token_entropy(
    *,
    client: BaseLLMClient,
    cache: SQLiteCache,
    query: str,
    model_id: str,
) -> Dict[str, Optional[float]]:
    cache_key, cached = cached_value(
        cache=cache,
        measure="token_entropy",
        query=query,
        model_id=model_id,
        prompt_version=PROMPT_VERSION,
        params={},
    )
    if cached is not None:
        return cached

    try:
        mean_entropy = client.score_text_entropy(query)
    except (UnsupportedFeatureError, LLMError):
        mean_entropy = None
    result = {"mean_token_entropy": mean_entropy}
    cache.set(cache_key, result)
    return result
