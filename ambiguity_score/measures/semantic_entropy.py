from typing import Dict, List

from ..cache import SQLiteCache
from ..llm_client import BaseLLMClient
from ..text_utils import cosine_similarity_from_vectors, shannon_entropy_from_counts
from .common import cached_generate, cached_value


PROMPT_VERSION = "answer_v3_embeddings_cleanup"


def compute_semantic_entropy(
    *,
    client: BaseLLMClient,
    cache: SQLiteCache,
    query: str,
    model_id: str,
    k_samples: int,
    embedding_model_id: str,
    cluster_threshold: float = 0.9,
    cluster_linkage: str = "complete",
) -> Dict:
    cache_key, cached = cached_value(
        cache=cache,
        measure="semantic_entropy",
        query=query,
        model_id=model_id,
        prompt_version=PROMPT_VERSION,
        params={
            "k_samples": k_samples,
            "cluster_threshold": cluster_threshold,
            "cluster_linkage": cluster_linkage,
            "embedding_model_id": embedding_model_id,
        },
    )
    if cached is not None:
        return cached

    prompt = f"Answer the following question.\nQuestion: {query}\nAnswer:"
    outputs = cached_generate(
        cache=cache,
        client=client,
        measure="semantic_entropy_generation",
        query=query,
        model_id=model_id,
        prompt_version=PROMPT_VERSION,
        prompt=prompt,
        n=k_samples,
        temperature=0.8,
        max_tokens=128,
        logprobs=False,
    )
    answers = [item.text for item in outputs]
    cleaned_answers = [_cleanup_answer_for_semantic_entropy(answer) for answer in answers]
    embeddings = client.embed_texts(cleaned_answers, model=embedding_model_id)
    clusters = _cluster_answers(
        cleaned_answers,
        embeddings=embeddings,
        threshold=cluster_threshold,
        linkage=cluster_linkage,
    )
    counts = [len(cluster) for cluster in clusters]
    result = {
        "samples": answers,
        "cleaned_samples": cleaned_answers,
        "clusters": clusters,
        "embedding_model": embedding_model_id,
        "cluster_threshold": cluster_threshold,
        "cluster_linkage": cluster_linkage,
        "semantic_entropy": shannon_entropy_from_counts(counts),
    }
    cache.set(cache_key, result)
    return result


def _cluster_answers(
    answers: List[str],
    embeddings: List[List[float]],
    threshold: float,
    linkage: str,
) -> List[List[str]]:
    clusters: List[List[str]] = []
    cluster_embeddings: List[List[float]] = []
    cluster_member_embeddings: List[List[List[float]]] = []
    for answer, embedding in zip(answers, embeddings):
        placed = False
        for index, representative_embedding in enumerate(cluster_embeddings):
            if _should_merge(
                embedding=embedding,
                cluster_member_embeddings=cluster_member_embeddings[index],
                representative_embedding=representative_embedding,
                threshold=threshold,
                linkage=linkage,
            ):
                cluster = clusters[index]
                cluster.append(answer)
                cluster_member_embeddings[index].append(embedding)
                cluster_embeddings[index] = _mean_embedding(cluster_embeddings[index], embedding, len(cluster))
                placed = True
                break
        if not placed:
            clusters.append([answer])
            cluster_embeddings.append(embedding)
            cluster_member_embeddings.append([embedding])
    return clusters


def _mean_embedding(current_mean: List[float], new_vector: List[float], count: int) -> List[float]:
    previous_count = count - 1
    return [
        ((current_mean[idx] * previous_count) + new_vector[idx]) / count
        for idx in range(len(current_mean))
    ]


def _should_merge(
    *,
    embedding: List[float],
    cluster_member_embeddings: List[List[float]],
    representative_embedding: List[float],
    threshold: float,
    linkage: str,
) -> bool:
    if linkage == "centroid":
        return cosine_similarity_from_vectors(embedding, representative_embedding) >= threshold
    if linkage == "complete":
        for member_embedding in cluster_member_embeddings:
            similarity = cosine_similarity_from_vectors(embedding, member_embedding)
            if similarity < threshold:
                return False
        return True
    raise ValueError(f"Unsupported cluster linkage: {linkage}")


def _cleanup_answer_for_semantic_entropy(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""

    stop_markers = [
        "\nHuman:",
        "\nAssistant:",
        "\nUser:",
        "\nQuestion:",
        "\nAnswer:",
        "\nWould you like",
        "\nIf you",
        "\nLet me know",
        "\nPlease let me know",
        "\nIs there anything else",
    ]
    for marker in stop_markers:
        index = cleaned.find(marker)
        if index != -1:
            cleaned = cleaned[:index].strip()

    cleaned = cleaned.split("\n\n", 1)[0].strip()
    cleaned = cleaned.split("\n", 1)[0].strip()

    sentence_endings = [". ", "! ", "? "]
    cut_positions = [
        cleaned.find(ending) + 1
        for ending in sentence_endings
        if cleaned.find(ending) != -1
    ]
    if cut_positions:
        cleaned = cleaned[: min(cut_positions)].strip()

    return cleaned or text.strip()
