import json
import math
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .cache import SQLiteCache
from .text_utils import normalize_text


class LLMError(RuntimeError):
    pass


class UnsupportedFeatureError(LLMError):
    pass


@dataclass
class GeneratedText:
    text: str
    token_entropies: Optional[List[float]] = None


class BaseLLMClient:
    def generate(
        self,
        prompt: str,
        *,
        n: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 128,
        logprobs: bool = False,
    ) -> List[GeneratedText]:
        raise NotImplementedError

    def score_text_entropy(self, text: str) -> Optional[float]:
        raise UnsupportedFeatureError("Text scoring is not supported by this backend.")

    def embed_texts(self, texts: Sequence[str], *, model: Optional[str] = None) -> List[List[float]]:
        raise UnsupportedFeatureError("Embeddings are not supported by this backend.")


class HFLocalClient(BaseLLMClient):
    """
    Local Hugging Face backend.
    - generation: model.generate
    - entropy: full-vocab per-token entropy (APA-style)
    - embeddings: mean pooled last hidden state (semantic clustering proxy)
    """

    def __init__(
        self,
        *,
        model_id: str,
        embedding_model_id: Optional[str] = None,
        device: str = "auto",
        torch_dtype: str = "bfloat16",
        max_input_length: int = 1024,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise LLMError("HF backend requires torch and transformers packages.") from exc

        self._torch = torch
        self.model_id = model_id
        self.embedding_model_id = embedding_model_id or model_id
        self.max_input_length = max_input_length

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        selected_dtype = dtype_map.get(torch_dtype, torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "torch_dtype": selected_dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        if device == "auto":
            model_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

        if device != "auto":
            self.model.to(device)
        self.model.eval()

        try:
            self.device = next(self.model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")

    def generate(
        self,
        prompt: str,
        *,
        n: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 128,
        logprobs: bool = False,
    ) -> List[GeneratedText]:
        del logprobs
        torch = self._torch
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        outputs: List[GeneratedText] = []
        for _ in range(n):
            do_sample = temperature > 0
            generate_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if do_sample:
                generate_kwargs["temperature"] = temperature
                generate_kwargs["top_p"] = 0.95
            with torch.no_grad():
                generated = self.model.generate(
                    **encoded,
                    **generate_kwargs,
                )
            prompt_len = encoded["input_ids"].shape[1]
            completion_ids = generated[0, prompt_len:]
            text = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            outputs.append(GeneratedText(text=text, token_entropies=None))
        return outputs

    def score_text_entropy(self, text: str) -> Optional[float]:
        torch = self._torch
        normalized = normalize_text(text)
        if not normalized:
            return 0.0
        encoded = self.tokenizer(
            normalized,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            # logits: [1, seq_len, vocab]
            probs = torch.softmax(logits, dim=-1)
            token_entropies = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
            mean_entropy = token_entropies.mean().item()
        return float(mean_entropy)

    def embed_texts(self, texts: Sequence[str], *, model: Optional[str] = None) -> List[List[float]]:
        del model
        torch = self._torch
        vectors: List[List[float]] = []
        for text in texts:
            encoded = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_length,
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
            hidden = outputs.hidden_states[-1][0]  # [seq_len, hidden]
            mask = attention_mask[0].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=0) / mask.sum(dim=0).clamp_min(1.0)
            vectors.append([float(value) for value in pooled.detach().cpu().tolist()])
        return vectors


class OpenAICompatibleClient(BaseLLMClient):
    def __init__(
        self,
        *,
        model_id: str,
        base_url: str,
        api_key: str,
        embedding_model_id: Optional[str] = None,
        cache: Optional[SQLiteCache] = None,
        timeout: int = 120,
    ) -> None:
        self.model_id = model_id
        self.embedding_model_id = embedding_model_id or os.environ.get("EMBEDDING_MODEL_ID", "text-embedding-3-small")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.cache = cache
        self.timeout = timeout

    def _post_json(self, path: str, payload: Dict) -> Dict:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMError(f"HTTP {exc.code} from {url}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"Request failed for {url}: {exc}") from exc

    def generate(
        self,
        prompt: str,
        *,
        n: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 128,
        logprobs: bool = False,
    ) -> List[GeneratedText]:
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "n": n,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 5

        data = self._post_json("/chat/completions", payload)
        items: List[GeneratedText] = []
        for choice in data.get("choices", []):
            content = choice.get("message", {}).get("content", "")
            token_entropies = self._extract_chat_token_entropies(choice.get("logprobs"))
            items.append(GeneratedText(text=content.strip(), token_entropies=token_entropies))
        return items

    def _extract_chat_token_entropies(self, logprob_block: Optional[Dict]) -> Optional[List[float]]:
        if not logprob_block:
            return None
        content = logprob_block.get("content")
        if not isinstance(content, list):
            return None
        entropies: List[float] = []
        for token_info in content:
            top_logprobs = token_info.get("top_logprobs") or []
            values = [candidate.get("logprob") for candidate in top_logprobs if candidate.get("logprob") is not None]
            entropy = _entropy_from_logprobs(values)
            if entropy is not None:
                entropies.append(entropy)
        return entropies or None

    def score_text_entropy(self, text: str) -> Optional[float]:
        try:
            return self._score_text_entropy_via_completions(text)
        except LLMError:
            return self._score_text_entropy_via_chat_echo(text)

    def _score_text_entropy_via_completions(self, text: str) -> Optional[float]:
        payload = {
            "model": self.model_id,
            "prompt": normalize_text(text),
            "max_tokens": 0,
            "temperature": 0,
            "echo": True,
            "logprobs": 5,
        }
        data = self._post_json("/completions", payload)
        choices = data.get("choices", [])
        if not choices:
            raise LLMError("No choices returned from /completions endpoint.")
        logprobs = choices[0].get("logprobs") or {}
        top_logprobs = logprobs.get("top_logprobs")
        if not top_logprobs:
            return None
        entropies: List[float] = []
        for token_candidates in top_logprobs:
            if not isinstance(token_candidates, dict):
                continue
            entropy = _entropy_from_logprobs(list(token_candidates.values()))
            if entropy is not None:
                entropies.append(entropy)
        if not entropies:
            return None
        return sum(entropies) / len(entropies)

    def _score_text_entropy_via_chat_echo(self, text: str) -> Optional[float]:
        normalized = normalize_text(text)
        prompt = (
            "Repeat the text exactly as provided. Do not add, remove, or explain anything.\n"
            f"Text: {normalized}"
        )
        outputs = self.generate(
            prompt,
            n=1,
            temperature=0.0,
            max_tokens=max(32, len(normalized.split()) * 4),
            logprobs=True,
        )
        if not outputs or not outputs[0].token_entropies:
            return None
        token_entropies = outputs[0].token_entropies
        return sum(token_entropies) / len(token_entropies)

    def embed_texts(self, texts: Sequence[str], *, model: Optional[str] = None) -> List[List[float]]:
        payload = {
            "model": model or self.embedding_model_id,
            "input": list(texts),
        }
        data = self._post_json("/embeddings", payload)
        vectors: List[List[float]] = []
        for item in data.get("data", []):
            embedding = item.get("embedding")
            if isinstance(embedding, list):
                vectors.append([float(value) for value in embedding])
        if len(vectors) != len(texts):
            raise LLMError("Embeddings response size does not match input size.")
        return vectors


class FakeLLMClient(BaseLLMClient):
    def generate(
        self,
        prompt: str,
        *,
        n: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 128,
        logprobs: bool = False,
    ) -> List[GeneratedText]:
        del temperature, max_tokens
        query = _extract_query_from_prompt(prompt)
        outputs: List[GeneratedText] = []
        for index in range(n):
            text = self._answer(query, sample_index=index, disambiguate=_looks_like_disambiguation(prompt))
            token_entropies = [self.score_text_entropy(text) or 0.0] if logprobs else None
            outputs.append(GeneratedText(text=text, token_entropies=token_entropies))
        return outputs

    def score_text_entropy(self, text: str) -> Optional[float]:
        normalized = normalize_text(text)
        if not normalized:
            return 0.0
        tokens = normalized.split()
        unique = len(set(tokens))
        ambiguity_markers = {
            "uga",
            "jaguar",
            "apple",
            "mercury",
            "python",
            "charge",
            "giants",
            "bass",
        }
        marker_bonus = sum(1 for token in tokens if token in ambiguity_markers) * 0.25
        return (unique / max(len(tokens), 1)) + marker_bonus

    def embed_texts(self, texts: Sequence[str], *, model: Optional[str] = None) -> List[List[float]]:
        del model
        return [_fake_embedding(text) for text in texts]

    def _answer(self, query: str, *, sample_index: int, disambiguate: bool) -> str:
        normalized = normalize_text(query)
        if disambiguate:
            if "uga" in normalized:
                return "Where is the University of Georgia in Athens, Georgia, located?"
            if "jaguar" in normalized:
                return "Tell me about the jaguar animal in the wild."
            if "apple" in normalized:
                return "What is Apple Inc.'s current stock price?"
            if "mercury" in normalized:
                return "What is the planet Mercury's average distance from the Sun?"
            return query.strip()

        canned = {
            "where is uga": [
                "UGA is in Athens, Georgia.",
                "UGA commonly refers to the University of Georgia in Athens.",
                "UGA may refer to the University of Georgia, based in Athens.",
            ],
            "what is apple worth": [
                "Apple is worth over 3 trillion dollars in market capitalization.",
                "Apple Inc. has a market cap in the multi-trillion-dollar range.",
                "Apple's valuation is roughly several trillion US dollars.",
            ],
            "tell me about jaguar": [
                "Jaguar is a big cat native to the Americas.",
                "Jaguar can also refer to a luxury car brand.",
                "Jaguar may mean either the animal or the car company.",
            ],
        }
        variants = canned.get(normalized)
        if variants:
            return variants[sample_index % len(variants)]
        return f"Answer about: {query.strip()}"


def build_llm_client(
    *,
    backend: str,
    cache: Optional[SQLiteCache] = None,
    model_id: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    embedding_model_id: Optional[str] = None,
    hf_device: str = "auto",
    hf_torch_dtype: str = "bfloat16",
    hf_max_input_length: int = 1024,
) -> BaseLLMClient:
    if backend == "fake":
        return FakeLLMClient()
    if backend == "hf":
        model_id = model_id or os.environ.get("MODEL_ID")
        if not model_id:
            raise LLMError("HF backend requires MODEL_ID.")
        return HFLocalClient(
            model_id=model_id,
            embedding_model_id=embedding_model_id,
            device=hf_device,
            torch_dtype=hf_torch_dtype,
            max_input_length=hf_max_input_length,
        )

    model_id = model_id or os.environ.get("MODEL_ID")
    base_url = base_url or os.environ.get("OPENAI_BASE_URL")
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
    if not model_id or not base_url:
        raise LLMError("OPENAI backend requires MODEL_ID and OPENAI_BASE_URL.")
    return OpenAICompatibleClient(
        model_id=model_id,
        base_url=base_url,
        api_key=api_key,
        embedding_model_id=embedding_model_id,
        cache=cache,
    )


def _extract_tail(prompt: str) -> str:
    lines = [line.strip() for line in prompt.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1].removeprefix("Input Question:").removeprefix("Question:").strip()


def _extract_query_from_prompt(prompt: str) -> str:
    patterns = [
        r"Input Question:\s*(.+?)\nDisambiguation:",
        r"Question:\s*(.+?)\nAnswer:",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, prompt, flags=re.IGNORECASE | re.DOTALL)
        if matches:
            return matches[-1].strip()
    return _extract_tail(prompt)


def _looks_like_disambiguation(prompt: str) -> bool:
    lowered = prompt.lower()
    return "disambiguation:" in lowered or "enhance it" in lowered


def _entropy_from_logprobs(logprobs: Sequence[Optional[float]]) -> Optional[float]:
    clean = [value for value in logprobs if value is not None]
    if not clean:
        return None
    probs = [math.exp(value) for value in clean]
    mass = sum(probs)
    if mass <= 0:
        return None
    probs = [value / mass for value in probs]
    entropy = 0.0
    for prob in probs:
        if prob > 0:
            entropy -= prob * math.log(prob)
    return entropy


def _fake_embedding(text: str) -> List[float]:
    normalized = normalize_text(text)
    buckets = [0.0] * 16
    for token in normalized.split():
        index = sum(ord(char) for char in token) % len(buckets)
        buckets[index] += 1.0
    total = sum(buckets) or 1.0
    return [value / total for value in buckets]
