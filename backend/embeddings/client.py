"""OpenAI 兼容 embedding 客户端（支持本地 sentence-transformers 模式）。

这一层只负责"文本转向量"，不关心 RAG 怎么排序或怎么组织 Prompt。
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import httpx

from config.settings import (
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
    USE_LOCAL_EMBEDDING,
    api_key_is_missing,
    load_course_env,
)


class EmbeddingClient:
    """调用 embedding 服务把文本转向量。

    默认走本地 sentence-transformers（USE_LOCAL_EMBEDDING=True），单次 <500ms；
    设为 False 时回退到 SiliconFlow 远程 API。
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        """保存 embedding 服务配置，并初始化进程内文本缓存。"""

        self.api_key = api_key
        self.base_url = base_url
        self.http_client = http_client
        self._cache: dict[str, list[float]] = {}

        # 本地模式：强制使用 LOCAL_EMBEDDING_MODEL，忽略传入的 model（那是远程模型名）。
        self.use_local = USE_LOCAL_EMBEDDING
        if self.use_local:
            self.model = LOCAL_EMBEDDING_MODEL
            self._local_model = None
        else:
            self.model = model  # 远程模式在 _settings() 里解析默认值

    def _cache_key(self, base_url: str, model: str, text: str) -> str:
        """用服务、模型和文本生成固定长度缓存 key。"""

        raw_key = json.dumps([base_url, model, text], ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def _settings(self) -> tuple[str | None, str, str]:
        """解析 embedding 调用所需的 Key、base_url 和模型名。"""

        load_course_env()
        if self.use_local:
            # 本地模式不需要 api_key；base_url 只用于区分缓存命名空间。
            return None, self.base_url or "local", self.model

        api_key = self.api_key or os.getenv("AGENT_OPENAI_API_KEY")
        if api_key_is_missing(api_key):
            raise RuntimeError("AGENT_OPENAI_API_KEY 未配置，无法调用商用 embedding 模型。")

        base_url = (self.base_url or os.getenv("AGENT_OPENAI_BASE_URL", DEFAULT_EMBEDDING_BASE_URL)).rstrip("/")
        model = self.model or os.getenv("AGENT_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        return api_key, base_url, model

    def _get_local_model(self):
        """懒加载本地 sentence-transformers 模型，避免远程模式时浪费内存。"""

        if self._local_model is None:
            # 国内访问 HuggingFace 较慢，默认走镜像；已设置 HF_ENDPOINT 时尊重用户配置。
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            from sentence_transformers import SentenceTransformer

            self._local_model = SentenceTransformer(self.model)
        return self._local_model

    def embed(self, text: str) -> list[float]:
        """把单条文本转换成 embedding。"""

        api_key, base_url, model = self._settings()
        cache_key = self._cache_key(base_url, model, text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self.use_local:
            # normalize_embeddings=True 让余弦相似度退化为点积，也是 bge 系列的标准用法。
            vector = self._get_local_model().encode(text, normalize_embeddings=True).tolist()
        else:
            response = self._post(
                f"{base_url}/embeddings",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "input": text},
            )
            response.raise_for_status()
            payload = response.json()
            embedding = payload["data"][0]["embedding"]
            vector = [float(value) for value in embedding]

        self._cache[cache_key] = vector
        return vector

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """批量转换文本，并复用本地缓存避免重复请求。"""

        api_key, base_url, model = self._settings()
        missing_texts: list[str] = []
        missing_keys: list[str] = []
        seen_missing_keys: set[str] = set()

        for text in texts:
            cache_key = self._cache_key(base_url, model, text)
            if cache_key in self._cache or cache_key in seen_missing_keys:
                continue
            seen_missing_keys.add(cache_key)
            missing_texts.append(text)
            missing_keys.append(cache_key)

        if missing_texts:
            if self.use_local:
                vectors = self._get_local_model().encode(
                    missing_texts, normalize_embeddings=True
                ).tolist()
                for cache_key, vec in zip(missing_keys, vectors):
                    self._cache[cache_key] = vec
            else:
                response = self._post(
                    f"{base_url}/embeddings",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={"model": model, "input": missing_texts},
                )
                response.raise_for_status()
                payload = response.json()
                data = sorted(
                    payload["data"],
                    key=lambda item: int(item.get("index", 0)),
                )
                for cache_key, item in zip(missing_keys, data):
                    self._cache[cache_key] = [float(value) for value in item["embedding"]]

        return [self._cache[self._cache_key(base_url, model, text)] for text in texts]

    def _post(self, url: str, headers: dict[str, str], json: dict[str, Any]) -> httpx.Response:
        """发送 embedding HTTP 请求，测试时可替换为 mock client。"""

        if self.http_client is not None:
            return self.http_client.post(url, headers=headers, json=json)
        return httpx.post(url, headers=headers, json=json, timeout=120)


DEFAULT_EMBEDDING_CLIENT = EmbeddingClient()


def read_embedding_model_name() -> str:
    """读取当前 embedding 模型名，用于调试输出。"""

    load_course_env()
    if USE_LOCAL_EMBEDDING:
        return LOCAL_EMBEDDING_MODEL
    return os.getenv("AGENT_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def embed_text(text: str, embedding_client: EmbeddingClient | None = None) -> list[float]:
    """用指定或默认 embedding 客户端向量化单条文本。"""

    client = embedding_client or DEFAULT_EMBEDDING_CLIENT
    return client.embed(text)


def embed_texts(texts: list[str], embedding_client: EmbeddingClient | None = None) -> list[list[float]]:
    """用指定或默认 embedding 客户端批量向量化文本。"""

    client = embedding_client or DEFAULT_EMBEDDING_CLIENT
    return client.embed_many(texts)
