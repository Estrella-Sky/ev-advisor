"""文本向量化：Qwen text-embedding-v3（DashScope OpenAI 兼容接口）。"""
from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

import config


def get_embeddings() -> OpenAIEmbeddings:
    """按配置构造 Embedding 客户端（无 Key 时抛出带指引的错误）。"""
    cfg = config.get_config()["embedding"]
    return OpenAIEmbeddings(
        model=cfg["model"],
        base_url=cfg["base_url"],
        api_key=config.require_api_key("embedding"),
        chunk_size=cfg["batch_size"],
        check_embedding_ctx_length=False,  # 非 OpenAI 端点，跳过 tiktoken 预处理
    )

