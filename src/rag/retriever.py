"""检索逻辑：语义检索 + 结构化过滤（价格 / 续航 / 品牌 / 车型名）。"""
from __future__ import annotations

import logging

from langchain_core.documents import Document

import config
from src.rag.vector_store import fmt, get_vector_store, norm_key

logger = logging.getLogger(__name__)


def build_where(max_price: float | None = None, min_range: float | None = None, brand: str | None = None) -> dict | None:
    """组装 Chroma 的 metadata 过滤条件；无过滤条件时返回 None。"""
    clauses: list[dict] = []
    if max_price is not None:
        # 预算口径：该车型的最低配在预算内就算候选
        clauses.append({"price_min": {"$lte": float(max_price)}})
    if min_range is not None:
        clauses.append({"range_cltc": {"$gte": float(min_range)}})
    if brand:
        clauses.append({"brand": brand})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _store(store=None):
    return store or get_vector_store()


def search(
    query: str,
    top_k: int | None = None,
    max_price: float | None = None,
    min_range: float | None = None,
    brand: str | None = None,
    store=None,
) -> list[Document]:
    """语义检索 + 结构化过滤。"""
    k = top_k or config.get_config()["rag"]["top_k"]
    where = build_where(max_price, min_range, brand)
    hits = _store(store).similarity_search(query, k=k, filter=where)
    logger.info("RAG 检索 %r（filter=%s）命中 %d 条", query, where, len(hits))
    return hits


def get_car(model_name: str, store=None) -> Document | None:
    """按车型名精确查找；精确命中失败后用语义检索兜底，仍不匹配则返回 None。

    返回 None 是「知识库中没有该车型」的明确信号，避免用相似车型冒充。
    """
    key = norm_key(model_name)
    if not key:
        return None
    vector_store = _store(store)
    for field in ("car_id", "model_key"):
        found = vector_store.get(where={field: key}, limit=1)
        documents = found.get("documents") or []
        if documents:
            return Document(page_content=documents[0], metadata=(found["metadatas"] or [{}])[0])

    candidates = vector_store.similarity_search(model_name, k=1)
    if candidates:
        candidate = candidates[0]
        name = norm_key(f"{candidate.metadata.get('brand', '')}{candidate.metadata.get('model', '')}")
        if key in name or name in key:
            return candidate
    return None


def format_document(document: Document) -> str:
    """Document -> 给 LLM 的紧凑文本（含关键字段，便于对比与引用）。"""
    meta = document.metadata
    price = f"{fmt(meta['price_min'])}-{fmt(meta['price_max'])} 万"
    return (
        f"{meta['brand']} {meta['model']} | 价格 {price} | 续航 {fmt(meta['range_cltc'])} km | "
        f"{meta['battery_type']} {fmt(meta['battery_capacity'])} kWh | 轴距 {fmt(meta['axis_length'])} mm | "
        f"{meta['drive_type']} | 零百 {fmt(meta['acceleration'])} s | 智驾 {meta['ad_level']}"
    )


def format_documents(documents: list[Document]) -> str:
    """多条检索结果 -> 一段文本；空结果给出明确提示。"""
    if not documents:
        return "本地知识库中没有匹配的车型。"
    return "\n".join(format_document(document) for document in documents)
