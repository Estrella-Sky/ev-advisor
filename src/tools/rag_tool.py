"""RAG 工具：本地车型知识库的参数查询与按预算筛选。"""
from __future__ import annotations

from langchain_core.tools import tool

from src.rag.retriever import format_document, format_documents, get_car, search


@tool
def search_car_knowledge(
    query: str,
    max_price: float | None = None,
    min_range: float | None = None,
    brand: str | None = None,
    top_k: int = 5,
) -> str:
    """在本地车型知识库中做语义检索，可按预算、续航、品牌筛选。

    用于：参数查询、按需求推荐车型、竞品对比时获取基础参数。

    Args:
        query: 检索用的自然语言，例如「15 万 家用 有小孩 续航长」。
        max_price: 预算上限（万元），例如 15；只在按预算推荐时传。
        min_range: 最低 CLTC 续航（公里），例如 500。
        brand: 限定品牌，例如「比亚迪」。
        top_k: 返回车型数量，默认 5；按预算推荐时可传 8。
    """
    documents = search(query, top_k=top_k, max_price=max_price, min_range=min_range, brand=brand)
    return format_documents(documents)


@tool
def get_car_specs(model_name: str) -> str:
    """按车型名精确查询某一款车的完整参数，例如「比亚迪海豹」「小米SU7」。

    这是参数查询与竞品对比的首选工具：知识库中没有该车型时会明确返回没有，
    不要用其他相似车型替代。

    Args:
        model_name: 车型名，可带品牌，例如「比亚迪海豹」或「海豹」。
    """
    document = get_car(model_name)
    if document is None:
        return f"本地知识库中没有「{model_name}」这款车，无法提供参数。请告知用户知识库未收录该车型。"
    return format_document(document) + f"\n完整参数：{document.page_content}"

