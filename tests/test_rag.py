"""RAG 层测试：数据质量、过滤条件与检索链路（用确定性假向量，不需要 API Key）。"""
from __future__ import annotations

import re
import zlib

import pytest
from langchain_core.embeddings import Embeddings

from src.data_processing.cleaner import COLUMNS, load_clean_rows
from src.rag.retriever import build_where, format_document, get_car, search
from src.rag.vector_store import build_documents, get_vector_store

DIM = 64


class FakeEmbeddings(Embeddings):
    """确定性哈希向量：只用于离线验证检索链路，不代表真实语义效果。"""

    @staticmethod
    def _vector(text: str) -> list[float]:
        vector = [0.0] * DIM
        for token in re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", text):
            vector[zlib.crc32(token.encode()) % DIM] += 1.0
        return vector

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


@pytest.fixture(scope="module")
def store():
    vector_store = get_vector_store(embedding_function=FakeEmbeddings(), persist=False)
    vector_store.add_documents(build_documents())
    return vector_store


def test_dataset_meets_quality_requirements():
    rows = load_clean_rows()
    assert len(rows) >= 50, "知识库车型数量应不少于 50 款"
    assert all(set(COLUMNS) <= set(row) for row in rows)
    names = [f"{row['brand']}{row['model']}" for row in rows]
    assert len(names) == len(set(names)), "品牌+车型不应重复"
    for row in rows:
        assert row["price_min"] <= row["price_max"]
        assert row["range_cltc"] >= 200 and row["axis_length"] >= 2000


def test_car_document_carries_metadata_for_filtering():
    row = next(item for item in load_clean_rows() if item["model"] == "海豹")
    documents = build_documents()
    document = next(item for item in documents if item.metadata["car_id"] == "比亚迪海豹")
    assert document.metadata["price_min"] == row["price_min"]
    assert "CTB" in document.page_content or "刀片" in document.page_content


def test_build_where_combinations():
    assert build_where() is None
    assert build_where(max_price=15) == {"price_min": {"$lte": 15.0}}
    assert build_where(brand="比亚迪") == {"brand": "比亚迪"}
    combined = build_where(max_price=20, min_range=600, brand="比亚迪")
    assert combined == {
        "$and": [
            {"price_min": {"$lte": 20.0}},
            {"range_cltc": {"$gte": 600.0}},
            {"brand": "比亚迪"},
        ]
    }


def test_search_respects_budget_filter(store):
    hits = search("15 万预算 家用 有小孩 空间大", max_price=15, top_k=5, store=store)
    assert hits, "预算 15 万以内应有候选车型"
    assert all(hit.metadata["price_min"] <= 15 for hit in hits)


def test_get_car_exact_match_and_unknown_model(store):
    hit = get_car("比亚迪海豹", store=store)
    assert hit is not None and hit.metadata["model"] == "海豹"
    # 车型名里带空格的也要能精确命中（回归：秦PLUS EV 曾被判成「知识库没有」）
    assert get_car("秦PLUS EV", store=store).metadata["model"] == "秦PLUS EV"
    assert get_car("比亚迪 秦PLUS EV", store=store).metadata["model"] == "秦PLUS EV"
    assert get_car("Model 3", store=store).metadata["model"] == "Model 3"
    assert get_car("AION Y Plus", store=store).metadata["model"] == "AION Y Plus"
    assert get_car("火星车 X9", store=store) is None, "不存在的车型必须返回 None，不能拿相似车型顶替"


def test_format_document_contains_key_fields(store):
    hit = get_car("比亚迪海豹", store=store)
    text = format_document(hit)
    assert "比亚迪 海豹" in text and "价格" in text and "续航" in text
