"""ChromaDB 向量库：把车型 CSV 构建成可语义检索的知识库。

用法：
    python -m src.rag.vector_store          # 增量构建（已存在则跳过）
    python -m src.rag.vector_store --force  # 强制重建
"""
from __future__ import annotations

import logging
import re

from langchain_chroma import Chroma
from langchain_core.documents import Document

import config
from src.data_processing.cleaner import load_clean_rows
from src.rag.embedder import get_embeddings

logger = logging.getLogger(__name__)

# 写入 Chroma metadata 的字段（必须是最简单的标量）
METADATA_FIELDS = (
    "brand",
    "model",
    "price_min",
    "price_max",
    "range_cltc",
    "battery_type",
    "battery_capacity",
    "axis_length",
    "drive_type",
    "acceleration",
    "ad_level",
)


def fmt(value: float) -> str:
    """17.980000 -> '17.98'，510.0 -> '510'。"""
    return f"{value:g}"


def norm_key(text: str) -> str:
    """去掉所有空白，用于车型名精确匹配（「秦PLUS EV」->「秦PLUSEV」）。"""
    return re.sub(r"\s+", "", text)


def car_to_text(row: dict) -> str:
    """把一行车型数据转成一句自然语言，供向量检索匹配。"""
    return (
        f"{row['brand']} {row['model']}：指导价 {fmt(row['price_min'])}-{fmt(row['price_max'])} 万元，"
        f"CLTC 续航 {fmt(row['range_cltc'])} 公里，{row['battery_type']}电池 "
        f"{fmt(row['battery_capacity'])} kWh，轴距 {fmt(row['axis_length'])} mm，"
        f"车身尺寸 {fmt(row['length'])}×{fmt(row['width'])}×{fmt(row['height'])} mm，"
        f"{row['drive_type']}驱动，零百加速 {fmt(row['acceleration'])} 秒，"
        f"智驾等级 {row['ad_level']}，安全配置：{row['safety_features']}，"
        f"核心卖点：{row['selling_points']}。"
    )


def car_to_document(row: dict) -> Document:
    """一行车型数据 -> 一个 Document（正文用于语义检索，metadata 用于过滤）。"""
    metadata = {field: row[field] for field in METADATA_FIELDS}
    metadata["car_id"] = norm_key(f"{row['brand']}{row['model']}")
    metadata["model_key"] = norm_key(row["model"])
    return Document(
        page_content=car_to_text(row),
        metadata=metadata,
        id=f"{row['brand']}-{row['model']}",
    )


def build_documents(csv_path=None) -> list[Document]:
    """读取全部车型并转成 Document 列表。"""
    return [car_to_document(row) for row in load_clean_rows(csv_path)]


def _persist_dir() -> str | None:
    """持久化目录；传 None 表示内存模式（测试用）。"""
    cfg = config.get_config()["rag"]
    directory = config.path(cfg["persist_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory)


def collection_name() -> str:
    return config.get_config()["rag"]["collection"]


def get_vector_store(
    embedding_function=None,
    persist: bool = True,
    autobuild: bool = True,
) -> Chroma:
    """打开（或创建）向量库；集合为空时自动构建，避免首次查询答不出参数。

    autobuild=False 供 build_index 内部调用，避免自引用递归。
    """
    store = Chroma(
        collection_name=collection_name(),
        embedding_function=embedding_function or get_embeddings(),
        persist_directory=_persist_dir() if persist else None,
    )
    if autobuild and count(store) == 0:
        documents = build_documents()
        store.add_documents(documents)
        logger.info("向量库为空，已自动写入 %d 款车型", len(documents))
    return store


def count(store: Chroma | None = None) -> int:
    """知识库中的车型数量。"""
    return (store or get_vector_store())._collection.count()  # noqa: SLF001


def build_index(force: bool = False, embedding_function=None, persist: bool = True) -> Chroma:
    """构建向量库；已存在且非 force 时直接复用。"""
    store = get_vector_store(embedding_function, persist, autobuild=False)
    if force and count(store):
        store.delete_collection()
        store = get_vector_store(embedding_function, persist, autobuild=False)
    if count(store):
        logger.info("向量库已有 %d 款车型，跳过构建（--force 可重建）", count(store))
        return store

    documents = build_documents()
    store.add_documents(documents)
    logger.info("已写入 %d 款车型到向量库", len(documents))
    return store


def main() -> None:
    import sys

    config.setup_logging()
    try:
        store = build_index(force="--force" in sys.argv)
    except RuntimeError as exc:  # 缺少 API Key 等配置问题，给出人话提示而不是堆栈
        print(f"构建失败：{exc}")
        raise SystemExit(1) from None
    print(f"向量库就绪，共 {count(store)} 款车型。")


if __name__ == "__main__":
    main()
