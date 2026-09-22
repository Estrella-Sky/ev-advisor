"""车型数据清洗与校验（对应需求文档 §6.3 数据质量要求）。

数据源为手工整理的公开资料 CSV，本模块负责读取、类型归一化与质量校验，
所有下游模块（向量库、工具）都通过 load_clean_rows() 取数，保证只有一份入口。
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import config

logger = logging.getLogger(__name__)

_rows_cache: dict[str, list[dict]] = {}

COLUMNS = (
    "brand",
    "model",
    "price_min",
    "price_max",
    "range_cltc",
    "battery_type",
    "battery_capacity",
    "axis_length",
    "length",
    "width",
    "height",
    "drive_type",
    "acceleration",
    "ad_level",
    "safety_features",
    "selling_points",
)
NUMERIC_COLUMNS = (
    "price_min",
    "price_max",
    "range_cltc",
    "battery_capacity",
    "axis_length",
    "length",
    "width",
    "height",
    "acceleration",
)


def _to_number(value: str) -> float:
    return float(str(value).strip())


def normalize_row(row: dict) -> dict:
    """去空白 + 数值列转 float；缺列补 None，多余列丢弃。"""
    cleaned: dict = {}
    for column in COLUMNS:
        value = str(row.get(column, "")).strip()
        if column in NUMERIC_COLUMNS:
            try:
                cleaned[column] = _to_number(value)
            except ValueError:
                cleaned[column] = None
        else:
            cleaned[column] = value
    return cleaned


def validate_row(row: dict) -> list[str]:
    """返回该行的问题列表，空列表表示通过校验。"""
    problems = [
        f"{column} 缺失"
        for column in COLUMNS
        if row.get(column) in (None, "")
    ]
    if row.get("price_min") is not None and row.get("price_max") is not None:
        if row["price_min"] > row["price_max"]:
            problems.append("price_min 大于 price_max")
    for column in ("range_cltc", "battery_capacity", "axis_length"):
        if row.get(column) is not None and row[column] <= 0:
            problems.append(f"{column} 必须为正数")
    return problems


def load_clean_rows(csv_path: Path | str | None = None) -> list[dict]:
    """读取车型 CSV，逐行清洗与校验；发现问题即抛错，避免脏数据进入知识库。"""
    target = Path(csv_path) if csv_path else config.path(config.get_config()["data"]["raw_csv"])
    cache_key = str(target)
    if cache_key in _rows_cache:
        return _rows_cache[cache_key]
    if not target.exists():
        raise FileNotFoundError(f"车型数据文件不存在：{target}")

    rows: list[dict] = []
    with target.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{target.name} 缺少字段：{', '.join(missing)}")
        for index, raw in enumerate(reader, start=2):
            row = normalize_row(raw)
            problems = validate_row(row)
            if problems:
                raise ValueError(f"{target.name} 第 {index} 行（{raw.get('model')}）数据有问题：{problems}")
            rows.append(row)

    if not rows:
        raise ValueError(f"{target.name} 中没有任何车型数据")
    logger.info("已加载 %d 款车型数据：%s", len(rows), target)
    _rows_cache[cache_key] = rows
    return rows


def distinct_brands(csv_path: Path | str | None = None) -> list[str]:
    """品牌列表（用于提示词与文档）。"""
    return sorted({row["brand"] for row in load_clean_rows(csv_path)})


def car_names(csv_path: Path | str | None = None) -> list[str]:
    """车型名列表（品牌全称 + 短名），按长度倒序，便于在文本中做最长匹配。

    短名只保留带中文或至少含一个字母的（「海豹」「元PLUS」「S05」），
    丢弃纯数字与单字符短名（「12」「07」「001」「X」）：它们在正文里会和
    「12 天」「X 型」这类普通文字撞车，把推荐顺序算错。
    """
    names: set[str] = set()
    for row in load_clean_rows(csv_path):
        names.add(f"{row['brand']}{row['model']}")
        short = row["model"]
        has_cjk = any("\u4e00" <= char <= "\u9fff" for char in short)
        has_letter = any(char.isalpha() for char in short)
        if has_cjk or (len(short) >= 2 and has_letter):
            names.add(short)
    return sorted(names, key=len, reverse=True)
