"""集中配置入口：读取 config/config.yaml，并用环境变量做覆盖。"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
_cache: dict | None = None


def _load_dotenv(path: Path) -> None:
    """把 .env 中的键值写入 os.environ（已存在的变量不覆盖）。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_config() -> dict:
    """全局配置（首次调用时加载 .env 与 config.yaml）。"""
    global _cache
    if _cache is None:
        _load_dotenv(ROOT / ".env")
        _cache = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
        _cache["llm"]["model"] = os.environ.get("EV_MODEL") or _cache["llm"]["model"]
        _cache["embedding"]["model"] = (
            os.environ.get("EV_EMBEDDING_MODEL") or _cache["embedding"]["model"]
        )
        _cache["logging"]["level"] = os.environ.get("EV_LOG_LEVEL") or _cache["logging"]["level"]
    return _cache


def path(relative: str) -> Path:
    """项目内相对路径 -> 绝对路径。"""
    return ROOT / relative


def api_key(section: str = "llm") -> str | None:
    """读取指定配置节使用的 API Key。"""
    env_name = get_config()[section]["api_key_env"]
    return os.environ.get(env_name) or None


def require_api_key(section: str = "llm") -> str:
    """取 API Key，缺失时抛出带操作指引的错误。"""
    key = api_key(section)
    if not key:
        env_name = get_config()[section]["api_key_env"]
        raise RuntimeError(
            f"未配置环境变量 {env_name}。请在项目根目录创建 .env（可复制 .env.example），"
            "填入阿里云百炼 DashScope 的 API Key 后重试。"
        )
    return key


def setup_logging() -> None:
    """统一日志格式，用于记录 Agent 的 Thought -> Action -> Observation 决策路径。"""
    logging.basicConfig(
        level=get_config()["logging"]["level"].upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

