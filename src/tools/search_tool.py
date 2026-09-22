"""联网搜索工具：DuckDuckGo，3 秒超时，失败时降级回本地知识库。"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

from langchain_core.tools import tool

import config

logger = logging.getLogger(__name__)

FALLBACK_HINT = "请改用本地知识库的数据作答，并明确告知用户实时信息暂时获取不到。"


def _search_blocking(query: str, max_results: int) -> list[dict]:
    from ddgs import DDGS  # 延迟导入：未安装时也能降级而不是启动失败

    with DDGS() as client:
        return list(client.text(query, max_results=max_results, region="cn-zh"))


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


@tool
def web_search(query: str) -> str:
    """联网搜索销量、车主口碑、最新新闻、降价等实时信息。

    Args:
        query: 搜索关键词，建议包含品牌与车型，例如「小米 SU7 月销量」。
    """
    cfg = config.get_config()["search"]
    # 不用 with：超时后不应再阻塞等待线程收尾，否则「3 秒超时」形同虚设
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        results = pool.submit(_search_blocking, query, cfg["max_results"]).result(
            timeout=cfg["timeout"]
        )
    except FutureTimeout:
        logger.warning("联网搜索超时：%s", query)
        return f"联网搜索超过 {cfg['timeout']} 秒未返回结果。{FALLBACK_HINT}"
    except Exception as exc:  # 网络异常、依赖缺失等都走同一个降级路径
        logger.warning("联网搜索失败：%s（%s）", query, exc)
        return f"联网搜索失败：{exc}。{FALLBACK_HINT}"
    finally:
        # ponytail: 线程留给后台自然结束，进程退出时由解释器回收；
        # 若将来出现线程堆积，再改成带取消的常驻执行器。
        pool.shutdown(wait=False)

    if not results:
        return f"联网搜索没有找到相关结果。{FALLBACK_HINT}"

    lines = [f"联网搜索结果（来源：DuckDuckGo，共 {len(results)} 条）："]
    for item in results:
        title = _clip(item.get("title", ""), cfg["snippet_max_chars"])
        body = _clip(item.get("body", ""), cfg["snippet_max_chars"])
        link = item.get("href", "")
        lines.append(f"- {title}｜{body}（{link}）")
    lines.append("请在回答中注明这些信息来自联网搜索，并提示用户以官方发布为准。")
    return "\n".join(lines)
