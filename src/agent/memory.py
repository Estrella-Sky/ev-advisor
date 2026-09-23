"""多轮对话记忆：预算、家庭情况、偏好品牌、已讨论车型，以及指代消解。

对应需求文档模块五：用户说「第二个的续航呢」时，能正确指代上一轮推荐的车型。
"""
from __future__ import annotations

import logging
import re

import config
from src.data_processing.cleaner import car_names, distinct_brands

logger = logging.getLogger(__name__)

BUDGET_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*万")
ORDINAL_PATTERN = re.compile(r"第\s*([一二三四五六123456])\s*(?:个|款|台|辆)")
ORDINALS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6}
REFERENCE_WORDS = ("这款", "那款", "它", "这台", "那台")

FAMILY_KEYWORDS = {
    "孩子": "有小孩",
    "小孩": "有小孩",
    "宝宝": "有小孩",
    "老人": "有老人",
    "父母": "有老人",
    "老婆": "有配偶",
    "老公": "有配偶",
    "单身": "单人用车",
    "一个人": "单人用车",
}
NEED_KEYWORDS = ("续航", "空间", "智驾", "智能驾驶", "安全", "动力", "充电", "价格", "省油", "保值")

# 推荐结果末尾的机器可读标记，用于可靠地解析「第二个」这类指代
RECOMMEND_MARKER = re.compile(r"^\s*\[推荐顺序\]\s*(.+?)\s*$", re.MULTILINE)


def parse_recommendation_marker(text: str, names: list[str]) -> list[str]:
    """解析「[推荐顺序] 车型A｜车型B」标记；没有标记时返回空列表。"""
    match = RECOMMEND_MARKER.search(text)
    if not match:
        return []
    ordered: list[str] = []
    for item in re.split(r"[｜|、,，/]", match.group(1)):
        found = match_model_names(item.strip(), names)
        if found and found[0] not in ordered:
            ordered.append(found[0])
    return ordered


def strip_recommendation_marker(text: str) -> str:
    """对用户隐藏内部标记行。"""
    return RECOMMEND_MARKER.sub("", text).strip()


def strip_streaming_marker(text: str) -> str:
    """流式输出时隐藏 [推荐顺序] 标记。

    流式场景下标记可能只传到一半（例如刚收到「[推荐顺」），所以要同时处理
    完整标记与它任意长度的前缀，避免用户看到半截内部标记。
    """
    marker = "[推荐顺序]"
    index = text.find(marker)
    if index >= 0:
        return text[:index].rstrip()
    stripped = text.rstrip()
    for length in range(len(marker) - 1, 1, -1):
        if stripped.endswith(marker[:length]):
            return stripped[: -length].rstrip()
    return text


def match_model_names(text: str, names: list[str]) -> list[str]:
    """按出现顺序找出文本中的车型名，忽略被更长名称覆盖的短名（如「海豹」之于「比亚迪海豹」）。"""
    hits: list[tuple[int, int, str]] = []
    for name in names:
        for found in _model_pattern(name).finditer(text):
            hits.append((found.start(), found.end(), name))
    accepted: list[tuple[int, int, str]] = []
    # 起点相同时先处理更长的名称，再丢掉被它覆盖的短名称（「极氪007」里的「07」）
    for start, end, name in sorted(hits, key=lambda item: (item[0], -(item[1] - item[0]))):
        if any(start >= s and end <= e for s, e, _ in accepted):
            continue
        accepted.append((start, end, name))
    accepted.sort()
    return list(dict.fromkeys(name for _, _, name in accepted))


def _model_pattern(name: str):
    """允许名称内部有空格（模型常写成「比亚迪 元PLUS」），并对英文数字加词边界。"""
    body = r"\s*".join(re.escape(char) for char in name)
    if name.isascii():
        return re.compile(rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])", re.IGNORECASE)
    return re.compile(body)


class ConversationMemory:
    """一轮对话的短期记忆。会话级实例，由 UI 持有。"""

    def __init__(self, window: int | None = None, known_models: list[str] | None = None):
        self.window = window or config.get_config()["agent"]["memory_window"]
        self.known_models = known_models if known_models is not None else car_names()
        self.known_brands = distinct_brands()
        self.turns: list[tuple[str, str]] = []
        self.profile: dict = {}
        self.discussed: list[str] = []
        self.last_recommended: list[str] = []
        self.last_model: str | None = None

    # ---------- 写入 ----------

    def add_user(self, text: str) -> None:
        self._remember_profile(text)

    def add_ai(self, text: str) -> None:
        marked = parse_recommendation_marker(text, self.known_models)
        mentioned = self._match_models(text)
        # 有标记按标记；没标记但一次提到多款车，视为在列推荐清单
        if marked:
            self.last_recommended = marked
        elif len(mentioned) >= 2:
            self.last_recommended = mentioned
        if mentioned:
            self.last_model = mentioned[0]
        for name in marked or mentioned:
            if name not in self.discussed:
                self.discussed.append(name)

    def add_turn(self, user: str, assistant: str) -> None:
        self.turns.append((user, assistant))

    def clear(self) -> None:
        self.turns.clear()
        self.profile.clear()
        self.discussed.clear()
        self.last_recommended.clear()
        self.last_model = None

    # ---------- 读取 ----------

    def history_text(self, window: int | None = None) -> str:
        """最近 N 轮对话，用于 System Prompt。"""
        recent = self.turns[-(window or self.window) :]
        if not recent:
            return ""
        lines = ["## 最近对话"]
        for user, assistant in recent:
            lines.append(f"用户：{user}")
            lines.append(f"顾问：{assistant}")
        return "\n".join(lines)

    def profile_text(self) -> str:
        """用户画像，用于 System Prompt。"""
        if not self.profile and not self.discussed:
            return ""
        parts = ["## 已知用户信息"]
        for key, label in (("budget", "预算"), ("family", "家庭情况"), ("brands", "偏好品牌"), ("needs", "关注点")):
            value = self.profile.get(key)
            if value:
                parts.append(f"- {label}：{value}")
        if self.discussed:
            parts.append(f"- 已讨论车型：{'、'.join(self.discussed)}")
        if self.last_recommended:
            ordered = "、".join(
                f"第{index}个 {name}" for index, name in enumerate(self.last_recommended, start=1)
            )
            parts.append(f"- 上一轮推荐顺序：{ordered}")
        return "\n".join(parts)

    # ---------- 指代消解 ----------

    def resolve(self, text: str) -> str:
        """把「第二个」「它」这类指代替换成具体车型名（一句话里可能同时出现两种指代）。"""
        if not self.last_recommended and not self.last_model:
            return text

        resolved = text
        if self.last_recommended:
            def replace_ordinal(match: re.Match) -> str:
                token = match.group(1)
                index = ORDINALS.get(token) or int(token)
                if 1 <= index <= len(self.last_recommended):
                    logger.info("指代消解：%s -> %s", match.group(0), self.last_recommended[index - 1])
                    return self.last_recommended[index - 1]
                return match.group(0)

            resolved = ORDINAL_PATTERN.sub(replace_ordinal, resolved)

        # 「它」指向最近讨论的那一款；没有则退回首推车型
        pronoun_target = self.last_model or (self.last_recommended[0] if self.last_recommended else None)
        for word in REFERENCE_WORDS:
            if word in resolved and pronoun_target:
                logger.info("指代消解：%s -> %s", word, pronoun_target)
                resolved = resolved.replace(word, pronoun_target, 1)
                break
        return resolved

    # ---------- 内部 ----------

    def _remember_profile(self, text: str) -> None:
        budgets = [float(value) for value in BUDGET_PATTERN.findall(text)]
        if budgets:
            self.profile["budget"] = f"{max(budgets):g} 万以内"
        family = {label for key, label in FAMILY_KEYWORDS.items() if key in text}
        if family:
            self.profile["family"] = "、".join(sorted(family))
        brands = [brand for brand in self.known_brands if brand in text]
        if brands:
            self.profile["brands"] = "、".join(brands)
        needs = [word for word in NEED_KEYWORDS if word in text]
        if needs:
            merged = list(dict.fromkeys(self.profile.get("needs", "").split("、") + needs))
            self.profile["needs"] = "、".join(filter(None, merged))

    def _match_models(self, text: str) -> list[str]:
        return match_model_names(text, self.known_models)
