"""Agent 核心：意图路由（Router）→ 提示规划（Planner）→ ReAct 执行（Executor）。

对应需求文档 §3.1 的功能架构与 §5.3.1 的决策流程。
"""
from __future__ import annotations

import logging
import re
import time
from typing import Literal

from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import config
from config.prompts import SYSTEM_PROMPT, render_context
from src.agent.memory import ConversationMemory, match_model_names, strip_recommendation_marker
from src.data_processing.cleaner import car_names
from src.tools import ALL_TOOLS

logger = logging.getLogger(__name__)

IntentName = Literal["recommend", "query_params", "compare", "market_info", "calculate", "chat"]


class Intent(BaseModel):
    """意图识别结果与抽取到的实体。"""

    intent: IntentName = Field(
        description="用户意图：recommend 推荐、query_params 参数查询、compare 对比、"
        "market_info 市场信息、calculate 价格计算、chat 闲聊"
    )
    budget: float | None = Field(default=None, description="预算上限，单位万元；没有就留空")
    models: list[str] = Field(default_factory=list, description="用户提到的车型名")
    needs: list[str] = Field(default_factory=list, description="用户关注点，如续航、空间、智驾、安全")


# 各意图对应的执行路径提示（Planner 的输出，注入当轮用户消息）
ROUTE_HINTS = {
    "recommend": "先确认预算与用车场景；再用 search_car_knowledge(max_price=预算, top_k=8) 取候选，"
    "最多推荐 3 款并逐一说明理由，最后提示可继续对比；回答末尾必须单独加一行"
    "「[推荐顺序] 车型A｜车型B｜车型C」，按推荐先后排列。",
    "query_params": "用 get_car_specs 取该车型完整参数后直接回答，不要展开无关卖点。",
    "compare": "先用 get_car_specs 逐款取参数，需要时用 web_search 补充市场表现，"
    "再用 Markdown 表格对比并给出购买建议。",
    "market_info": "用 web_search 查询最新信息，注明来源与时间；搜索失败则说明并改用本地知识库。",
    "calculate": "用 calc_landing_price 计算，列出购置税、保险、上牌费与贷款月供明细；"
    "直接采用工具返回值里的税费口径与合计，不要自行改成免征或重新计算。",
    "chat": "直接回答，不需要调用工具；可以引导用户说出预算与用车需求。",
}

INTENT_KEYWORDS = (
    ("compare", ("对比", "比较", "哪个好", "哪个更", "区别", "差别", "vs")),
    ("calculate", ("落地", "购置税", "月供", "贷款", "首付", "上牌", "总价")),
    ("market_info", ("销量", "卖了多少", "口碑", "降价", "优惠", "新闻", "保值率", "最新")),
    ("query_params", ("续航", "电池", "参数", "轴距", "尺寸", "加速", "配置", "充电", "多少公里")),
    ("recommend", ("推荐", "买什么", "选哪", "适合", "预算", "帮我选", "家用")),
)
NEED_WORDS = ("续航", "空间", "智驾", "智能驾驶", "安全", "动力", "充电", "保值", "便宜")

# 展示用的中文标签与执行路径（对应 §3.1 的 Router -> Planner -> Executor）
INTENT_LABELS = {
    "recommend": "按预算推荐",
    "query_params": "参数查询",
    "compare": "竞品对比",
    "market_info": "市场信息",
    "calculate": "落地价计算",
    "chat": "闲聊",
}
PLAN_LABELS = {
    "recommend": "RAG 按预算筛选候选 → LLM 排序并给出理由",
    "query_params": "RAG 精确查询车型参数",
    "compare": "RAG 逐款取参数（必要时联网补充）→ Markdown 对比表",
    "market_info": "联网检索实时信息 → 摘要（失败降级本地知识库）",
    "calculate": "调用计算器工具 → 税费/保险/月供明细",
    "chat": "不调用工具，直接回答",
}


def clip(text: str, limit: int = 220) -> str:
    """压平空白并截断，用于展示工具返回。"""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def rules_intent(text: str) -> Intent:
    """规则兜底路由：LLM 不可用时仍能给出意图与实体。"""
    lowered = text.lower().replace(" ", "")
    intent: IntentName = "chat"
    for name, keywords in INTENT_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            intent = name  # type: ignore[assignment]
            break
    budgets = [float(value) for value in re.findall(r"(\d+(?:\.\d+)?)\s*万", text)]
    return Intent(
        intent=intent,
        budget=max(budgets) if budgets else None,
        models=match_model_names(text, car_names()),
        needs=[word for word in NEED_WORDS if word in text],
    )


def build_llm() -> ChatOpenAI:
    """按配置构造 Qwen（DashScope OpenAI 兼容接口）客户端。"""
    cfg = config.get_config()["llm"]
    return ChatOpenAI(
        model=cfg["model"],
        base_url=cfg["base_url"],
        api_key=config.require_api_key("llm"),
        temperature=cfg["temperature"],
        timeout=cfg["timeout"],
    )


def message_text(content) -> str:
    """兼容字符串与多模态分块两种消息内容格式。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            parts.append(block.get("text", "") if isinstance(block, dict) else str(block))
        return "".join(parts)
    return str(content)


class EVAdvisor:
    """智能导购 Agent：一个实例服务一段会话。"""

    def __init__(self, memory: ConversationMemory | None = None, llm=None, tools=None):
        self.memory = memory or ConversationMemory()
        self.tools = tools or ALL_TOOLS
        self._llm = llm

    @property
    def llm(self):
        if self._llm is None:
            self._llm = build_llm()
        return self._llm

    def reset(self) -> None:
        """清空记忆，开始新会话。"""
        self.memory.clear()

    def classify(self, text: str) -> Intent:
        """意图识别：优先用 Qwen Function Calling，失败降级为规则路由。"""
        try:
            structured = self.llm.with_structured_output(Intent)
            return structured.invoke(
                "识别下面这句购车咨询的意图，并抽取预算、车型、关注点。"
                "预算单位是万元；没有提到的字段留空。\n\n"
                f"用户：{text}"
            )
        except Exception as exc:
            logger.warning("意图识别降级为规则路由：%s", exc)
            return rules_intent(text)

    def chat(self, message: str) -> str:
        """处理一轮用户输入，只返回 Markdown 回答。"""
        return self.chat_with_trace(message)[0]

    def chat_with_trace(self, message: str) -> tuple[str, str]:
        """处理一轮用户输入，返回（回答, Agent 决策路径 Markdown）。

        决策路径用于界面展示：指代消解 → 意图识别 → 实体抽取 → 执行路径 →
        工具调用 → 工具返回 → 耗时，让 ReAct 循环对用户可见。
        """
        steps: list[str] = []
        started = time.monotonic()
        try:
            resolved = self.memory.resolve(message)  # 指代消解：第二个 -> 具体车型
            if resolved != message:
                steps.append(f"🔗 **指代消解**　`{message}` → `{resolved}`")
            self.memory.add_user(resolved)
            intent = self.classify(resolved)
            # 价格计算的关键词不会误伤（落地/月供/贷款…），LLM 漏判时用规则纠正
            if rules_intent(resolved).intent == "calculate" and intent.intent != "calculate":
                logger.info("路由修正：%s -> calculate", intent.intent)
                intent = intent.model_copy(update={"intent": "calculate"})
                steps.append("🧭 **路由修正**　命中价格关键词，意图纠正为 `calculate`")
            logger.info("Intent: %s | 实体: %s", intent.intent, intent.model_dump())
            steps.append(f"🎯 **意图识别**　`{intent.intent}`（{INTENT_LABELS[intent.intent]}）")
            entities = self._entity_text(intent)
            if entities:
                steps.append(f"🧩 **实体抽取**　{entities}")
            steps.append(f"🗺 **执行路径**　{PLAN_LABELS[intent.intent]}")

            system_prompt = SYSTEM_PROMPT.format(
                context=render_context(self.memory.profile_text(), self.memory.history_text())
            )
            user_prompt = f"{resolved}\n\n[执行路径] {ROUTE_HINTS[intent.intent]}"
            answer = self._execute(system_prompt, user_prompt, steps)

            self.memory.add_ai(answer)  # 读取 [推荐顺序] 标记，决定「第二个」指向谁
            answer = strip_recommendation_marker(answer)
            self.memory.add_turn(message, answer)
            steps.append(f"⏱ **总耗时**　{time.monotonic() - started:.1f} 秒")
            return answer, "\n".join(f"- {step}" for step in steps)
        except RuntimeError as exc:  # 配置类问题：API Key 缺失等
            logger.error("配置错误：%s", exc)
            steps.append(f"⚠️ **中断**　{exc}")
            return f"⚠️ 暂时无法回答：{exc}", "\n".join(f"- {step}" for step in steps)
        except Exception as exc:
            logger.exception("处理用户输入失败")
            steps.append(f"⚠️ **中断**　{exc}")
            return (
                f"⚠️ 处理失败：{exc}。可以换个说法再问一次。",
                "\n".join(f"- {step}" for step in steps),
            )

    @staticmethod
    def _entity_text(intent: Intent) -> str:
        """把抽取到的实体拼成一行展示文本。"""
        parts = []
        if intent.budget is not None:
            parts.append(f"预算 {intent.budget:g} 万")
        if intent.models:
            parts.append("车型 " + "、".join(intent.models))
        if intent.needs:
            parts.append("关注点 " + "、".join(intent.needs))
        return " ｜ ".join(parts)

    def _execute(self, system_prompt: str, user_prompt: str, steps: list[str]) -> str:
        """ReAct 执行：Thought → Action → Observation 循环，同时记录决策轨迹。"""
        agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt,
            name="ev_advisor",
        )
        limit = config.get_config()["agent"]["max_iterations"] * 2
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_prompt}]},
            config={"recursion_limit": limit},
        )
        messages = result["messages"]
        self._collect_trace(messages, steps)
        return message_text(messages[-1].content)

    @staticmethod
    def _collect_trace(messages, steps: list[str]) -> None:
        """记录 Agent 决策路径：写日志 + 追加到界面展示的轨迹。"""
        tool_index = 0
        for message in messages:
            for call in getattr(message, "tool_calls", None) or []:
                tool_index += 1
                logger.info("Action: %s(%s)", call.get("name"), call.get("args"))
                steps.append(
                    f"🛠 **工具调用 ×{tool_index}**　`{call.get('name')}`"
                    f"　参数 `{call.get('args')}`"
                )
            if isinstance(message, ToolMessage):
                observation = message_text(message.content)
                logger.info("Observation: %s", observation[:200])
                steps.append(f"👀 **工具返回**　{clip(observation)}")
