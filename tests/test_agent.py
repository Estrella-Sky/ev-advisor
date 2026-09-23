"""Agent 层测试：规则路由、实体抽取、多轮记忆与指代消解。"""
from __future__ import annotations

import config
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool

from src.agent.core import EVAdvisor, Intent, rules_intent
from src.agent.memory import (
    ConversationMemory,
    match_model_names,
    parse_recommendation_marker,
    strip_recommendation_marker,
)
from src.data_processing.cleaner import car_names


class ScriptedToolCallingModel(BaseChatModel):
    """按脚本回答的假模型：第一轮调用工具，第二轮给出最终答案。

    用来在不需要 API Key 的情况下验证 ReAct 执行链路本身。
    """

    responses: list[AIMessage]

    @property
    def _llm_type(self) -> str:
        return "scripted-tool-calling"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001, ANN003
        used = sum(1 for message in messages if message.type == "ai")
        reply = self.responses[min(used, len(self.responses) - 1)]
        return ChatResult(generations=[ChatGeneration(message=reply)])


def test_rules_intent_covers_six_types():
    cases = {
        "15万预算推荐什么车": "recommend",
        "比亚迪海豹的续航是多少": "query_params",
        "对比一下 Model Y 和理想 L6": "compare",
        "小米SU7上个月卖了多少台": "market_info",
        "裸车20万落地多少钱": "calculate",
        "你好呀": "chat",
    }
    for text, expected in cases.items():
        assert rules_intent(text).intent == expected, text


def test_rules_intent_extracts_entities():
    intent = rules_intent("15万预算，家里有小孩，主要看续航")
    assert intent.budget == 15.0
    assert "续航" in intent.needs


def test_short_model_names_need_word_boundaries():
    names = car_names()
    assert match_model_names("极氪007 和 阿维塔07", names) == ["极氪007", "阿维塔07"]
    assert match_model_names("XNGP 是智驾系统", names) == []
    assert match_model_names("推荐 比亚迪 元PLUS 和 特斯拉 Model 3", names) == [
        "比亚迪元PLUS",
        "特斯拉Model 3",
    ]


def test_memory_ignores_plain_numbers_in_prose():
    """回归：答案里的「12 天」「2720mm」不能被当成车型「阿维塔 12」。"""
    memory = ConversationMemory()
    memory.add_ai(
        "首选 比亚迪元PLUS：充一次电能用12天以上，轴距2720mm。\n"
        "备选 吉利银河E5：价格更低。"
    )
    assert memory.last_recommended == ["比亚迪元PLUS", "吉利银河E5"]


def test_recommendation_marker_wins_over_prose_mentions():
    answer = (
        "推荐比亚迪元PLUS，比海鸥、好猫更宽敞。\n"
        "[推荐顺序] 比亚迪元PLUS｜吉利银河E5｜比亚迪海豚\n"
    )
    memory = ConversationMemory()
    memory.add_ai(answer)

    assert memory.last_recommended == ["比亚迪元PLUS", "吉利银河E5", "比亚迪海豚"]
    assert memory.resolve("第二个的续航呢？") == "吉利银河E5的续航呢？"
    assert parse_recommendation_marker(answer, car_names()) == memory.last_recommended
    assert "[推荐顺序]" not in strip_recommendation_marker(answer)


def test_ordinals_survive_a_followup_parameter_answer():
    """问完「第二个的续航」之后，再问「第二个」仍应指向原推荐清单。"""
    memory = ConversationMemory()
    memory.add_ai("[推荐顺序] 比亚迪秦L EV｜吉利银河E5｜埃安AION Y Plus")
    memory.add_ai("吉利银河E5 的 CLTC 续航是 530 公里。")

    assert memory.last_recommended == ["比亚迪秦L EV", "吉利银河E5", "埃安AION Y Plus"]
    assert memory.resolve("第二个呢？") == "吉利银河E5呢？"
    assert memory.resolve("它的续航呢？") == "吉利银河E5的续航呢？"


def test_resolve_handles_pronoun_and_ordinal_in_one_sentence():
    memory = ConversationMemory()
    memory.add_ai("[推荐顺序] 比亚迪秦L EV｜吉利银河E5｜埃安AION Y Plus")
    assert memory.resolve("它和第二个相比，哪个更适合家用？") == (
        "比亚迪秦L EV和吉利银河E5相比，哪个更适合家用？"
    )


def test_memory_resolves_ordinal_to_second_recommendation():
    memory = ConversationMemory()
    memory.add_user("15万预算，家里有小孩")
    memory.add_ai("推荐这三款：\n1. 比亚迪元PLUS\n2. 吉利银河E5\n3. 比亚迪海豚")

    assert memory.last_recommended == ["比亚迪元PLUS", "吉利银河E5", "比亚迪海豚"]
    assert memory.resolve("第二个的续航怎么样？") == "吉利银河E5的续航怎么样？"
    assert "预算：15 万以内" in memory.profile_text()
    assert "有小孩" in memory.profile_text()


def test_memory_resolves_pronoun():
    memory = ConversationMemory()
    memory.add_ai("推荐 小鹏G6，性价比高。")
    assert memory.resolve("它的续航呢？") == "小鹏G6的续航呢？"


def test_missing_api_key_returns_friendly_message(monkeypatch):
    def boom(section: str = "llm"):
        raise RuntimeError("未配置环境变量 DASHSCOPE_API_KEY")

    monkeypatch.setattr(config, "require_api_key", boom)
    answer = EVAdvisor().chat("你好")
    assert answer.startswith("⚠️") and "DASHSCOPE_API_KEY" in answer


def test_full_turn_executes_tool_and_updates_memory(monkeypatch):
    called: list[str] = []

    def fake_specs(model_name: str) -> str:
        """按车型名查询参数。"""
        called.append(model_name)
        return "比亚迪 海豹 | 价格 17.98-24.98 万 | 续航 550 km"

    tool = StructuredTool.from_function(fake_specs, name="get_car_specs", description="按车型名查询参数")
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "get_car_specs", "args": {"model_name": "比亚迪海豹"}, "id": "call-1"}],
            ),
            AIMessage(content="比亚迪海豹的 CLTC 续航是 550 km，值得推荐。"),
        ]
    )
    advisor = EVAdvisor(llm=model, tools=[tool])
    monkeypatch.setattr(EVAdvisor, "classify", lambda self, text: Intent(intent="query_params"))

    answer = advisor.chat("比亚迪海豹续航多少？")

    assert called == ["比亚迪海豹"]
    assert "550" in answer
    # 单车型回答只更新「最近讨论的车型」，不覆盖推荐清单
    assert advisor.memory.last_model == "比亚迪海豹"
    assert advisor.memory.discussed == ["比亚迪海豹"]
    assert advisor.memory.turns == [("比亚迪海豹续航多少？", answer)]


def test_price_question_is_routed_to_calculator(monkeypatch):
    """回归：LLM 把「落地多少」误判成参数查询时，规则路由要纠正为 calculate。"""
    called: list[float] = []

    def fake_calc(price: float) -> str:
        """计算落地价。"""
        called.append(price)
        return "全款落地价：214,300 元"

    tool = StructuredTool.from_function(fake_calc, name="calc_landing_price", description="计算落地价")
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "calc_landing_price", "args": {"price": 20}, "id": "call-2"}],
            ),
            AIMessage(content="20 万裸车落地大约 21.43 万元。"),
        ]
    )
    advisor = EVAdvisor(llm=model, tools=[tool])
    monkeypatch.setattr(EVAdvisor, "classify", lambda self, text: Intent(intent="chat"))

    answer = advisor.chat("裸车20万落地多少钱")

    assert called == [20]
    assert "21.43" in answer


def test_trace_exposes_intent_and_tool_calls(monkeypatch):
    """界面上展示的决策路径要能看到意图、工具调用和工具返回。"""

    def fake_specs(model_name: str) -> str:
        """按车型名查询参数。"""
        return "比亚迪 海豹 | 续航 550 km"

    tool = StructuredTool.from_function(fake_specs, name="get_car_specs", description="按车型名查询参数")
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "get_car_specs", "args": {"model_name": "比亚迪海豹"}, "id": "t1"}],
            ),
            AIMessage(content="比亚迪海豹续航 550 km。"),
        ]
    )
    advisor = EVAdvisor(llm=model, tools=[tool])
    monkeypatch.setattr(
        EVAdvisor, "classify", lambda self, text: Intent(intent="query_params", models=["比亚迪海豹"])
    )

    answer, trace = advisor.chat_with_trace("比亚迪海豹续航多少？")

    assert "`query_params`" in trace         # 意图识别
    assert "车型 比亚迪海豹" in trace          # 实体抽取
    assert "get_car_specs" in trace          # 工具调用
    assert "550" in trace                    # 工具返回
    assert "总耗时" in trace
    assert "550" in answer


def test_stream_chat_pushes_trace_before_answer(monkeypatch):
    """流式输出：先到轨迹（意图/工具调用），答案逐帧增长，最后一帧是完整结果。"""

    def fake_specs(model_name: str) -> str:
        """按车型名查询参数。"""
        return "比亚迪 海豹 | 续航 550 km"

    tool = StructuredTool.from_function(fake_specs, name="get_car_specs", description="按车型名查询参数")
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "get_car_specs", "args": {"model_name": "比亚迪海豹"}, "id": "s1"}],
            ),
            AIMessage(content="比亚迪海豹续航 550 km。"),
        ]
    )
    advisor = EVAdvisor(llm=model, tools=[tool])
    monkeypatch.setattr(EVAdvisor, "classify", lambda self, text: Intent(intent="query_params"))

    chunks = list(advisor.stream_chat("比亚迪海豹续航多少？"))

    assert len(chunks) >= 3, "至少要有：意图轨迹、工具调用、最终答案三帧"
    assert chunks[0][0] == "", "第一帧只推轨迹，不应有答案"
    assert "`query_params`" in chunks[0][1]
    assert any("get_car_specs" in trace for _, trace in chunks)
    assert any("工具返回" in trace for _, trace in chunks)

    answers = [answer for answer, _ in chunks]
    assert answers == sorted(answers, key=len), "答案应逐帧累积增长"
    final_answer, final_trace = chunks[-1]
    assert "550" in final_answer
    assert "总耗时" in final_trace


def test_streaming_hides_recommendation_marker():
    from src.agent.memory import strip_streaming_marker

    full = "推荐 比亚迪元PLUS。\n[推荐顺序] 比亚迪元PLUS｜吉利银河E5"
    partial = "推荐 比亚迪元PLUS。\n[推荐顺"
    assert strip_streaming_marker(full) == "推荐 比亚迪元PLUS。"
    assert strip_streaming_marker(partial) == "推荐 比亚迪元PLUS。"
    assert strip_streaming_marker("推荐 比亚迪元PLUS。") == "推荐 比亚迪元PLUS。"
