"""Gradio 界面：聊天输入框 + 对话历史 + 示例问题 + 清空。"""
from __future__ import annotations

import logging

import gradio as gr

import config
from src.agent.core import EVAdvisor
from src.data_processing.cleaner import distinct_brands, load_clean_rows

logger = logging.getLogger(__name__)

EXAMPLES = [
    "15 万预算，家里有小孩，推荐什么车？",
    "比亚迪海豹的电池容量和续航是多少？",
    "对比一下 Model Y 和理想 L6",
    "小米 SU7 最新月销量是多少？",
    "裸车 20 万，落地大概多少钱？",
    "它和第二个相比，哪个更适合家用？",
]


def _description() -> str:
    rows = load_clean_rows()
    brands = len(distinct_brands())
    return (
        f"本地知识库已收录 **{len(rows)} 款**新能源车型、覆盖 **{brands} 个**品牌，"
        "支持参数查询、按预算推荐、多车型对比、实时信息检索与落地价计算。\n\n"
        "试试下面的示例问题，或者直接描述你的预算和用车场景。"
    )


def build_demo() -> gr.ChatInterface:
    """构建 Gradio 应用（Hugging Face Spaces 读取 app.py 中的 demo 变量）。"""
    advisor = EVAdvisor()

    def respond(message: str, history: list) -> str:
        if not history:  # 新会话（首次提问或点了清空）
            advisor.reset()
        return advisor.chat(message)

    cfg = config.get_config()["ui"]
    return gr.ChatInterface(
        fn=respond,
        type="messages",
        title=cfg["title"],
        description=_description(),
        examples=EXAMPLES,
        chatbot=gr.Chatbot(height=520, type="messages", label="对话历史"),
        textbox=gr.Textbox(placeholder="例如：20 万预算，通勤 40 公里，有老人，推荐什么车？", label="你的问题"),
        cache_examples=False,
    )
