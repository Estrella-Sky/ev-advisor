"""Gradio 界面：左侧对话，右侧实时展示 Agent 决策路径与会话记忆。

需求文档 §3.1 的 Router → Planner → Executor 流程对用户是黑盒，
这里把意图识别、实体抽取、工具调用与返回结果直接铺在界面上。
"""
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


PENDING_HINT = "_提问后，这里会显示本轮 Agent 的完整决策路径。_"
PROFILE_HINT = "_还没有捕捉到预算 / 家庭情况 / 偏好品牌。_"


def build_demo() -> gr.Blocks:
    """构建 Gradio 应用（app.py 暴露为模块级 demo，创空间与 Spaces 都读它）。"""
    advisor = EVAdvisor()
    cfg = config.get_config()["ui"]

    def respond(message: str, history: list):
        if not message or not str(message).strip():
            return history, PENDING_HINT, _profile_text(advisor), ""
        if not history:  # 新会话（首次提问或点了清空）
            advisor.reset()
        answer, trace = advisor.chat_with_trace(message)
        history = list(history) + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ]
        return history, trace, _profile_text(advisor), ""

    def clear():
        advisor.reset()
        return [], PENDING_HINT, PROFILE_HINT, ""

    with gr.Blocks(title=cfg["title"]) as demo:
        gr.Markdown(f"# {cfg['title']}")
        gr.Markdown(_description())
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(type="messages", height=520, label="对话历史")
                textbox = gr.Textbox(
                    placeholder="例如：20 万预算，通勤 40 公里，有老人，推荐什么车？",
                    label="你的问题",
                    lines=1,
                )
                with gr.Row():
                    send_btn = gr.Button("发送", variant="primary")
                    clear_btn = gr.Button("清空对话")
                gr.Examples(examples=EXAMPLES, inputs=textbox, label="示例问题（点击填入）")
            with gr.Column(scale=2):
                gr.Markdown("### 🧠 Agent 决策路径")
                gr.Markdown("意图识别 → 实体抽取 → 工具调用 → 观察结果，逐步展开。")
                trace_md = gr.Markdown(PENDING_HINT)
                gr.Markdown("### 📋 会话记忆")
                gr.Markdown("多轮对话中累积的预算、家庭情况、偏好品牌与已讨论车型。")
                profile_md = gr.Markdown(PROFILE_HINT)

        outputs = [chatbot, trace_md, profile_md, textbox]
        textbox.submit(respond, inputs=[textbox, chatbot], outputs=outputs)
        send_btn.click(respond, inputs=[textbox, chatbot], outputs=outputs)
        clear_btn.click(clear, outputs=outputs)
    return demo


def _profile_text(advisor: EVAdvisor) -> str:
    """会话记忆面板：没有内容时给占位提示。"""
    return advisor.memory.profile_text().replace("## 已知用户信息\n", "") or PROFILE_HINT
