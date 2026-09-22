"""应用入口：本地运行与 Hugging Face Spaces 启动文件（模块级 demo 变量）。"""
from __future__ import annotations

import config
from src.ui.gradio_app import build_demo

config.setup_logging()

demo = build_demo()

if __name__ == "__main__":
    ui = config.get_config()["ui"]
    demo.queue().launch(server_name=ui["server_name"], server_port=ui["server_port"])

