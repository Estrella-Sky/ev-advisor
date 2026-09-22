---
domain:
  - nlp
tags:
  - 新能源汽车
  - 智能导购
  - 竞品对比
  - RAG
  - LangChain
  - Agent
license: MIT
---

# EV-Advisor：新能源汽车智能导购与竞品分析 Agent

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-1.x-1c3c3c)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange)
![License](https://img.shields.io/badge/License-MIT-green)

一个能聊、能查、能比、能算的选车助手：用 RAG 把 66 款热门新能源车型的参数变成可检索的知识库，
用 ReAct Agent 自主决定「什么时候查本地库、什么时候联网、什么时候算账」，并用多轮记忆记住你的预算和用车场景。

## 解决的痛点

- **信息碎片化**：参数、口碑、销量散落在不同页面，需要手动来回切换对比
- **缺乏个性化**：没有工具能按「预算 + 家庭人数 + 通勤距离」给出可解释的推荐
- **决策成本高**：CLTC 续航、磷酸铁锂与三元锂的差别，消费者需要有人翻译成大白话
- **信息滞后**：静态页面看不到最新的销量、降价与口碑变化

## 核心能力

| 能力 | 说明 | 关键实现 |
| --- | --- | --- |
| 意图识别与实体抽取 | 识别 6 类意图（推荐 / 参数 / 对比 / 行情 / 计算 / 闲聊），抽取预算、车型、关注点 | Qwen Function Calling（`with_structured_output`），失败降级为规则路由 |
| 参数知识库（RAG） | 66 款车型、30 个品牌的参数语义检索，支持按预算/续航/品牌过滤 | Qwen `text-embedding-v3` + ChromaDB 持久化 |
| 按需联网 | 销量、口碑、降价、新闻等时效信息实时检索，3 秒超时自动降级为本地数据 | DuckDuckGo（`ddgs`）+ 线程超时 |
| 竞品对比 | 输出 Markdown 对比表 + 购买建议，覆盖价格/续航/动力/空间/智驾/安全/用车成本 | ReAct 串联 `get_car_specs`、`web_search`、`calc_landing_price` |
| 多轮记忆 | 记住预算、家庭情况、偏好品牌、已讨论车型，支持「第二个」「它」这类指代 | `ConversationMemory` + 车型名最长匹配 |
| 落地价计算 | 购置税（含 2026-2027 新能源减半政策）、保险、上牌费、等额本息月供 | 纯 Python 计算工具，配置可调 |

## 架构

```
Gradio 界面（输入框 / 对话历史 / 示例问题 / 清空）
        │
        ▼
Agent Core ── Router 意图路由 ── Planner 执行路径 ── Executor（ReAct 循环）
        │                                              │
        │                                     ConversationBufferMemory
        ▼                                              │
  ┌─────────────┬──────────────┬──────────────────────┐
  │ RAG Tool    │ Search Tool  │ Calculator Tool      │
  │ ChromaDB    │ DuckDuckGo   │ 购置税/保险/月供      │
  └─────────────┴──────────────┴──────────────────────┘
```

执行流程：用户输入 → 指代消解 → 意图识别与实体抽取 → 组装 System Prompt（含用户画像与最近对话）
→ ReAct 循环（Thought → Action → Observation）→ 生成推荐/对比/参数回答 → 写回记忆。

## 快速开始

### 1. 准备环境

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env      # 填入 DASHSCOPE_API_KEY（阿里云百炼控制台获取）
```

Qwen LLM 与 Embedding 共用同一个 Key，百炼提供免费额度，全流程零成本。

### 3. 构建向量库

```bash
python -m src.rag.vector_store            # 首次构建（已存在则跳过）
python -m src.rag.vector_store --force    # 数据更新后重建
```

不执行这一步也能启动应用：首次提问时会自动构建并落盘到 `data/processed/embeddings/`。

### 4. 启动

```bash
python app.py            # 打开 http://127.0.0.1:7860
```

### 5. Docker 一键部署

```bash
docker compose up --build     # 同样访问 http://127.0.0.1:7860
```

镜像也发布在 GHCR：`docker run -p 7860:7860 -e DASHSCOPE_API_KEY=sk-xxx ghcr.io/<owner>/<repo>:latest`

### 6. 在线演示（魔搭创空间）

在线地址：<https://www.modelscope.cn/studios/EstrellaSky/ev-advisor>

> 为什么不用 Hugging Face Spaces？HF 现行政策下免费账号只能创建 Static Space，
> 运行 Gradio / Docker Space 需要 PRO 订阅；魔搭创空间提供免费 CPU 档
> （2 核 8G，支持 Gradio），本项目因此选择魔搭作为在线演示平台。

重新部署到自己的魔搭账号：

1. 在 [modelscope.cn](https://modelscope.cn) 新建创空间：SDK 选 Gradio，入口文件保持默认 `app.py`
2. 在创空间「设置 → 环境变量」里添加 Secret：`DASHSCOPE_API_KEY`
3. 推送代码（创空间是独立 git 仓库，默认分支为 `master`）：

   ```bash
   git remote add modelscope https://www.modelscope.cn/studios/<用户名>/<空间名>.git
   git push -f modelscope main:master
   ```

4. 平台自动构建，几分钟后即可通过 `https://www.modelscope.cn/studios/<用户名>/<空间名>` 访问

## 自测与验收

```bash
pytest -q
```

测试不依赖 API Key（检索链路用确定性假向量验证），覆盖需求文档 §7.1 的用例：

| 用例 | 场景 | 对应测试 |
| --- | --- | --- |
| TC-001 | 意图识别-推荐 / 实体抽取 | `tests/test_agent.py::test_rules_intent_*` |
| TC-002 | 意图识别-对比 | `tests/test_agent.py::test_rules_intent_covers_six_types` |
| TC-003 | RAG 参数查询与精确命中 | `tests/test_rag.py::test_get_car_exact_match_and_unknown_model` |
| TC-004 | 联网搜索（含失败降级） | `tests/test_tools.py::test_web_search_*` |
| TC-005 | 多轮记忆与指代消解 | `tests/test_agent.py::test_memory_resolves_*` |
| TC-006 | 搜索超时优雅降级 | `tests/test_tools.py::test_web_search_times_out_fast` |
| TC-007 | 查询不存在的车型 | `tests/test_rag.py::test_get_car_exact_match_and_unknown_model` |
| TC-008 | 落地价计算准确性 | `tests/test_tools.py::test_purchase_tax_rule_per_period`、`test_landing_price_matches_manual_calculation` |

## 目录结构

```
EV-Advisor/
├── app.py                        # 入口（模块级 demo 变量，供创空间/Spaces 读取）
├── config/
│   ├── config.yaml               # 全局配置：模型、检索、超时、计算口径
│   └── prompts.py                # System Prompt 模板
├── data/raw/ev_cars.csv          # 车型知识库（66 款）
├── src/
│   ├── agent/{core,memory}.py    # 意图路由 + ReAct 执行 + 多轮记忆
│   ├── rag/{embedder,vector_store,retriever}.py
│   ├── tools/{rag_tool,search_tool,calculator_tool}.py
│   ├── data_processing/cleaner.py
│   └── ui/gradio_app.py
├── tests/                        # pytest 用例
├── Dockerfile / docker-compose.yml
└── .github/workflows/docker-publish.yml
```

`src/data_processing/scraper.py`（需求文档标注为「可选」）未实现：车型数据改为一次性手工整理的
CSV，避免引入不稳定的爬虫与合规风险。

## 配置说明（`config/config.yaml`）

| 配置节 | 关键项 | 说明 |
| --- | --- | --- |
| `llm` | `model: qwen-plus` | 可用 `EV_MODEL` 环境变量覆盖 |
| `embedding` | `model: text-embedding-v3` | `batch_size: 10`，规避单次批量上限 |
| `rag` | `top_k`、`persist_dir` | 检索条数与向量库落盘位置 |
| `search` | `timeout: 3` | 联网搜索超时（秒），超时即降级 |
| `agent` | `max_iterations`、`memory_window` | ReAct 最大轮数与注入的对话轮数 |
| `calculator` | 保险比例、上牌费、首付、期数、利率 | 落地价口径，按当地行情调整 |

## 数据说明

- 数据为手工整理的公开资料参考值（厂商指导价口径，2025-2026 款为主），用于学习与演示；
  正式使用请以厂商官网与经销商报价为准。
- 增程 / 插混车型（理想 L 系列、问界 M 系列）的 `range_cltc` 填的是 **CLTC 综合续航**，
  括号内注明纯电续航；纯电车型填纯电续航。
- 数据版权归原始来源所有，本项目仅用于学习，不做商业用途。

## 已知限制

- 车型价格与配置随市场变化，知识库需要手动更新后重建向量库
- 联网搜索依赖 DuckDuckGo 的可用性，失败时只能给本地数据（已做降级提示）
- 演示场景按 1-3 并发设计，未做多用户隔离，会话记忆保存在进程内

## 部署清单

| 步骤 | 操作 | 状态 |
| --- | --- | --- |
| 1 | GitHub 创建仓库并推送代码 | ☐ |
| 2 | 配置 `.env.example` 与 `.gitignore` | ☑ |
| 3 | 编写魔搭创空间所需的 README YAML 头 | ☑ |
| 4 | 创建创空间并推送代码 | ☑ |
| 5 | 创空间配置 `DASHSCOPE_API_KEY` 环境变量 | ☑ |
| 6 | 编写 GitHub Actions 工作流（测试 + 推送 GHCR） | ☑ |
| 7 | 触发 Actions 构建镜像至 GHCR | ☐ |
| 8 | 验证在线 Demo 与镜像可访问 | ☑ |
| 9 | 录制演示 GIF 并更新 README | ☐ |

## License

MIT
