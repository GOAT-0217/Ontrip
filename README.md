# `Ontrip-main`

---

## 📋 项目概述

本项目实现了一个面向客户支持场景的**多智能体检索增强生成系统**，并针对旅行场景（如 ✈️ 瑞士航空公司）进行了专项适配。系统采用 Python 语言开发，核心依赖于 **LangChain** 和 **LangGraph** 等关键库。

> 💡 **核心理念**  
> 设立一个主控助手来处理通用查询，并将更为复杂、专业化的任务（例如预订机票、车辆、酒店或短途旅行）路由分发至专属的子助手执行。此种架构设计提升了系统的模块化水平与可扩展能力，同时实现了细粒度的流程控制，尤其针对那些必须经过用户确认方可执行的敏感操作。

---

### 🌟 核心功能特性

| 图标 | 功能模块 | 说明 |
|:---:|:---|:---|
| 🕸️ | **多智能体架构** | 利用 LangGraph 定义具有状态的执行工作流，该工作流包含一个主控助手及多个专属子助手 |
| 🔍 | **RAG 集成** | 采用 Qdrant 向量数据库高效检索相关信息（如航班详情、政策条款），以此增强大语言模型的回答质量 |
| 🛡️ | **安全机制** | 内置"敏感工具"拦截功能，在工作流执行涉及敏感操作前暂停进程，并强制要求获取用户明确批准 |
| 👁️ | **可观测性** | 集成 LangSmith 工具，用于追踪与监控各智能体间的交互轨迹 |
| 🧩 | **模块化设计** | 代码遵循清晰的模块划分原则（`customer_support_chat`、`vectorizer`），以实现明确的关注点分离 |

---

## 📁 项目结构

```
D:\Ontrip-main\
├── 📄.dev.env                # 环境变量模板文件
├── 📄.gitignore
├── 🐳 docker-compose.yml # 定义服务（如 Qdrant）
├── 🐳 Dockerfile
├── 🔧 Makefile # 定义常用项目命令
├── 🔒 poetry.lock # Poetry依赖锁定文件
├── ⚙️ pyproject.toml # 项目元数据及依赖配置（Poetry）
├── 📖 README.md # 主项目文档
├── 📁.vscode\                # VS Code 配置文件
│   └── 📄 launch.json
├── 📁 customer_support_chat\ # 💬 主应用模块
│   ├── 📖 README.md # 模块详细文档
│   ├── 📄 init.py
│   ├── 📁 app\ # 核心应用逻辑
│   │   ├── 📄 init.py
│   │   ├── 📁 core\ # 核心组件（状态、配置、日志记录器）
│   │   ├── 📁 data\ # （可能存在的）本地数据文件
│   │   ├── 🕸️ graph.py # 定义 LangGraph 工作流
│   │   ├── 🚀 main.py # 聊天应用的主入口点
│   │   └── 📁 services\ # 助手、工具、实用程序、向量数据库接口
│   └── 📁 data\ # 本地 SQLite 数据库文件 (travel2.sqlite)
├── 📁graphs\                 # 📊 图可视化输出目录
│   └── 🖼️ multi-agent-rag-system-graph.png
├── 📁images\                 # 🖼️ 文档图片资源
└── 📁vectorizer\             # 🧬 用于生成嵌入向量并填充 Qdrant 的模块
    ├── 📖README.md           # 模块详细文档
    ├── 📄__init__.py
    └── 📁 app\                # 核心向量化逻辑
        ├── 📄__init__.py
        ├── 📁core\           # 核心组件（日志记录器、配置）
        ├── 📁embeddings\     # 嵌入向量生成逻辑
        ├── 🚀main.py         # 向量化过程的入口点
        └── 📁vectordb\       # Qdrant 数据库交互逻辑
```

---

## 🛠️ 主要技术栈

| 类别 | 技术选型 | 说明 |
|:---|:---|:---|
| 🐍 **编程语言** | Python 3.12+ | — |
| 📦 **包管理器** | Poetry | — |
| 🧠 **核心框架/库** | `langgraph` | 用于构建多智能体状态机/图 |
| | `langchain` | 用于大语言模型交互、提示词管理与工具调用 |
| | `langchain-openai` | 用于集成 OpenAI 大语言模型及嵌入模型 |
| | `qdrant-client` | 用于与 Qdrant 向量数据库进行交互 |
| 🗄️ **向量数据库** | Qdrant | 可通过 Docker 在本地运行 |
| 💾 **数据源** | SQLite (`travel2.sqlite`) | 内含旅行相关数据 |
| 🔭 **可观测性** | LangSmith | （可选） |
| 🌍 **环境管理** | `python-dotenv` | — |

---

## 🚀 构建与运行指南

### 📌 前置条件

- ✅ Python 3.12 或更高版本
- ✅ Poetry
- ✅ Docker 与 Docker Compose
- ✅ OpenAI API 密钥
- ✅ LangSmith API 密钥（可选）

---

### 🔧 环境配置与执行步骤
1️⃣ **环境变量设置**
* 复制环境模板文件: `cp .dev.env .env`
* 编辑 `.env`文件,填入您的密钥 `OPENAI_API_KEY` 和 `LANGCHAIN_API_KEY`(可选).

2️⃣ **安装依赖包**
* 运行 `poetry install` 安装 `pyproject.toml`中定义的全部Python依赖.

3️⃣ **准备向量数据库（Qdrant）**
* 后台启动Qdrant服务: `docker compose up qdrant -d`
* *（可选）* 通过浏览器访问 Qdrant 控制台界面：http://localhost:6333/dashboard#

4️⃣ **生成并存储嵌入向量**
* 🧬 此步骤将处理数据并将向量填充至 Qdrant 数据库: `poetry run python vectorizer/app/main.py`

5️⃣ **运行客户支持聊天应用**
* 💬 启动主聊天应用: `poetry run python ./customer_support_chat/app/main.py`
* ⌨️通过命令行界面与聊天机器人进行交互. 输入 `quit`, `exit`, 或者 `q` 即可停止程序.

## 📐 开发规范

*   **📦 依赖管理:** 依赖项通过 Poetry 进行管理 (`pyproject.toml`, `poetry.lock`).
*   **🧩 模块化:** 代码拆分为 `vectorizer` 和 `customer_support_chat` 两个独立模块. 各模块均包含自身的 `README.md`说明文档.
*   **💾 状态管理:** 采用 `langgraph.checkpoint.memory.MemorySaver` 实现对话轮次间的内存级状态持久化.
*   **🕸️ 图定义:** 对话流程在 `customer_support_chat/app/graph.py` 文件中，通过 LangGraph 的 `StateGraph`进行定义.
*   **🤖 助手类:** 各专属子助手均继承自 `customer_support_chat/app/services/assistants/assistant_base.py`中定义的基类.
*   **🔧 工具集:** 大语言模型可调用的函数工具定义在 `customer_support_chat/app/services/tools/`目录下.
*   **⚙️ 配置管理:** 应用设置通过 `core/settings.py`中的 Pydantic 模型进行管理，相关参数从环境变量中加载.
*   **📝 日志记录:** 采用 `core/logger.py`中配置的自定义日志记录器.

## 🧠 Qwen 补充记忆
* ⚠️ 关于 Markdown 换行符的正确处理
* * 在生成包含多行文本的 Markdown 文件时，如果直接在 write_file 的 content 参数中使用字面量 \n 字符串，将导致文件中出现字面量字符 \n，而非实际的换行符。
* ✅ 正确做法
* * 在 Python 字符串内使用真实的换行符（例如，使用三重引号 ''' 或 """ 来包裹多行字符串，或者在字符串中使用 \n 转义序列并确保其在写入时被正确解释为换行符）。

* 🔧 补救措施
* * 当需要从一个包含字面量 \n 的文件创建格式正确的文件时，必须借助脚本（如 PowerShell 或 Python）来替换这些字面量字符。

* 📋 需求变更记录
| 需求项 | 详细说明 | 状态 |
|:---|:---|:---|
| 👥 多用户支持 | 允许多个用户同时使用系统 | ⏳ 待实现 |
| 💬 聊天记录保存 | 持久化存储用户的对话历史 | ⏳ 待实现 |
| 🌐 Web 聊天界面 | 提供 HTML 形式的用户聊天界面 | ⏳ 待实现 |
