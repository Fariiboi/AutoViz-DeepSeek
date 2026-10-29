# AutoViz V3.0

**基于多智能体架构的自动化数据分析平台**

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white">
  <img alt="Pydantic-AI" src="https://img.shields.io/badge/Agent-Pydantic--AI-E92063">
  <img alt="Modal" src="https://img.shields.io/badge/Sandbox-Modal-1F2937">
  <img alt="pgvector" src="https://img.shields.io/badge/RAG-pgvector%20%2B%20CrossEncoder-336791?logo=postgresql&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

> AutoViz 是一套基于多智能体架构的自动化数据分析平台。只需上传 CSV，系统即可自动完成特征工程、图表规划、代码生成、云端沙箱隔离渲染及双语报告生成。

无需编写一行绘图代码：把 CSV 交给它，它会读懂数据结构，自行规划该画什么图，在隔离沙箱里写代码、跑代码、自我审查，最后交付一套中文图表与中英双语商业报告。

---

## 核心特性

| 能力 | 实现 |
|---|---|
| **Pydantic-AI 多智能体架构** | Planner / Coder / Critic / Annotator / Selector / Reporter / Summarizer / Translator 等 Agent 各司其职，基于 `pydantic-ai` 的类型化输出协作，主 Agent 拆解任务、子 Agent 并发执行 |
| **Modal 云端沙箱** | 所有 AI 生成的 Python/matplotlib 代码都在 Modal 远程容器中执行，本地永不 `exec()`；镜像预装依赖与 CJK 字体，产物以 PNG 字节流回传 |
| **pgvector + CrossEncoder 两阶段 RAG** | 本地 BGE 向量化历史图表标注 → pgvector(HNSW) 召回 Top-20 → `bge-reranker-base` CrossEncoder 精排 Top-K，重排失败自动降级为纯向量检索 |
| **中文字体修复** | 沙箱内 glob 扫描 Noto CJK / 文泉驿字体并 `addfont()` 动态注册，猴补 `plt.style.use`、`sns.set_theme` 与 `rcParams` 覆写，彻底解决中文乱码与 `axes.unicode_minus` 豆腐块 |
| **Streamlit 前端** | 上传 CSV → 实时进度/剩余时间 → 图表画廊与报告渲染；内置「智能历史洞察搜索」页，可对整批结果生成商业洞察并语义检索历史图表 |

**其他工程细节**

- **纯文本 Critic / Annotator**：不依赖视觉模型。先把图表压缩成确定性文本摘要（任务 + 数据画像 + 从代码中静态解析出的坐标轴、刻度、分组、聚合方式），再交给 `deepseek-chat` 审核与标注，成本低且可解释。
- **编码自适应**：CSV 读取按 UTF-8(BOM) → GB18030 依次尝试，Excel 导出的中文表格直接可用；超过 2000 行自动随机采样分析。
- **免 API 的向量化**：嵌入模型跑在本地 CPU，不产生额外 embedding 调用费用。
- **双语交付**：英文报告 + 自动翻译的中文商业分析报告 `report_chn.md`。

---

## 工作原理

```text
CSV 文件
   │
   ▼
1. CSV 预处理       解析列类型 / 缺失值 / 统计摘要（编码自适应 + 大文件采样）
2. 特征工程          Feature Engineer Agent 决策衍生特征，在 Modal 沙箱中物化
3. 可视化规划        Planner Agent 产出图表任务列表（类型 / 列 / 意图）
4. 并发绘图          Coder 生成代码 → Modal 沙箱渲染 → Critic 审核(可回退重写) → Annotator 标注
5. 报告生成          Reporter 输出 Markdown/JSON 报告 → Translator 生成中文版
6. 语义入库（可选）   标注向量化写入 pgvector，供后续检索与洞察复用
```

---

## 环境依赖

| 依赖 | 说明 |
|---|---|
| Python ≥ 3.12 | |
| [uv](https://docs.astral.sh/uv/) | 包管理 / 虚拟环境 |
| [DeepSeek](https://platform.deepseek.com) API Key | 默认 `deepseek-chat`（纯文本模型，全流程够用） |
| [Modal](https://modal.com) 账号 | 代码沙箱，有免费额度 |
| [PostgreSQL 16 + pgvector](https://github.com/pgvector/pgvector) | 可选：历史图表语义检索（Docker 一键启动） |

---

## 快速开始

```bash
# 1. 克隆
git clone https://github.com/Fariiboi/AutoViz-DeepSeek.git
cd AutoViz-DeepSeek

# 2. 安装依赖（uv 自动创建 .venv）
uv sync

# 3. 生成配置模板并填入自己的密钥
cp .env.example .env        # Windows PowerShell: Copy-Item .env.example .env
```

`.env` 最少需要三项：

```env
DEEPSEEK_API_KEY=sk-...        # https://platform.deepseek.com/api_keys
MODAL_TOKEN_ID=ak-...          # modal.com 注册后执行 `uv run modal token new`
MODAL_TOKEN_SECRET=as-...
AI_MODEL=deepseek-chat
```

> ⚠️ `.env` 已被 `.gitignore` 永久忽略。**请勿把真实密钥提交到仓库**，也不要在 Issue / 截图里暴露。

> **Modal 首次使用**：`uv run modal token new` 会引导登录并把 Token 写入 `.env`。

> **TLS 证书报错（代理/抓包软件）**：若出现 `CERTIFICATE_VERIFY_FAILED`，可在 `.env` 设 `DISABLE_SSL_VERIFY=true`。该开关会关闭 DeepSeek(httpx) 与 Modal(grpclib) 的证书校验，**仅限本地开发**。

---

## 使用方式

### CLI

```bash
# 基础：分析 CSV，生成 10 张图
uv run autoviz analyze run data.csv

# 指定输出目录与图表数量，并生成报告
uv run autoviz analyze run data.csv -o ./results -n 5 --report

# 跳过特征工程（更快）
uv run autoviz analyze run data.csv --no-feature-eng

# JSON 报告
uv run autoviz analyze run data.csv --report --format json

# 对既有结果单独出报告
uv run autoviz analyze report --from ./autoviz-output/data/20260419_143022

uv run autoviz --help
```

### Streamlit 前端

```bash
uv run streamlit run app.py     # http://localhost:8501
```

左侧选择图表数量、是否启用 AI 特征工程 → 上传 CSV → 查看实时进度 → 完成后直接浏览中文图表与 `report.md`。

### 语义检索（可选，PostgreSQL + pgvector）

```bash
docker compose up -d                                  # pgvector/pgvector:pg16 → 5433
# .env: DATABASE_URL=postgresql://autoviz:autoviz@127.0.0.1:5433/autoviz

uv run autoviz db init                                # 建表 + HNSW 索引
uv run autoviz analyze run data.csv -n 5 --report     # 自动入库
uv run autoviz db ingest autoviz-output/data/<ts>     # 或回填历史目录
uv run autoviz search "价格和年龄的关系" -k 5
uv run autoviz db prune                               # 清理死链
uv run autoviz db stats
```

检索链路：pgvector 召回 `RAG_RECALL_K`(默认 20) 个候选 → `BAAI/bge-reranker-base` 精排 → Top-K；重排异常时自动降级。

> 首次检索会下载约 100 MB 的 `bge-small-zh-v1.5` 与约 1.1 GB 的 `bge-reranker-base` 到本地缓存；国内可设 `HF_ENDPOINT=https://hf-mirror.com`。

无 Docker 的 Windows 机器可使用仓库自带的便携版 PostgreSQL 16 + pgvector（端口 5433）：双击 `start_db.bat` / `stop_db.bat`，数据目录为 `workspace\pgdata`（该目录不入库）。

---

## 输出结构

```text
autoviz-output/
└── <csv-name>/
    └── <timestamp>/
        ├── charts/                 # PNG 图表
        │   ├── chart_001.png
        │   └── ...
        ├── csv_summary.json        # 列统计摘要
        ├── charts_metadata.json    # 图表元信息（report 命令复用）
        ├── report.md               # 英文分析报告
        └── report_chn.md           # 中文商业分析报告
```

---

## 项目结构

```text
autoviz/
├── app.py                        # Streamlit 界面
├── autoviz/
│   ├── cli.py                    # CLI 入口 + 流水线编排
│   ├── config.py                 # Pydantic Settings（读取 .env）
│   ├── events.py                 # VizResult / 消息总线
│   ├── agents/                   # 多智能体
│   │   ├── planner.py            # 图表规划
│   │   ├── coder.py              # matplotlib 代码生成
│   │   ├── critic.py             # 质量审核（纯文本摘要）
│   │   ├── annotator.py          # 图表洞察标注
│   │   ├── selector.py           # 可行性与信息增益筛选
│   │   ├── feature_engineer.py   # 特征工程决策
│   │   ├── reporter.py           # 报告生成
│   │   ├── summarizer.py         # 批次级商业洞察
│   │   ├── translator.py         # 英文报告 → 中文
│   │   ├── chart_digest.py       # 图表文本摘要（代码静态解析）
│   │   ├── context.py            # 模型 / Key / 语言上下文
│   │   └── prompts.py            # Prompt 模板
│   ├── pipeline/
│   │   ├── csv.py                # CSV 预处理（编码/采样/画像）
│   │   ├── features.py           # 特征工程流水线
│   │   ├── charts.py             # 可视化编排
│   │   ├── sandbox.py            # Modal 沙箱 + 中文字体修复
│   │   └── runner.py             # 子 Agent 并发管理
│   └── store/                    # pgvector 语义层
│       ├── embeddings.py         # 本地 BGE 封装
│       ├── schema.py             # DDL + HNSW
│       ├── repository.py         # 检索 / 入库 / 死链清理
│       ├── rerank.py             # CrossEncoder 精排
│       └── persist.py            # 结果入库
├── pages/1_智能历史洞察搜索.py     # 批次洞察 + 语义检索页
├── tests/                        # 离线回归 + pgvector 集成测试
├── docker-compose.yml
├── start_db.bat / stop_db.bat    # Windows 便携数据库
└── pyproject.toml
```

---

## 测试

```bash
uv run --extra dev pytest -q
uv run --extra dev ruff check autoviz tests pages
```

---

## 安全与合规

- 仓库**不包含任何凭证**：`.env` / `.env.*` 全部忽略，仅保留占位符模板 `.env.example`。
- 仓库**不包含任何数据集**：`*.csv` 及常见表格/二进制数据格式全局忽略，请自备数据（如 `sample/` 下无示例文件，可自行放置）。
- `workspace/`、`autoviz-output/`、`.venv/`、`__pycache__/`、`node_modules/` 均被忽略。

---

## License

MIT

---

## 🙏 致谢 (Acknowledgements)

本项目基于原作者 feiyiLB 的优秀开源项目 AutoViz 进行企业级深度重构与二次开发。特别感谢原作者提供的多 Agent 基础架构与图表生成思路。
