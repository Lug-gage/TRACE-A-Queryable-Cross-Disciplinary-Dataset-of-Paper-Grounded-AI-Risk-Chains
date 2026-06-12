# HEVI — AI Paper Risk Assessment Pipeline

基于 **HEVI 框架**（Hazard-Exposure-Vulnerability-Impact）自动评估 AI/ML 论文中潜在社会风险的流水线。CS Agent 与 SS Agent 通过**双边共识协议**协同分析，将论文的 impact statement 转化为可追溯的因果风险链。

## 核心概念

HEVI 框架改编自 Turner et al. (2003) 的脆弱性分析理论，定义 **6 个风险槽位**：

| 槽位 | 含义 | 归属 |
|------|------|------|
| Hazard | 论文方法引入或放大的技术危害 | CS Agent |
| Exposure | 谁/哪些系统面临该危害（具体场景） | CS Agent |
| Dose-Response | 危害规模如何转化为影响程度 | 共识合成 |
| Vulnerability | 系统对危害的敏感条件 | SS Agent |
| Impact | 负面社会/伦理/经济后果 | SS Agent |
| Key Control Nodes | 可阻断风险链的干预点 | SS Agent |

详细定义见 `hipporag/hevi_workflow/hevi_framework.py`。

## 项目结构

```
hevi_package/
├── hevi_run.py                 # 统一入口 CLI
├── api_key.txt                 # OpenAI API Key（一行纯文本）
├── requirements.txt
├── data/
│   └── icml_corpus_with_len.csv   # ICML 论文语料（需自行准备）
├── indices/                    # 预构建检索索引
│   ├── cs/                     #   CS 文献索引
│   └── ss/                     #   SS 文献索引
├── outputs/                    # 所有产出
│   ├── hevi_workflow/
│   │   ├── hevi_icml_{model}/  # Stage 1+2：提取 & 审计结果
│   │   │   └── group_*/        #   并行跑时按 group 分目录
│   │   ├── hevi_{model}/       # Stage 3-5：流水线每条论文的结果
│   │   │   └── group_*/icml_*/
│   │   │       ├── 1_reference.json
│   │   │       ├── 2_cs_proposal.json
│   │   │       ├── 3_ss_response.json
│   │   │       ├── 4_consensus.json
│   │   │       └── 5_compare.json
│   │   └── hevi_comparison.csv # 导出对比表
│   └── dataset.json            # 汇总数据集
├── hipporag/                   # HippoRAG 引擎 + HEVI workflow
│   ├── hevi_workflow/
│   │   ├── agents.py           #  RiskLLM, CSAgent, SSAgent
│   │   ├── pipeline.py         #  双边共识流水线核心逻辑
│   │   ├── retrievers.py       #  CS/SS 文献检索
│   │   ├── hevi_compare.py     #  HEVI 对比评测
│   │   ├── hevi_framework.py   #  框架定义
│   │   ├── evaluator.py        #  质量审计
│   │   └── utils.py            #  JSON 修复、API Key 加载等
│   └── ...                     #  HippoRAG 引擎（embedding, LLM, 检索）
└── scripts/                    # CLI 子命令脚本
    ├── extract_reference_hevi.py  # Stage 1：提取 reference HEVI
    ├── audit_hevi_quality.py      # Stage 2：质量审计
    ├── run_hevi_pipeline.py       # Stage 3-5：流水线
    ├── export_hevi_csv.py         # 导出对比 CSV
    ├── build_dataset.py           # 汇总 dataset JSON
    └── summarize_hevi_compare.py  # 汇总统计数据
```

## 环境配置

### 1. 创建 conda 环境

```bash
conda create -n hipporag python=3.11
conda activate hipporag
pip install -r requirements.txt
```

### 2. 配置 API Key

在项目根目录创建 `api_key.txt`，写入 OpenAI 兼容 API 的 key：

```
sk-your-key-here
```

或直接设置环境变量：

```bash
export OPENAI_API_KEY="sk-your-key-here"
```

## 运行方式

**所有命令必须在项目根目录执行**（或通过 `hevi_run.py` 入口，它会自动切换工作目录）。

### 一键全流程

```bash
python hevi_run.py all --target 10 --top-k-cs 5 --top-k-ss 5
```

自动串行执行：提取 → 审计 → 流水线 → 可视化，攒够 `--target` 篇审计通过的论文（verdict=keep）后触发流水线。

### 分步执行

```bash
# Stage 1：从论文中提取 reference HEVI 风险描述
python hevi_run.py extract

# Stage 2：对提取结果做质量审计（打分 → keep/reject）
python hevi_run.py audit

# Stage 3-5：运行 CS/SS 双边共识流水线
python hevi_run.py run

# 导出 ref_hevi vs workflow 对比 CSV
python hevi_run.py export
```

### 直接调用脚本（更多参数）

```bash
# 流水线——按 group 跑（每个 group ≈ 70 篇，适合并行）
python scripts/run_hevi_pipeline.py --group 1
python scripts/run_hevi_pipeline.py --group 2
# ... group 1-6

# 指定论文
python scripts/run_hevi_pipeline.py --paper-ids "icml_2024_0001,icml_2024_0004"

# 汇总数据集（默认 ≥3 HEVI slots，筛掉缺 hazard/exposure 的 chain）
python scripts/build_dataset.py
python scripts/build_dataset.py --group 1          # 单 group
python scripts/build_dataset.py --min-slots 2      # 放宽 slot 阈值

# 导出对比 CSV
python scripts/export_hevi_csv.py
python scripts/export_hevi_csv.py --group 1
```

### 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--llm-name` | `deepseek-v4-pro` | LLM 模型名 |
| `--llm-base-url` | `https://www.highland-api.top/v1` | OpenAI 兼容 API 地址 |
| `--embedding-name` | `text-embedding-3-large` | Embedding 模型 |
| `--top-k-cs` | 5 | CS 检索返回数量 |
| `--top-k-ss` | 5 | SS 检索返回数量 |
| `--theta` | 0.8 | 双边共识收敛阈值 |
| `--max-consensus-rounds` | 0 | 共识最大轮数（0 = 跳过 critique 轮，仅 DR 合成） |
| `--min-slots` | 3（build_dataset） | 至少 N 个非空 HEVI slot 才入选 |
| `--no-resume` | — | 禁掉 resume，从头跑全部论文 |

## 流水线五个阶段

```
Stage 1 ─ REFERENCE ─ 加载预提取的 ref_hevi（6 个风险槽位）
Stage 2 ─ CS PROPOSE ─ CS Agent 检索文献 → 提案 Hazard→Exposure 段
Stage 3 ─ SS RESPOND ─ SS Agent 检索文献 → 响应 Vulnerability→Impact 段
Stage 4 ─ CONSENSUS ─ 双边互评修订 → Dose-Response 合成因果链
Stage 5 ─ COMPARE ─ 流水线 HEVI vs reference HEVI 对比评估
```

## 数据格式

### 输入：`data/icml_corpus_with_len.csv`

CSV 文件，每行一篇论文，关键字段：

| 字段 | 说明 |
|------|------|
| `paper_id` | 论文 ID（如 `icml_2024_0001`） |
| `title` | 标题 |
| `abstract` | 摘要 |
| `impact` | 影响声明 |
| `impact_chars` | 影响声明字符数 |

### 中间产出：`4_consensus.json`

每条 paper 目录下最重要的输出，包含通过双边共识的完整因果链：

```json
{
  "rounds": 1,
  "converged": true,
  "chains": [
    {
      "scenario": "...",
      "issue": "...",
      "hazard": ["..."],
      "exposure": ["..."],
      "dose_response": ["..."],
      "vulnerability": ["..."],
      "impact": ["..."],
      "key_control_nodes": ["..."],
      "confidence": "high"
    }
  ]
}
```

### 最终产出：`outputs/dataset.json`

`build_dataset.py` 生成的汇总数据集，合并每条论文的 ref_hevi 与 workflow chains。

## 引用

- Turner, B.L. et al. (2003). A framework for vulnerability analysis in sustainability science. *PNAS*, 100(14), 8074–8079.
- HippoRAG: Gutierrez et al. (2024). HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. *NeurIPS*.
