# TRACE: A Queryable Cross-Disciplinary Dataset of Paper-Grounded AI Risk Chains

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**TRACE** 是一个从 ICML 论文中自动提取、并通过跨学科证据检索（CS/SS）验证的 **AI 风险因果链数据集**。该项目产出了 267 篇论文 × 818 条 HEVI（Hazard-Exposure-Vulnerability-Impact）风险链，并支持对 4 种 RAG 范式（HippoRAG、LightRAG、GraphRAG、纯 LLM）进行系统基准测试。

---

## 目录结构

```
TRACE/
├── HippoRAG-build/       # 索引构建管线：构建 CS 和 SS 知识图谱索引
├── hevi_package/         # HEVI 管线：多智能体双边共识提取风险链 → dataset.json
├── HippoRAG/             # 实验 1：HippoRAG 检索增强风险评估
├── LightRAG/             # 实验 2：LightRAG 检索增强风险评估
├── graphrag/             # 实验 3：Microsoft GraphRAG 检索增强风险评估
└── LLM/                  # 实验 4：纯 LLM 基线（无检索）
```
---

## 1. HippoRAG-build — 索引构建管线

基于 **HippoRAG 2**（OSU NLP Group, ICML 2025）构建两个知识图谱索引：

### 步骤
1. **文档加载**: 从 `papers/processed/{cs,ss}_corpus.csv` 读取论文（含 title、abstract、impact、doc_text）
2. **OpenIE 提取**: 使用 `deepseek-v4-pro` 进行 NER 命名实体识别 + 三元组关系提取
3. **嵌入向量化**: 使用 `text-embedding-3-large` 对文本块/实体/事实进行嵌入
4. **知识图谱构建**: 构建 igraph 图（实体节点 + 事实边 + 同义边）
5. **输出**: `openie_results_ner_*.json`（NER+三元组）、嵌入 Parquet 文件、图 pickle

### 关键文件
```
HippoRAG-build/
├── scripts/build_paper_indexes.py          # 索引构建入口
├── src/hipporag/                           # HippoRAG 2 核心库
│   ├── HippoRAG.py                         # 索引/检索/QA 主类
│   ├── embedding_store.py                  # Parquet 向量存储
│   ├── hevi_workflow/                      # HEVI 风险分析工作流
│   │   ├── agents.py                       # CSAgent + SSAgent 双边协商
│   │   ├── pipeline.py                     # 管线编排
│   │   ├── retrievers.py                   # HippoRAG 检索器封装
│   │   └── evaluator.py                    # 命中评估
│   ├── information_extraction/             # OpenIE 模块
│   ├── prompts/templates/                  # 风险感知 NER/三元组模板
│   └── rerank.py                           # DSPy 事实重排
├── indices/
│   ├── cs/openie_results_ner_deepseek-v4-pro.json   # CS 语料库 OpenIE 结果
│   └── ss/openie_results_ner_deepseek-v4-pro.json   # SS 语料库 OpenIE 结果
└── papers/processed/{cs,ss}_corpus.csv     # 原始语料库
```

> **注意**: 嵌入向量文件 (`deepseek-v4-pro_text-embedding-3-large/`) 和 `llm_cache/` **不包含在此仓库中**，可以通过运行 `build_paper_indexes.py` 重新生成。

---

## 2. hevi_package — HEVI 风险链提取管线

基于 **Turner et al. (2003)** 脆弱性分析框架，通过 **CS-SS 双边协商多智能体协议** 从 ICML 论文的 impact statement 中自动提取和扩增 AI 风险链。

### HEVI 框架（6 个风险槽位）

| 槽位 | 定义 | 负责智能体 |
|------|------|-----------|
| **H**azard（危害）| 论文方法引入或放大的技术能力 | CSAgent |
| **E**xposure（暴露）| 面临危害的对象/系统元素 | CSAgent |
| **D**ose-Response（剂量-反应）| 危害程度转化为影响幅度的因果翻译 | 合成 |
| **V**ulnerability（脆弱性）| 使危害更可能发生的条件/差距 | SSAgent |
| **I**mpact（影响）| 负面社会后果 | SSAgent |
| **K**ey Control Nodes（关键控制节点）| 阻断风险链的干预点 | SSAgent |

### 步骤

1. **Reference Extraction**: 从 ICML 论文 impact statement 中提取参考 HEVI 槽位（使用"替换测试"排除泛化模板）
2. **Quality Audit**: 对提取结果进行 6 维度质量评分（技术锚定性、方向正确性、具体性等），筛选 `keep` 论文
3. **CS Proposal**: CSAgent 基于 CS 索引检索证据，提出 Hazard → Exposure
4. **SS Response**: SSAgent 基于 SS 索引检索证据，提出 Vulnerability → Impact → KCN
5. **Bilateral Consensus**: 两个智能体互相批评和修订（最多 3 轮），直到双方自评分 ≥ 0.8，然后合成 Dose-Response 链
6. **Comparison**: 将流水线生成的链与参考提取进行召回率比较

### 关键文件
```
hevi_package/
├── hevi_run.py                              # CLI 入口（extract/audit/run/export/all）
├── hipporag/hevi_workflow/
│   ├── agents.py                            # CSAgent + SSAgent + RiskLLM
│   ├── pipeline.py                          # 双边共识管线编排
│   └── retrievers.py                        # HippoRAGRetriever 封装
├── prompts/                                 # 10 个 LLM 提示词模板
│   ├── 01_cs_propose.md                     # CS 智能体提案
│   ├── 04_ss_respond.md                     # SS 智能体响应
│   ├── 05_cs_critique_ss.md                 # CS 批评 SS
│   ├── 06_ss_critique_cs.md                 # SS 批评 CS
│   └── 09_dr_synthesis.md                   # 剂量-反应合成
├── scripts/
│   ├── extract_reference_hevi.py            # 阶段 0: 参考提取
│   ├── audit_hevi_quality.py                # 阶段 1: 质量审计
│   ├── run_hevi_pipeline.py                 # 阶段 2-5: 完整管线
│   ├── build_dataset.py          ★          # 合并两路产出 → dataset.json
│   ├── export_hevi_csv.py                   # 导出 CSV 对比表
│   └── generate_visual.py                   # 交互式可视化
├── data/icml_corpus_with_len.csv            # 5,940 篇 ICML 论文语料库
├── indices/{cs,ss}/                         # 复用的索引（来自 HippoRAG-build）
└── outputs/
    ├── hevi_workflow/
    │   ├── hevi_icml_deepseek-v4-pro/       # 阶段 0 提取结果 ({paper_id}.json)
    │   │   └── {paper_id}.json   ─────────┐
    │   └── hevi_deepseek-v4-pro/           │   # 阶段 2-5 管线结果 ({paper_id}/)
    │       └── {paper_id}/                 │
    │           ├── 1_reference.json        │
    │           ├── 2_cs_proposal.json      │
    │           ├── 3_ss_response.json      │
    │           ├── 4_consensus.json ───────┤
    │           └── 5_compare.json          │
    └── dataset.json              ★         │   # 最终数据集
                                            │
              build_dataset.py 合并逻辑 ────┘
              
    # build_dataset.py 从两条路径读取，取交集：
    #   路径 1: outputs/hevi_workflow/hevi_icml_deepseek-v4-pro/{paper_id}.json
    #           → 提取 title, abstract, impact, ref_hevi
    #   路径 2: outputs/hevi_workflow/hevi_deepseek-v4-pro/{paper_id}/4_consensus.json
    #           → extract_chain(): 裁剪每条 chain 为 {scenario, issue} + 6 HEVI 槽位
    #   仅保留两路均存在的论文 → 输出 dataset.json (267 篇 × 818 条链)
```

---

## 3. 实验 1: HippoRAG — 神经生物学启发的图 RAG

基于 **HippoRAG 2**（From RAG to Memory, ICML 2025）框架进行 HEVI 风险评估。

### 检索策略
- **双路 DPR 检索**: `he_query`（hazard+exposure）→ Top 3，`si_query`（scenario+issue）→ Top 3，合并去重
- **DSPyFilter 重排**: 可选 LLM 事实过滤
- **PPR 图搜索**: 个性化 PageRank 在 igraph 知识图谱上传播
- **两步骤生成**: Step 1 (VI) → 门控 → Step 2 (DR)，无检索纯推理

### 关键文件
```
HippoRAG/
├── main.py                                  # 标准多跳 QA 基线实验
├── build_hlg_index.py                       # 层级知识图谱索引构建
├── mine_innovation_pairs.py                 # 跨论文创新节点对挖掘
├── run_node_eval.py                         # 层级知识图谱对齐评估（V2+V3）
├── hevi_query/
│   ├── dataset.json                         # HEVI 评估数据集
│   └── scripts/
│       ├── hevi_query_hipporag.py           # HippoRAG 增强生成
│       ├── hevi_query_llm.py                # 纯 LLM 对比
│       ├── eval_hevi.py                     # LLM-as-Judge 评估
│       ├── eval_hevi_embedding.py           # 嵌入相似度评估
│       └── analyze_joint.py                 # 联合分布分析
├── indices/{cs,ss}/                         # 预建索引
└── reproduce/dataset/                       # 标准多跳 QA 数据集
```

---

## 4. 实验 2: LightRAG — 轻量级图 RAG

基于 **LightRAG**（HKUDS, arXiv 2410.05779）框架进行 HEVI 风险评估。

### 检索策略
- **Mix 模式**: 同时检索 Entity + Relation + Chunk（局部 KG + 全局 KG + 向量）
- **向量存储**: NanoVectorDB（余弦相似度，3,072 维 text-embedding-3-large）
- **图存储**: NetworkX（实体-关系-文本块图）
- **定制 KG 构建**: 从 OpenIE JSON 直接注入实体/关系/块，绕过了 LightRAG 原生 LLM 提取器
- **查询**: 使用 `aquery_data()` 获取结构化检索数据（不经过 LLM 生成）

### 关键文件
```
LightRAG/
├── lightrag/                                # LightRAG 核心库
├── indices/
│   ├── build_kg.py                          # 定制知识图谱构建
│   └── ss/openie_results_ner_deepseek-v4-pro.json
├── hevi_query/
│   ├── dataset.json
│   └── scripts/
│       ├── hevi_query.py                    # LightRAG 增强生成
│       ├── eval_hevi.py                     # LLM-as-Judge 评估
│       └── eval_hevi_embedding.py           # 嵌入相似度评估
└── examples/                                # 官方演示脚本
```

---

## 5. 实验 3: GraphRAG — 层次社区增强图 RAG

基于 **Microsoft GraphRAG v3.1.0** 框架进行 HEVI 风险评估。

### 检索策略
- **Local Search 模式**: 查询嵌入 → 实体检索（LanceDB 向量相似度）→ 关系扩展 → 社区上下文注入
- **向量存储**: LanceDB（3,072 维 text-embedding-3-large）
- **图结构**: Parquet DataFrame + Leiden 层次聚类
- **定制索引入口**: `build_index.py` 从 OpenIE JSON 构建 Parquet 索引，绕过了官方 LLM 提取管道
- **查询**: 使用 `local_search()` API（检索+生成合一）

### 关键文件
```
graphrag/
├── build_index.py                           # 定制索引入口
├── packages/                                # GraphRAG monorepo（8 个子包）
│   ├── graphrag/                            # 核心 CLI/查询引擎
│   ├── graphrag-llm/                        # LLM 接口（litellm）
│   ├── graphrag-vectors/                    # 向量存储（LanceDB）
│   └── graphrag-storage/                    # 存储后端
├── hevi_query/
│   ├── dataset.json
│   └── scripts/
│       ├── hevi_query_graphrag.py           # GraphRAG 增强生成
│       ├── eval_hevi.py
│       └── eval_hevi_embedding.py
├── indices/ss/
│   ├── settings.yaml                        # GraphRAG 配置
│   └── openie_results_ner_deepseek-v4-pro.json
└── openie_results_ner_deepseek-v4-pro.json  # 预提取的 OpenIE 结果
```

---

## 6. 实验 4: 纯 LLM 基线（无检索）

**纯参数推理**的基线实验，不依赖任何外部知识库。

### 设计
- **无知识库**: LLM 仅接收论文标题、摘要、impact statement 和链上下文（scenario/issue/hazard/exposure）
- **两步骤提示**: Step 1 生成 vulnerability + impact，Step 2 生成 dose_response
- **严格长度约束**: VI ≤ 30 词，DR ≤ 40 词，禁止推测性语言
- **评估**: 与其余实验使用相同的 LLM-as-Judge 和嵌入相似度评估流程

### 关键文件
```
LLM/
├── dataset.json                             # 评估数据集（267 篇论文，818 条链）
├── scripts/
│   ├── hevi_query_llm.py                    # 纯 LLM 生成
│   ├── eval_hevi.py                         # LLM-as-Judge 评估
│   ├── eval_hevi_embedding.py               # 嵌入相似度评估
│   ├── hevi_vuln_impact.txt                 # VI 生成提示模板
│   ├── hevi_dr.txt                          # DR 生成提示模板
│   └── eval_prompt.txt                      # 评估提示模板
├── llm_results/                             # 818 条链的生成结果
├── evaluation_llm/                          # LLM-as-Judge 评估结果
└── evaluation_embedding/                    # 嵌入相似度评估结果
```

---

## 依赖

所有项目共享以下核心依赖：

- **Python** ≥ 3.10
- **LLM API**: OpenAI 兼容端点（DeepSeek-V4-Pro）
- **嵌入**: text-embedding-3-large（3,072 维）
- **核心库**: numpy, pandas, openai, tiktoken

各项目的特定依赖详见各自的 `requirements.txt`。

---

## 生成索引库

所有嵌入向量和缓存文件 **不包含在此仓库中**，需按以下步骤重新生成。三个框架共享同一份 OpenIE 中间产物 `openie_results_ner_deepseek-v4-pro.json`，由 HippoRAG-build 产出，GraphRAG 和 LightRAG 各自从其构建自己的索引格式。

> ⚠️ 以下命令均需在对应项目目录下运行，并确保 API key 已配置（HippoRAG-build 读 `.env`，GraphRAG/LightRAG 脚本内硬编码）。

### 1. HippoRAG（OpenIE + 嵌入 + igraph）

产出 CS/SS 两个索引：OpenIE JSON、Parquet 嵌入文件、igraph 图 pickle。

```bash
cd HippoRAG-build

# CS 索引（2,973 篇论文，模板 ner_risk_cs / triple_extraction_risk_cs）
python scripts/build_paper_indexes.py \
    --source cs \
    --llm-name deepseek-v4-pro \
    --embedding-name text-embedding-3-large \
    --openie-workers 4

# SS 索引（6,934 篇论文，模板 ner_risk_ss / triple_extraction_risk_ss）
python scripts/build_paper_indexes.py \
    --source ss \
    --llm-name deepseek-v4-pro \
    --embedding-name text-embedding-3-large \
    --openie-workers 2
```

**产出位置**: `indices/{cs,ss}/`  

### 2. GraphRAG（OpenIE JSON → Parquet + LanceDB + Leiden 社区）

GraphRAG 的 `build_index.py` 直接读取 HippoRAG-build 产出的 `openie_results_ner_*.json`，将其转换为 GraphRAG 原生格式（DataFrame → Leiden 聚类 → LanceDB 向量库）。

```bash
cd graphrag

# SS 完整建库（含 LLM 社区报告）
python build_index.py ss

# CS 完整建库（~15 min）
python build_index.py cs

# 快速模式：跳过 LLM 社区报告，用占位文本（local/basic 查询可用）
python build_index.py ss --fast
python build_index.py cs --fast
```
**产出位置**: `indices/{cs,ss}/output/`（`*.parquet` + `lancedb/` + `settings.yaml`）

### 3. LightRAG（OpenIE JSON → NanoVectorDB + NetworkX）

LightRAG 的 `build_kg.py` 从 OpenIE JSON 提取 chunk/entity/relation 三元组，通过 `ainsert_custom_kg()` 注入 LightRAG，构建 NanoVectorDB 向量库 + NetworkX 图。

```bash
cd LightRAG

# 仅建 SS
python indices/build_kg.py ss

# 仅建 CS
python indices/build_kg.py cs

```
**产出位置**: `rag_storage_{cs,ss}/`（`vdb_*.json` + `graph_chunk_entity_relation.graphml` + `kv_store_*.json`）



## 引用

本项目使用了以下开源框架：

- **HippoRAG 2**: Gutierrez et al., "From RAG to Memory: Non-Parametric Continual Learning for Large Language Models", ICML 2025. [arXiv 2502.14802](https://arxiv.org/abs/2502.14802)
- **LightRAG**: Guo et al., "LightRAG: Simple and Fast Retrieval-Augmented Generation", 2024. [arXiv 2410.05779](https://arxiv.org/abs/2410.05779)
- **GraphRAG**: Microsoft Research, "GraphRAG: A modular graph-based Retrieval-Augmented Generation system", 2024.
- **Turner et al.**: "A framework for vulnerability analysis in sustainability science", PNAS 2003.

---

## 许可证

MIT License — 各子项目的原始许可证保留在其各自目录中。
