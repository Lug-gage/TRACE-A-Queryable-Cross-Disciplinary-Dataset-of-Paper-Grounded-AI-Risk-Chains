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

## 整体数据流

```
                          ┌───────────────────────────────────────────┐
                          │  ① HippoRAG-build  索引构建               │
                          │                                           │
                          │  CS 语料 (2,973 篇) ──┐                    │
                          │  SS 语料 (6,934 篇) ──┤                    │
                          │                       ▼                    │
                          │              OpenIE 提取                  │
                          │           (NER + 三元组)                  │
                          │                 │                         │
                          │                 ▼                         │
                          │          嵌入向量化                       │
                          │    (text-embedding-3-large)               │
                          │                 │                         │
                          │                 ▼                         │
                          │      ┌──────────────────┐                 │
                          │      │  知识图谱索引     │                 │
                          │      │  igraph + Parquet │                │
                          │      └────────┬─────────┘                 │
                          └───────────────┼───────────────────────────┘
                                          │  CS / SS 索引
                                          ▼
                          ┌───────────────────────────────────────────┐
                          │  ② hevi_package  HEVI 风险链提取            │
                          │                                           │
                          │  ICML 论文 (5,940 篇)                     │
                          │       │                                   │
                          │       ▼                                   │
                          │  参考 HEVI 提取 ──→ 质量审计 ──→ 311 篇    │
                          │       (替换测试)      (6 维度评分)          │
                          │                                           │
                          │  ┌─────────────────────────────────┐      │
                          │  │       双边协商协议               │      │
                          │  │                                 │      │
                          │  │  CSAgent ─── 提案 ─── Hazard    │      │
                          │  │    │                    Exposure│      │
                          │  │    │  批评 ──────────────────►  │      │
                          │  │    │                    ◄── 修订│      │
                          │  │    ▼                            │      │
                          │  │  SSAgent ─── 响应 ─── Vuln     │      │
                          │  │                      Impact    │      │
                          │  │                      KCN       │      │
                          │  └─────────────────────┬──────────┘      │
                          │                       │                   │
                          │                       ▼                   │
                          │              合成 Dose-Response           │
                          │                       │                   │
                          │                       ▼                   │
                          │      ┌──────────────────────────┐         │
                          │      │  dataset.json             │         │
                          │      │  267 篇论文 × 818 条风险链 │         │
                          │      └────────────┬─────────────┘         │
                          └──────────────────┼───────────────────────┘
                                             │
          ┌──────────────────────────────────┼──────────────────────────────────┐
          │                                  │                                  │
          ▼                                  ▼                                  │
┌─────────────────────┐        ┌─────────────────────┐        ┌─────────────────────┐
│  ③ 实验评估          │        │                      │        │                      │
│                     │        │   HippoRAG (实验 1)   │        │   LightRAG (实验 2)   │
│  dataset.json       │        │   DPR + PPR 图搜索    │        │   Mix 模式检索       │
│       │             │        │   igraph 知识图谱      │        │   NanoVectorDB       │
│       ├─────────────┤        │                      │        │   + NetworkX         │
│       │             │        └──────────┬───────────┘        └──────────┬───────────┘
│       │             │                   │                               │
│       │             │        ┌──────────▼───────────┐        ┌──────────▼───────────┐
│       ├─────────────┤        │  GraphRAG (实验 3)    │        │  纯 LLM (实验 4)     │
│       │             │        │  Local Search         │        │  无检索，参数推理     │
│       │             │        │  LanceDB + Leiden     │        │                      │
│       │             │        └──────────┬───────────┘        └──────────┬───────────┘
│       │             │                   │                               │
│       └──────┬──────┘        ┌──────────▼───────────┐        ┌──────────▼───────────┐
│              │               │  两步骤生成           │        │  两步骤提示           │
│              │               │  VI → 门控 → DR       │        │  VI → 门控 → DR       │
│              │               └──────────┬───────────┘        └──────────┬───────────┘
│              │                          │                               │
│              └──────────────────────────┼───────────────────────────────┘
│                                         │
│                                         ▼
│                          ┌─────────────────────────────┐
│                          │  统一评估                     │
│                          │  LLM-as-Judge (covered/      │
│                          │  partial/not)                │
│                          │  + 嵌入余弦相似度              │
│                          └─────────────────────────────┘
```

---

## 1. HippoRAG-build — 索引构建管线

基于 **HippoRAG 2**（OSU NLP Group, ICML 2025）构建两个跨学科知识图谱索引：

| 索引 | 语料库 | 论文数 | NER 模板 | 状态 |
|------|--------|--------|----------|------|
| **CS** | 计算机科学论文 | 2,973 | `ner_risk_cs` / `triple_extraction_risk_cs` | ✅ 已构建 |
| **SS** | 社会科学论文 | 6,934 | `ner_risk_ss` / `triple_extraction_risk_ss` | ✅ 已构建 |

### 数据流

```
papers/processed/{cs,ss}_corpus.csv                # 原始语料 (title, abstract, impact, doc_text)
        │
        ▼
scripts/build_paper_indexes.py                     # CLI 入口，调度整个管线
        │
        ▼
src/hipporag/HippoRAG.py :: index()                # 核心索引方法
        │
        ├──▶ src/hipporag/information_extraction/   # OpenIE 提取
        │    openie_openai.py                       #   NER (ner_risk_cs / ner_risk_ss 模板)
        │    │                                      #   三元组提取 (triple_extraction_risk_* 模板)
        │    │                                      #   LLM: deepseek-v4-pro
        │    ▼
        ├──▶ src/hipporag/embedding_model/          # 嵌入向量化
        │    OpenAI.py                              #   text-embedding-3-large (3072 维)
        │    │                                      #   → chunk/entity/fact 三类向量
        │    ▼
        ├──▶ src/hipporag/embedding_store.py        # Parquet 持久化
        │    │                                      #   vdb_chunk.parquet
        │    │                                      #   vdb_entity.parquet
        │    │                                      #   vdb_fact.parquet
        │    ▼
        └──▶ src/hipporag/HippoRAG.py               # 知识图谱构建
             :: augment_graph()                     #   igraph: 实体节点 + 事实边 + 同义边
             :: save_igraph()                       #   graph.pickle
                    │
                    ▼
indices/{cs,ss}/
        ├── openie_results_ner_deepseek-v4-pro.json # NER + 三元组结果 (~17 MB)
        ├── index_manifest.json                     # 索引构建元数据
        └── deepseek-v4-pro_text-embedding-3-large/ # 嵌入向量 (⚠️ 不包含在仓库中)
            ├── chunk_embeddings/
            ├── entity_embeddings/
            └── fact_embeddings/
```

> **注意**: 嵌入向量文件 (`deepseek-v4-pro_text-embedding-3-large/`) 和 `llm_cache/` **不包含在此仓库中**，可通过运行 `build_paper_indexes.py` 重新生成。

---

## 2. hevi_package — HEVI 风险链提取管线

基于 **Turner et al. (2003)** 脆弱性分析框架，使用 **CS-SS 双边协商多智能体协议** 自动提取和扩增 AI 风险链。

### HEVI 框架（6 个风险槽位）

| 槽位 | 定义 | 负责智能体 |
|------|------|-----------|
| **H**azard（危害）| 论文方法引入或放大的技术能力 | CSAgent |
| **E**xposure（暴露）| 面临危害的对象/系统元素 | CSAgent |
| **D**ose-Response（剂量-反应）| 危害程度转化为影响幅度的因果翻译 | 合成 |
| **V**ulnerability（脆弱性）| 使危害更可能发生的条件/差距 | SSAgent |
| **I**mpact（影响）| 负面社会后果 | SSAgent |
| **K**ey Control Nodes（关键控制节点）| 阻断风险链的干预点 | SSAgent |

### 数据流

```
data/icml_corpus_with_len.csv                      # 5,940 篇 ICML 论文
        │
        ▼
scripts/extract_reference_hevi.py                   # 阶段 0: 参考 HEVI 提取
        │   prompts: (内嵌于 pipeline.py)
        │   LLM 调用 A: title+abstract → CS 检索查询
        │   LLM 调用 B: impact statement → ref_hevi 槽位 (替换测试过滤泛化模板)
        │   impact_chars < 500 → 跳过
        ▼
scripts/audit_hevi_quality.py                       # 阶段 1: 质量审计
        │   prompts: (内嵌于 pipeline.py)
        │   6 维度评分: discovery_feasibility / technical_anchoring /
        │              direction_correctness / grounding / slot_correctness / specificity
        │   硬过滤: overall ≥ 0.75, ≥ 3 个非空槽位
        │   产出: quality_report.json → 311 篇 keep 论文
        ▼
scripts/run_hevi_pipeline.py                        # 阶段 2-5: 双边协商管线
        │
        │   ┌─ 阶段 2: CS 智能体提案 ─────────────────────────────────┐
        │   │  hipporag/hevi_workflow/agents.py :: CSAgent            │
        │   │  prompts/01_cs_propose.md                               │
        │   │  indices/cs/ ──▶ RiskRetriever ──▶ CS 证据检索          │
        │   │  LLM 调用 C: 生成 hazard + exposure + nexus_candidates │
        │   │  LLM 调用 D: judge_hits_with_llm (证据相关性)           │
        │   └────────────────────────────────────────────────────────┘
        │                    │
        │   ┌─ 阶段 3: SS 智能体响应 ─────────────────────────────────┐
        │   │  hipporag/hevi_workflow/agents.py :: SSAgent            │
        │   │  prompts/03_ss_query_planning.md                        │
        │   │  prompts/04_ss_respond.md                               │
        │   │  indices/ss/ ──▶ RiskRetriever ──▶ SS 证据检索          │
        │   │  LLM 调用 E: 从 CS nexus → SS 检索查询                  │
        │   │  LLM 调用 F: 生成 vulnerability + impact + KCN         │
        │   └────────────────────────────────────────────────────────┘
        │                    │
        │   ┌─ 阶段 4: 双边共识 ──────────────────────────────────────┐
        │   │  hipporag/hevi_workflow/pipeline.py                     │
        │   │  prompts/05_cs_critique_ss.md ──▶ CSAgent 批评 SS      │
        │   │  prompts/06_ss_critique_cs.md ──▶ SSAgent 批评 CS      │
        │   │  prompts/07_cs_revise.md ──▶ CSAgent 修订提案           │
        │   │  prompts/08_ss_revise.md ──▶ SSAgent 修订响应           │
        │   │  ← 循环 3 轮或 self_score ≥ 0.8 →                       │
        │   │  prompts/09_dr_synthesis.md ──▶ 合成 Dose-Response      │
        │   └────────────────────────────────────────────────────────┘
        │                    │
        │   ┌─ 阶段 5: 召回比较 ──────────────────────────────────────┐
        │   │  prompts/10_hevi_compare.md                             │
        │   │  LLM 调用: workflow HEVI 与 ref_hevi 逐项语义召回率      │
        │   └────────────────────────────────────────────────────────┘
        │
        ▼
scripts/build_dataset.py                             # 合并所有阶段产出
        │
        ▼
outputs/dataset.json                                 # 267 篇论文 × 818 条 HEVI 风险链
```

---

## 3. 实验 1: HippoRAG — 神经生物学启发的图 RAG

基于 **HippoRAG 2**（From RAG to Memory, ICML 2025）框架进行 HEVI 风险评估。

### 数据流

```
dataset.json                                         # 267 篇论文, 818 条链
        │
        ▼
hevi_query/scripts/hevi_query_hipporag.py             # 生成入口
        │
        ├──▶ he_query = f"{hazard}. {exposure}."     # 双路 DPR 检索
        │    si_query = f"{scenario}. {issue}."       #
        │    │                                        #
        │    ▼                                        #
        │    src/hipporag/HippoRAG.py :: retrieve()   # 检索
        │    │    DPR 向量检索 → top 3 per query      #
        │    │    DSPyFilter 重排 (rerank.py)         #
        │    │    PPR 图搜索 (igraph)                 #
        │    │    → 合并去重 (最多 6 篇)               #
        │    ▼                                        #
        │    检索上下文 (截断 10k 字符)                 #
        │                                             #
        ├──▶ 阶段 1: VI 生成                          # LLM 推理
        │    prompt: hevi_vuln_impact.txt             #
        │    │    Turner 定义 + 检索上下文 + 论文元数据  #
        │    │    → vulnerability (≤30 词)             #
        │    │    → impact (≤30 词)                    #
        │    ▼                                        #
        │    ├─ VI 非空 → 进入阶段 2                    #
        │    └─ VI 为空 → 跳过 DR, 标记失败             #
        │                                             #
        └──▶ 阶段 2: DR 合成                          #
             prompt: hevi_dr.txt                      #
             │    无检索, 纯 LLM 推理                   #
             │    → dose_response (≤40 词)             #
             ▼                                        #
hevi_query/hipporag_results/{paper_id}_chain{N}.json   # 生成结果
        │
        ▼
hevi_query/scripts/eval_hevi.py                       # LLM-as-Judge 评估
        │   prompt: eval_prompt.txt
        │   → covered / partial / not + 理由
        ▼
hevi_query/evaluation/{paper_id}_chain{N}.json
        │
hevi_query/scripts/eval_hevi_embedding.py             # 嵌入相似度评估
        │   text-embedding-3-large → 余弦相似度 (0-1)
        ▼
hevi_query/evaluation_embedding/{paper_id}_chain{N}.json
```

---

## 4. 实验 2: LightRAG — 轻量级图 RAG

基于 **LightRAG**（HKUDS, arXiv 2410.05779）框架进行 HEVI 风险评估。

### 数据流

```
indices/ss/openie_results_ner_deepseek-v4-pro.json    # HippoRAG-build 产出的 OpenIE 结果
        │                                              # 6,934 块, 56,505 实体, 26,394 三元组
        ▼
indices/build_kg.py                                   # 定制 KG 构建
        │   load_indices_as_custom_kg()
        │   → 注入 chunk / entity / relation
        │   → ainsert_custom_kg()
        │   → NanoVectorDB (实体/关系/块向量, 3072d)
        │   → NetworkX (Chunk-Entity-Relation 图)
        ▼
lightrag/                                             # LightRAG 核心库
        │
        ▼
hevi_query/scripts/hevi_query.py                      # 生成入口
        │
        ├──▶ 检索阶段
        │    aquery_data(mode="mix", top_k=40)
        │    he_query + si_query → 并行检索
        │    合并去重实体/关系/块 → 截断 10k 字符
        │
        ├──▶ 阶段 1: VI 生成
        │    检索上下文 + Turner 定义 → vulnerability + impact
        │    ├─ VI 非空 → 进入阶段 2
        │    └─ VI 为空 → 跳过 (纯 LLM 回退)
        │
        └──▶ 阶段 2: DR 合成
             无检索，纯 LLM → dose_response
        │
        ▼
hevi_query/lightrag_results/{paper_id}_chain{N}.json
        │
        ├──▶ eval_hevi.py ──▶ evaluation/
        └──▶ eval_hevi_embedding.py ──▶ evaluation_embedding/
```

---

## 5. 实验 3: GraphRAG — 层次社区增强图 RAG

基于 **Microsoft GraphRAG v3.1.0** 框架进行 HEVI 风险评估。

### 数据流

```
openie_results_ner_deepseek-v4-pro.json               # HippoRAG-build 产出的 OpenIE 结果
        │
        ▼
build_index.py                                        # 定制索引入口 (绕过官方 LLM 提取管道)
        │
        ├──▶ 解析 JSON → 实体/关系 DataFrame
        ├──▶ cluster_graph() → Leiden 层次聚类 (max 50)
        ├──▶ LLM 生成社区报告 (或 --fast 占位符)
        ├──▶ text-embedding-3-large → LanceDB (entity_description 表)
        └──▶ 输出 Parquet: entities / relationships / text_units /
                            communities / community_reports / documents
        ▼
indices/ss/output/                                    # 索引产物
        ├── *.parquet
        ├── lancedb/                                  # LanceDB 向量存储
        └── settings.yaml                             # 查询配置
        │
        ▼
hevi_query/scripts/hevi_query_graphrag.py              # 生成入口
        │
        ├──▶ 检索: local_search()                      # 检索+生成合一
        │    查询嵌入 → LanceDB 实体检索                #
        │    → 关系扩展 → 社区上下文注入                 #
        │    → LLM 生成 vuln + impact + dr              #
        │
        ▼
hevi_query/graph_result/{paper_id}_chain{N}.json
        │
        ├──▶ eval_hevi.py ──▶ evaluation/
        └──▶ eval_hevi_embedding.py ──▶ evaluation_embedding/
```

---

## 6. 实验 4: 纯 LLM 基线（无检索）

**纯参数推理**基线，不依赖任何外部知识库。LLM 仅接收论文元数据，无知识图谱、无向量检索。

### 数据流

```
dataset.json                                         # 267 篇论文, 818 条链
        │
        ▼
scripts/hevi_query_llm.py                             # 生成入口
        │
        ├──▶ 阶段 1: VI 生成
        │    prompt: hevi_vuln_impact.txt
        │    输入: title + abstract + scenario + issue + hazard + exposure
        │    输出: vulnerability (≤30 词) + impact (≤30 词)
        │    ├─ VI 非空 → 进入阶段 2
        │    └─ VI 为空 → 跳过
        │
        │    * 无检索: 仅依赖 LLM 参数化知识
        │
        └──▶ 阶段 2: DR 合成
             prompt: hevi_dr.txt
             输入: VI 结果 + impact statement
             输出: dose_response (≤40 词)
        │
        ▼
llm_results/{paper_id}_chain{N}.json                  # 818 条链的生成结果
        │
        ├──▶ scripts/eval_hevi.py ──▶ evaluation_llm/
        └──▶ scripts/eval_hevi_embedding.py ──▶ evaluation_embedding/
```

### 基线结果（DeepSeek-V4-Pro）

| 字段 | Covered | 覆盖率 |
|------|---------|--------|
| Vulnerability | 426/818 | **52.1%** |
| Impact | 334/818 | **40.8%** |
| Dose-Response | 309/818 | **37.8%** |
| **综合** | 1069/2454 | **43.6%** |

---

## 数据集统计

| 指标 | 数值 |
|------|------|
| ICML 论文总数 | **267**（ICML 2024: 129, ICML 2025: 138）|
| HEVI 风险链总数 | **818** |
| 平均链数/论文 | **3.06** |
| CS 语料库（索引）| 2,973 篇 |
| SS 语料库（索引）| 6,934 篇 |
| SS 索引实体 | 56,505 |
| SS 索引三元组 | 26,394 |
| LLM 模型 | DeepSeek-V4-Pro |
| 嵌入模型 | text-embedding-3-large（3,072 维）|

---

## 实验对照

| 维度 | HippoRAG | LightRAG | GraphRAG | LLM 基线 |
|------|----------|----------|----------|----------|
| **检索类型** | DPR + PPR 图搜索 | Mix 模式（KG+向量）| Local Search | 无检索 |
| **图结构** | igraph + 同义词边 | NetworkX | Parquet + Leiden 聚类 | — |
| **向量存储** | Parquet（余弦）| NanoVectorDB（余弦）| LanceDB（余弦）| — |
| **重排序** | DSPyFilter | 可选内置 | 社区上下文 | — |
| **生成策略** | 检索→VI→门控→DR | 检索→VI→门控→DR | 检索+生成合一 | 纯推理 VI→DR |
| **评估方法** | LLM-Judge + 嵌入 | LLM-Judge + 嵌入 | LLM-Judge + 嵌入 | LLM-Judge + 嵌入 |

---

## 生成索引库

索引库的嵌入向量文件（~2.5 GB/索引）**不包含在此仓库中**。如需重新生成：

```bash
# CS 索引
cd HippoRAG-build
python scripts/build_paper_indexes.py --source cs \
    --corpus-path papers/processed/cs_corpus.csv \
    --llm deepseek-v4-pro

# SS 索引
python scripts/build_paper_indexes.py --source ss \
    --corpus-path papers/processed/ss_corpus.csv \
    --llm deepseek-v4-pro
```

---

## 引用

本项目使用了以下开源框架：

- **HippoRAG 2**: Gutierrez et al., "From RAG to Memory: Non-Parametric Continual Learning for Large Language Models", ICML 2025. [arXiv 2502.14802](https://arxiv.org/abs/2502.14802)
- **LightRAG**: Guo et al., "LightRAG: Simple and Fast Retrieval-Augmented Generation", 2024. [arXiv 2410.05779](https://arxiv.org/abs/2410.05779)
- **GraphRAG**: Microsoft Research, "GraphRAG: A modular graph-based Retrieval-Augmented Generation system", 2024.
- **Turner et al.**: "A framework for vulnerability analysis in sustainability science", PNAS 2003.

---

## 许可证

MIT License — 各子项目的原始许可证保留在其各自目录中。
