# TRACE: A Queryable Cross-Disciplinary Dataset of Paper-Grounded AI Risk Chains

从 ICML 论文中自动提取 AI 风险因果链，并通过跨学科证据检索（CS/SS）验证。产出 **267 篇论文 × 818 条 HEVI 风险链**，支持 4 种 RAG 范式的系统基准测试。

---

## 目录结构

```
TRACE/
├── HippoRAG-build/    # 索引构建：CS + SS 知识图谱
├── hevi_package/      # HEVI 提取管线 → dataset.json
├── HippoRAG/          # 实验 1：HippoRAG
├── LightRAG/          # 实验 2：LightRAG
├── graphrag/          # 实验 3：GraphRAG
└── LLM/               # 实验 4：纯 LLM 基线
```

---

## 整体流程

```
                   ┌─────────────────────┐
                   │   ① HippoRAG-build   │
                   │   CS 2,973 篇        │
                   │   SS 6,934 篇        │
                   │   → KG 索引          │
                   └──────────┬──────────┘
                              │ indices/
                              ▼
                   ┌─────────────────────┐
                   │   ② hevi_package     │
                   │   ICML 5,940 篇      │
                   │   双边协商协议        │
                   │   → dataset.json     │
                   └──────────┬──────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
   ┌──────────┐       ┌──────────┐       ┌──────────────┐
   │ HippoRAG │       │ LightRAG │       │  GraphRAG    │       ┌──────────┐
   │ DPR+PPR  │       │ Mix      │       │  Local       │       │ 纯 LLM   │
   └──────────┘       └──────────┘       │  Search      │       │ 无检索   │
                                         └──────────────┘       └──────────┘
         │                    │                    │                  │
         └────────────────────┴────────────────────┴──────────────────┘
                              │
                              ▼
                     统一评估 (LLM-as-Judge + 嵌入余弦相似度)
```

---

## ① HippoRAG-build — 索引构建

基于 HippoRAG 2 (ICML 2025) 构建 CS/SS 两个知识图谱索引。

| 索引 | 论文数 | NER 模板 |
|------|--------|----------|
| CS | 2,973 | `ner_risk_cs` / `triple_extraction_risk_cs` |
| SS | 6,934 | `ner_risk_ss` / `triple_extraction_risk_ss` |

**管线**：`corpus.csv` → OpenIE 提取 (NER + 三元组) → 嵌入向量化 (text-embedding-3-large) → igraph 知识图谱。

> ⚠️ 嵌入向量文件 (`deepseek-v4-pro_text-embedding-3-large/`) 和 `llm_cache/` 不包含在仓库中，需通过 `build_paper_indexes.py` 重新生成。

**入口**：`scripts/build_paper_indexes.py`

---

## ② hevi_package — HEVI 风险链提取

基于 Turner et al. (2003) 框架，使用 CS/SS 智能体双边协商自动提取 6 槽位风险链：

| 槽位 | 含义 | 负责 |
|------|------|------|
| Hazard | 技术能力 | CSAgent |
| Exposure | 接触对象 | CSAgent |
| Dose-Response | 因果翻译 | 合成 |
| Vulnerability | 条件/差距 | SSAgent |
| Impact | 社会后果 | SSAgent |
| Key Control Nodes | 干预点 | SSAgent |

**管线 6 阶段**：

1. **参考提取** — 从 impact statement 提取 ref_hevi（替换测试过滤泛化模板）
2. **质量审计** — 6 维度评分，硬过滤保留 ~311 篇
3. **CS 提案** — CSAgent 检索 CS 索引，生成 Hazard + Exposure
4. **SS 响应** — SSAgent 检索 SS 索引，生成 Vuln + Impact + KCN
5. **双边共识** — 互相批评修订，直到自评分 ≥ 0.8（最多 3 轮），合成 Dose-Response
6. **召回比较** — workflow HEVI 与 ref_hevi 逐项语义召回率

**入口**：`hevi_run.py` (子命令 extract/audit/run/all)

### dataset.json 构造

`scripts/build_dataset.py` 从两条路径取交集合并：

```
阶段 0 输出:  hevi_icml_deepseek-v4-pro/{paper_id}.json        → title, abstract, impact, ref_hevi
阶段 4 输出:  hevi_deepseek-v4-pro/{paper_id}/4_consensus.json  → chains
                                                        ↓
                                              dataset.json
                                         (267 篇 × 818 条链)
```

每条 chain 包含：`scenario, issue, hazard[], exposure[], dose_response[], vulnerability[], impact[], key_control_nodes[]`

---

## ③ 实验

四条评估管道共享相同的数据集和评估协议。

### 生成

所有实验采用统一的两步骤生成：

1. **VI 生成** — 输入 chain 上下文 + 检索上下文（如果有），输出 vulnerability + impact
2. **门控** — VI 为空则跳过 DR
3. **DR 合成** — 无检索，纯 LLM 推理，输出 dose_response

### 评估

- **LLM-as-Judge** — DeepSeek-V4-Pro 语义判断 covered/partial/not
- **嵌入余弦相似度** — text-embedding-3-large 确定性评分

### 实验对照

| 维度 | HippoRAG | LightRAG | GraphRAG | LLM 基线 |
|------|----------|----------|----------|----------|
| 检索 | DPR + PPR 图搜索 | Mix (KG+向量) | Local Search | 无 |
| 图存储 | igraph | NetworkX | Parquet + Leiden | — |
| 向量存储 | Parquet | NanoVectorDB | LanceDB | — |
| 重排序 | DSPyFilter | 可选 | 社区上下文 | — |

### 基线结果（纯 LLM）

| 字段 | 覆盖率 |
|------|--------|
| Vulnerability | 52.1% |
| Impact | 40.8% |
| Dose-Response | 37.8% |
| **综合** | **43.6%** |

---

## 数据统计

| 指标 | 数值 |
|------|------|
| ICML 论文 | 267 (ICML 2024: 129, 2025: 138) |
| HEVI 风险链 | 818 |
| CS 索引 | 2,973 篇 |
| SS 索引 (实体/三元组) | 6,934 篇 (56,505 / 26,394) |
| LLM / 嵌入 | DeepSeek-V4-Pro / text-embedding-3-large |

---

## 重建索引

```bash
cd HippoRAG-build
python scripts/build_paper_indexes.py --source cs --corpus-path papers/processed/cs_corpus.csv --llm deepseek-v4-pro
python scripts/build_paper_indexes.py --source ss --corpus-path papers/processed/ss_corpus.csv --llm deepseek-v4-pro
```
