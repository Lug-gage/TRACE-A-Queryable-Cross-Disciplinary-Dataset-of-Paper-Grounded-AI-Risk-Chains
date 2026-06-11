# HEVI 实验工作流程

## 基础信息

### 项目结构

```
HippoRAG-main/
├── indices/ss/
│   ├── openie_results_ner_deepseek-v4-pro.json           # ① OpenIE 提取产物（6,934 docs）
│   ├── llm_cache/
│   │   └── deepseek-v4-pro_cache.sqlite                   # DSPyFilter 重排序缓存
│   └── deepseek-v4-pro_text-embedding-3-large/            # ② 索引产物（1.182 GB）
│       ├── chunk_embeddings/                               # 6,934 文本块向量（154.77 MB）
│       ├── fact_embeddings/                                # 26,125 三元组向量（564.17 MB）
│       ├── entity_embeddings/                              # 22,073 实体向量（474.60 MB）
│       └── graph.pickle                                    # PPR 知识图谱（29,007 节点 / 120,469 边 / 17.20 MB）
├── api_key.txt
└── hevi_query/
    ├── dataset.json                                       # ③ 267 篇论文 × 818 条 chain
    ├── README.md
    ├── WORKFLOW.md                                        # 本文件
    ├── api_key.txt                                        # 自包含分发副本
    ├── scripts/
    │   ├── hevi_query_hipporag.py                         # ④ 预测：HippoRAG 双路检索 + LLM 二步生成
    │   ├── hevi_query_llm.py                              # ⑤ 预测：纯 LLM 基线（无 RAG）
    │   ├── eval_hevi.py                                   # ⑥ 评价：LLM-as-a-Judge（covered/partial/not）
    │   ├── eval_hevi_embedding.py                         # ⑦ 评价：Embedding 相似度（0~1，确定性）
    │   ├── analyze_joint.py                               # ⑧ 联合分布分析
    │   ├── hevi_vuln_impact.txt                           # VI prompt
    │   ├── hevi_dr.txt                                    # DR prompt
    │   └── eval_prompt.txt                                # 评价 prompt
    ├── hipporag_results/                                  # ⑨ 预测输出（HippoRAG 版）
    ├── llm_results/                                       # ⑩ 预测输出（纯 LLM 版）
    ├── evaluation/                                        # ⑪ 评价输出（LLM-as-a-Judge）
    ├── evaluation_llm/                                    # ⑫ 评价输出（LLM-as-a-Judge，纯 LLM 版）
    ├── evaluation_embedding/                              # ⑬ 评价输出（Embedding 相似度）
    └── evaluation_embedding_llm/                          # ⑭ 评价输出（Embedding 相似度，纯 LLM 版）
```

### 数据规模

| 项目 | 数值 |
|------|------|
| 论文 | 267 篇（ICML 2024: 129, ICML 2025: 138） |
| workflow chains | 818 条（全部含 hazard+exposure） |
| 知识库实体 | 22,073 |
| 知识库三元组 | 26,125 |
| 知识库文本块 | 6,934 |
| 向量维度 | 3,072（text-embedding-3-large） |
| 知识库大小 | 1.182 GB |

### 模型配置

| 组件 | 选型 |
|------|------|
| LLM | deepseek-v4-pro |
| Embedding | text-embedding-3-large（dim=3072） |
| RAG 框架 | HippoRAG 2（OSU-NLP） |
| 检索 | DPR（双路 top 3）→ DSPyFilter LLM 重排序 → PPR 图搜索 |
| 向量存储 | Parquet（cosine） |
| 图存储 | igraph（PPR） |
| 重试 | 5 次，指数退避（2s/4s/8s/16s/30s） |

### 关键参数

| 参数 | 值 |
|------|-----|
| LLM timeout | 120s |
| Embedding timeout | 60s |
| max_tokens | 384,000 |
| DPR 检索 top_k | 10 per 查询 |
| 检索合并 | ≤6 docs（两路各 top 3 去重） |
| Rerank | DSPyFilter LLM（deepseek-v4-pro） |
| max_completion_tokens | 384,000（修复硬编码 512→384000） |
| 预测 temperature | 模型默认 |
| 评价 temperature | 0（确保可复现） |

### 评价标准（LLM-as-a-Judge，三级标签）

按字段逐条语义对比，每个字段输出 `covered` / `partial` / `not`：

- **vulnerability** = SENSITIVITY：covered = 参考中的敏感性条件语义存在；partial = 相关但不同；not = 遗漏或虚构
- **impact** = CONSEQUENCE：covered = 受影响方 **AND** 损害类型均语义存在；partial = 仅一方；not = 均缺失
- **dose_response** = CAUSAL TRANSLATION ARC：covered = 因果逻辑语义一致；partial = 缺失某环节；not = 错误或颠倒

### 评价标准（Embedding 相似度，0~1 连续值）

无 prompt、无 temperature，reference 与 generated 各调 `text-embedding-3-large` 向量化后计算 cosine similarity。确定性——相同输入永远相同输出。后续可设阈值分 covered / partial / not 档位。

### 输出文件命名

```
hipporag_results/{paper_id}_chain{N}.json        # 预测（HippoRAG）
llm_results/{paper_id}_chain{N}.json              # 预测（纯 LLM）
evaluation/{paper_id}_chain{N}.json               # 评价（HippoRAG → LLM Judge）
evaluation_llm/{paper_id}_chain{N}.json           # 评价（纯 LLM → LLM Judge）
evaluation_embedding/{paper_id}_chain{N}.json     # 评价（HippoRAG → Embedding）
evaluation_embedding_llm/{paper_id}_chain{N}.json # 评价（纯 LLM → Embedding）
```

---

## 1. 检索库搭建

```
Step 1: OpenIE 提取
  论文全文 → deepseek-v4-pro NER + 三元组提取
  ↓
  indices/ss/openie_results_ner_deepseek-v4-pro.json（6,934 passages, 56,505 实体, 26,394 三元组）
  
Step 2: HippoRAG 索引构建
  读取 OpenIE JSON → 筛选实体/三元组 → text-embedding-3-large 向量化(3072d) → 构建 PPR 图
  ↓
  输出: indices/ss/deepseek-v4-pro_text-embedding-3-large/
    chunk_embeddings/   → 6,934 文本块向量（154.77 MB）
    fact_embeddings/    → 26,125 三元组向量（564.17 MB）
    entity_embeddings/  → 22,073 实体向量（474.60 MB）
    graph.pickle        → 29,007 节点 / 120,469 边（17.20 MB）
```

---

## 2. 预测

### HippoRAG 版（hevi_query_hipporag.py）

```
每条 chain：
  
  输入: {hazard, exposure, scenario, issue, title, abstract}
  
  ┌─ 检索与生成分离（防止检索上下文淹没格式约束）─┐
  │                                                    │
  │  两条聚焦查询并行检索 → 合并去重                   │
  │  he_query = "{hazard}. {exposure}."                 │
  │  si_query = "{scenario}. {issue}."                  │
  │        ↓                                            │
  │  DPR → DSPyFilter LLM 重排序 → PPR 图搜索           │
  │        ↓                                            │
  │  合并去重（he top 3 + si top 3，≤6 docs）            │
  ├────────────────────────────────────────────────────┤
  │  Step 1: VI 生成（有检索）                          │
  │    system_prompt ← Turner(2003) 定义 + ≤30词 + 断言  │
  │    user_prompt   ← 检索上下文 + 论文元数据           │
  │    → LLM → JSON{vulnerability, impact}              │
  │         │                                           │
  │         ├─ 为空 → 门控: 跳过 DR, 标记失败            │
  │         └─ 有效 → 进入 Step 2                       │
  ├────────────────────────────────────────────────────┤
  │  Step 2: DR 生成（不检索）                           │
  │    system_prompt ← Turner(2003) DR 定义 + ≤40词      │
  │    input ← {abstract, paper_impact, scenario,        │
  │             issue, hazard, exposure,                 │
  │             vulnerability, impact}                   │
  │    → LLM → JSON{dose_response}                       │
  └────────────────────────────────────────────────────┘
  
  输出: hipporag_results/{paper_id}_chain{N}.json
     {
       paper_id, chain_index, title, abstract,
       query_input: {scenario, issue, hazard, exposure},
       hipporag_result: {vulnerability, impact, dose_response},
       reference: {dose_response, vulnerability, impact}
     }
```

**并行策略：** `--workers N` → 818 条链轮询分配（`i % N`）到 N 个线程，每个线程独立 HippoRAG 实例 + LLM client。断点续做：启动时跳过已有 `.json` 的 chain。

**容错：** VI 失败跳过 DR | DR 解析失败取 raw[:300] | JSON 解析三层兜底（直接→代码块→正则）| LLM 5 次重试 + 指数退避 | DSPyFilter max_completion_tokens 修复（512→384000）

**运行命令：**
```bash
python hevi_query/scripts/hevi_query_hipporag.py --workers 5           # 5 线程并行
python hevi_query/scripts/hevi_query_hipporag.py --limit 3             # 测试
python hevi_query/scripts/hevi_query_hipporag.py --paper icml_2024_0001 # 单篇
python hevi_query/scripts/hevi_query_hipporag.py --start 0 --count 200 # 多机分片
python hevi_query/scripts/hevi_query_hipporag.py --no-kg               # 纯 LLM（不走 KG）
```

### 纯 LLM 版（hevi_query_llm.py）

与 HippoRAG 版相同框架，仅去掉检索步骤。无任何外部依赖（仅需 `openai`）。

```
每条 chain：
  
  输入: {hazard, exposure, scenario, issue, title, abstract}
  
  Step 1: VI 生成（无检索）
    input ← title + abstract + scenario + issue + hazard + exposure
    → LLM → JSON{vulnerability, impact}
         │
         ├─ 为空 → 门控: 跳过 DR, 标记失败
         └─ 有效 → 进入 Step 2
  
  Step 2: DR 生成（不检索）
    input ← abstract + paper_impact + scenario + issue
            + hazard + exposure + vulnerability + impact
    → LLM → JSON{dose_response}
  
  输出: llm_results/{paper_id}_chain{N}.json
```

**运行命令：**
```bash
python hevi_query/scripts/hevi_query_llm.py --workers 5           # 5 线程并行
python hevi_query/scripts/hevi_query_llm.py --limit 3             # 测试
python hevi_query/scripts/hevi_query_llm.py --paper icml_2024_0001 # 单篇
python hevi_query/scripts/hevi_query_llm.py --start 0 --count 200 # 多机分片
```

---

## 3. 评价

### LLM-as-a-Judge（eval_hevi.py）

```
每条预测结果：
  
  读取 results/{pid}_chain{N}.json
        ↓
  构建评价 prompt:
    上下文: abstract + paper_impact
    REFERENCE:  {ref_vulnerability, ref_impact, ref_dose_response}
    GENERATED:  {gen_vulnerability, gen_impact, gen_dose_response}
        ↓
  LLM (temperature=0) → JSON{
    vulnerability: {match: "covered"|"partial"|"not", reason: "..."},
    impact:        {match, reason},
    dose_response: {match, reason}
  }
        ↓
  原子写入 evaluation/{pid}_chain{N}.json（.tmp → rename）
  
  整体:
    全部完成后 → 汇总统计:
      vulnerability:  covered=N partial=N not=N → coverage%
      impact:         covered=N partial=N not=N → coverage%
      dose_response:  covered=N partial=N not=N → coverage%
      整体:           overall_coverage%
```

**监控模式：** `--watch` → 1 线程每 5s 扫描新结果文件，N 线程并发评价。`--expected 818` 满额自动退出。

**运行命令：**
```bash
# HippoRAG 结果
python hevi_query/scripts/eval_hevi.py --workers 3 --expected 818    # 批量
python hevi_query/scripts/eval_hevi.py --watch --workers 7 --expected 818  # 边预测边评

# 纯 LLM 结果
python hevi_query/scripts/eval_hevi.py --workers 3 --results-dir llm_results
```

### Embedding 相似度（eval_hevi_embedding.py）

```
每条预测结果：
  
  读取 results/{pid}_chain{N}.json
        ↓
  6 个文本批量 embedding（ref + gen × 3 字段）
    → text-embedding-3-large → 6 个向量
        ↓
  cosine similarity per field → 0~1 连续值（确定性）
        ↓
  原子写入 evaluation_embedding/{pid}_chain{N}.json（.tmp → rename）
  
  整体:
    全部完成后 → 分位数汇总:
      vulnerability:  min / Q25 / median / Q75 / max / mean
      impact:         同上
      dose_response:  同上
      overall:        同上
    生成柱状图 → evaluation_embedding/png/distribution.png
```

**运行命令：**
```bash
# HippoRAG 结果
python hevi_query/scripts/eval_hevi_embedding.py --workers 5

# 纯 LLM 结果
python hevi_query/scripts/eval_hevi_embedding.py --workers 5 --results-dir llm_results
```

---

## 4. 联合分布分析（analyze_joint.py）

将 LLM-as-a-Judge 三级标签与 Embedding 相似度按 `paper_id + chain_index` 配对。

```
evaluation/{pid}_chain{N}.json  +  evaluation_embedding/{pid}_chain{N}.json
        │                                    │
        └────────── 按 pid + chain 配对 ──────┘
                     │
                     ▼
  每字段: covered / partial / not 各档位的相似度分布
    → 统计表（mean / median / std / min / max per match）
    → 箱线图（boxplot.png）
    → 叠加直方图（hist_by_match.png）
    → 阈值校准建议（covered ≥ X, partial ≥ Y, not < Y）
        ↓
  analysis_joint/
```

**运行命令：**
```bash
python hevi_query/scripts/analyze_joint.py
```

---

## 5. 工作流程图

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          HEVI 实验总流程                                    │
└──────────────────────────────────────────────────────────────────────────┘

  6,934 篇 SS 安全论文 (full text)
       │
       ▼
┌──────────────────┐
│   OpenIE 提取       │  deepseek-v4-pro NER + 三元组提取
│                    │  → 6,934 docs / 56,505 实体 / 26,394 三元组
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│   HippoRAG 索引     │  text-embedding-3-large (3072d)
│   chunk(6,934)    │  → DPR 向量（3 种）+ PPR 图
│   entity(22,073)  │    29,007 节点 / 120,469 边
│   fact(26,125)    │  输出: 1.182 GB
└──────┬───────────┘
       │
       │  ┌─────────────────────────────────────┐
       │  │         dataset.json (818 chains)    │  每条: hazard, exposure, scenario,
       │  │          267 papers × 3~4 chains     │        issue + reference(VI+DR)
       │  └──────────────┬──────────────────────┘
       │                 │
       ▼                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                          预 测                                     │
│                                                                    │
│  ┌─ HippoRAG 版 ──────────────────────────────────────────────┐  │
│  │                                                              │  │
│  │  每条 chain:                                                  │  │
│  │    he_query = hazard + exposure  ──→ DPR top 3               │  │
│  │    si_query = scenario + issue    ──→ DPR top 3               │  │
│  │         │                          │                          │  │
│  │         └── DSPyFilter LLM 重排序 ──┘                          │  │
│  │                     │                                         │  │
│  │              PPR 图搜索 (graph.pickle)                         │  │
│  │                     │                                         │  │
│  │              合并去重 (≤6 docs)                                 │  │
│  │                     │                                         │  │
│  │    ┌────────────────▼────────────────┐                        │  │
│  │    │  Step 1: VI 生成 (with retrieval)│                        │  │
│  │    │   生成 vulnerability + impact   │                        │  │
│  │    └───────────────┬─────────────────┘                        │  │
│  │                    │ 门控: VI 为空?                            │  │
│  │                    ├─ YES → ✗ 跳过                             │  │
│  │                    └─ NO  → 进入 Step 2                        │  │
│  │    ┌───────────────▼─────────────────┐                        │  │
│  │    │  Step 2: DR 生成 (no retrieval) │                        │  │
│  │    │   合成 dose_response            │                        │  │
│  │    └───────────────┬─────────────────┘                        │  │
│  └────────────────────┼──────────────────────────────────────────┘  │
│                       │                                             │
│                       ▼ 原子写入 (.tmp → rename)                      │
│     hipporag_results/{pid}_chain{N}.json                             │
│                                                                     │
│  ┌─ 纯 LLM 版 ─────────────────────────────────────────────────┐   │
│  │                                                              │   │
│  │  无检索 → 仅 title + abstract + H + E + S + I                │   │
│  │  Step 1: VI → Step 2: DR（同上框架）                          │   │
│  │                       │                                      │   │
│  │                       ▼ 原子写入                               │   │
│  │     llm_results/{pid}_chain{N}.json                           │   │
│  └──────────────────────────────────────────────────────────────┘   │
└───────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                          评 价                                     │
│                                                                    │
│  ┌─ LLM-as-a-Judge ──────────────────────────────────────────┐   │
│  │  REFERENCE vs GENERATED → LLM (t=0)                       │   │
│  │  → covered / partial / not + reason                       │   │
│  │  → evaluation/ 或 evaluation_llm/                          │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                    │
│  ┌─ Embedding 相似度 ───────────────────────────────────────┐    │
│  │  REFERENCE vs GENERATED × 3 fields → 6 texts batch embed  │    │
│  │  → cosine similarity per field (0~1, 确定性)                │    │
│  │  → evaluation_embedding/                                    │    │
│  └───────────────────────────────────────────────────────────┘    │
│                            │                                       │
│                            ▼                                       │
│  ┌─ 联合分布 ───────────────────────────────────────────────┐    │
│  │  LLM 标签 × Embedding 相似度 → 箱线图 + 直方图                │    │
│  │  → analysis_joint/                                           │    │
│  └──────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```
