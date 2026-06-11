# HEVI 实验工作流程

## 基础信息

### 项目结构

```
graphrag-main/
├── indices/ss/
│   ├── openie_results_ner_deepseek-v4-pro.json   # OpenIE 提取产物（6,934 docs）
│   ├── output/                                    # 索引产物（576 MB）
│   │   ├── entities.parquet                       # 41,539 实体
│   │   ├── relationships.parquet                  # 26,182 关系
│   │   ├── communities.parquet                    # 3,224 社区
│   │   ├── community_reports.parquet              # 3,224 社区报告（fast 模式占位）
│   │   ├── text_units.parquet                     # 6,934 文本块
│   │   ├── documents.parquet                      # 6,934 文档
│   │   └── lancedb/                               # 向量嵌入（559 MB）
│   └── settings.yaml                              # GraphRAG 配置
├── api_key.txt
└── hevi_query/
    ├── dataset.json                               # 267 篇论文 × 818 条 chain
    ├── README.md
    ├── WORKFLOW.md                                # 本文件
    ├── scripts/
    │   ├── hevi_query_graphrag.py                 # 预测
    │   ├── eval_hevi.py                           # 评价（LLM-as-a-Judge）
    │   ├── eval_hevi_embedding.py                 # 评价（Embedding 相似度）
    │   ├── hevi_vuln_impact.txt                   # VI prompt
    │   ├── hevi_dr.txt                            # DR prompt
    │   └── eval_prompt.txt                        # 评价 prompt
    ├── graph_result/                              # 预测输出（818 个 JSON）
    └── evaluation/                                # 评价输出（818 个 JSON）
```

### 数据规模

| 项目 | 数值 |
|------|------|
| 论文 | 267 篇（ICML 2024: 129, ICML 2025: 138） |
| workflow chains | 818 条（全部含 hazard+exposure） |
| 知识库实体 | 41,539 |
| 知识库关系 | 26,182 |
| 知识库文本块 | 6,934 |
| 向量维度 | 3,072（text-embedding-3-large） |
| 知识库大小 | 576 MB（含 559 MB 向量） |

### 模型配置

| 组件 | 选型 |
|------|------|
| LLM | deepseek-v4-pro |
| Embedding | text-embedding-3-large（dim=3072） |
| RAG 框架 | Microsoft GraphRAG v3.1.0 |
| 向量存储 | LanceDB（cosine） |
| 索引格式 | Parquet（结构化数据）+ LanceDB（向量数据） |
| 检索模式 | local_search（实体 + 关系 + 文本块融合） |
| 并发请求 | 5（settings.yaml concurrent_requests） |
| 社区上下文占比 | 0%（community_prop: 0.0，禁用社区报告） |
| 重试 | 5 次，指数退避（2s/4s/8s/16s/30s） |

### 关键参数

| 参数 | 值 |
|------|-----|
| LLM timeout | 120s |
| Embedding timeout | 120s |
| max_tokens | 384,000 |
| 预测 temperature | 模型默认 |
| 评价 temperature | 0（确保可复现） |
| GraphRAG community_level | 0（最细粒度） |
| GraphRAG response_type | "multiple paragraphs" |

### 评价标准

**LLM-as-a-Judge（三级标签）** — 按字段逐条语义对比，每个字段输出 `covered` / `partial` / `not`：

- **vulnerability** = SENSITIVITY：covered = 参考中的敏感性条件语义存在；partial = 相关但不同；not = 遗漏或虚构
- **impact** = CONSEQUENCE：covered = 受影响方 **AND** 损害类型均语义存在；partial = 仅一方；not = 均缺失
- **dose_response** = CAUSAL TRANSLATION ARC：covered = 因果逻辑语义一致；partial = 缺失某环节；not = 错误或颠倒

**Embedding 相似度（确定性）** — text-embedding-3-large 对 reference 和 generated 分别向量化，计算 cosine similarity（0~1），输出五数分布 + 0.1 分段柱状图。

### 输出文件命名

```
graph_result/{paper_id}_chain{N}.json        # 预测
evaluation/{paper_id}_chain{N}.json          # 评价（LLM）
evaluation_embedding/{paper_id}_chain{N}.json # 评价（Embedding）
```

---

## 1. 检索库搭建

```
Step 1: OpenIE 提取
  论文标题+摘要 → deepseek-v4-pro NER + 三元组提取
  ↓
  indices/ss/openie_results_ner_deepseek-v4-pro.json（6,934 passages）

Step 2: 索引构建（build_index.py --fast）
  读取 JSON → 实体聚类 → 关系映射 → 向量化
  entities:       41,539（含 title, type, description, text_unit_ids, degree）
  relationships:  26,182（含 source, target, description, weight, combined_degree）
  text_units:      6,934（含 text, n_tokens, document_id, entity_ids）
  communities:     3,224（聚类结果，含 entity_ids, level, parent/children 层级）

Step 3: 向量化 + 存储
  text-embedding-3-large 向量化(3072d) → LanceDB
  结构化数据 → Parquet
  社区报告 → 占位（fast 模式跳过 LLM 生成）
  输出: indices/ss/output/（7 个 .parquet + lancedb/ 向量目录）
```

**执行命令：**
```bash
uv run python build_index.py ss --fast
```

---

## 2. 预测

```
每条 chain：

  输入: {hazard, exposure, scenario, issue, title, abstract}

  ┌─ 检索+生成合一（GraphRAG local_search）──────────────┐
  │                                                        │
  │  VI/DR prompt 直接作为 query_text 传入 local_search()  │
  │  GraphRAG 内部自动完成:                                │
  │    · 向量检索 → 匹配相关 entities                       │
  │    · 关系扩展 → 跟踪 relationships                     │
  │    · 文本匹配 → 关联 text_units                         │
  │    · 上下文混合 → entity + relationship + text_unit     │
  │    → LLM 生成 → 返回完整响应                            │
  ├────────────────────────────────────────────────────────┤
  │  Step 1: VI 生成（有检索）                              │
  │    query ← hevi_vuln_impact.txt + 论文+chain 各字段     │
  │    → local_search() → JSON{vulnerability, impact}      │
  │         │                                               │
  │         ├─ 为空 → 门控: 跳过 DR, 标记失败                │
  │         └─ 有效 → 进入 Step 2                           │
  ├────────────────────────────────────────────────────────┤
  │  Step 2: DR 生成（有检索）                              │
  │    query ← hevi_dr.txt + VI 结果 + 论文上下文            │
  │    → local_search() → JSON{dose_response}              │
  └────────────────────────────────────────────────────────┘

  输出: graph_result/{paper_id}_chain{N}.json
     {
       paper_id, chain_index, title, abstract, impact,
       query_input: {scenario, issue, hazard, exposure},
       graphrag_result: {vulnerability, impact, dose_response},
       reference: {dose_response, vulnerability, impact}
     }
```

**并行策略：** `--workers N` → 818 条链轮询分配（`i % N`）到 N 个线程，主线程加载一次 GraphRAG 配置/数据，各 worker 共享。断点续做：启动时跳过已有 `.json` 的 chain。

**容错：** VI/DR 查询失败 → 跳过，下次重跑补 | DR 解析失败取 raw[:300] | JSON 解析三层兜底（直接→代码块→正则） | `--no-kg` 模式绕过 GraphRAG，直接 OpenAI 直调。

**运行命令：**
```bash
uv run python hevi_query/scripts/hevi_query_graphrag.py --workers 5           # 5 线程并行
uv run python hevi_query/scripts/hevi_query_graphrag.py --limit 3             # 测试
uv run python hevi_query/scripts/hevi_query_graphrag.py --paper icml_2024_0001 # 单篇
uv run python hevi_query/scripts/hevi_query_graphrag.py --no-kg               # 纯 LLM
uv run python hevi_query/scripts/hevi_query_graphrag.py --mode global         # global 模式
```

---

## 3. 评价

### 3.1 LLM-as-a-Judge

```
每条预测结果：

  读取 graph_result/{pid}_chain{N}.json
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

**运行命令：**
```bash
uv run python hevi_query/scripts/eval_hevi.py --workers 3 --expected 818    # 批量
uv run python hevi_query/scripts/eval_hevi.py --watch --expected 818        # 边预测边评
```

### 3.2 Embedding 相似度

```
每条预测结果：

  读取 graph_result/{pid}_chain{N}.json
        ↓
  6 个文本批量 embedding（ref+gen × 3 字段）→ text-embedding-3-large
        ↓
  cosine similarity × 3 → {vulnerability: sim, impact: sim, dose_response: sim}
        ↓
  原子写入 evaluation_embedding/{pid}_chain{N}.json（.tmp → rename）

  整体:
    全部完成后 → 汇总统计:
      五数分布: min / Q25 / median / Q75 / max / mean
      0.1 分段柱状图 → evaluation_embedding/png/distribution.png
```

**运行命令：**
```bash
uv run python hevi_query/scripts/eval_hevi_embedding.py --workers 5
uv run python hevi_query/scripts/eval_hevi_embedding.py --limit 5            # 测试
```

---

## 4. 工作流程图

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          HEVI 实验总流程                                    │
└──────────────────────────────────────────────────────────────────────────┘

  论文标题+摘要 (267篇)
       │
       ▼
┌──────────────────┐
│   索引构建          │  deepseek-v4-pro NER + 三元组提取
│   (build_index.py) │  → openie_results_ner_deepseek-v4-pro.json
└──────┬───────────┘
       │  聚类 + 向量化
       ▼
┌──────────────────┐
│   GraphRAG        │  text-embedding-3-large (3072d)
│   向量化 + 图构建   │  → 7× Parquet + LanceDB
└──────┬───────────┘  → 9.6 向量索引, 嵌入 41,539 实体
       │
       ▼
┌──────────────────┐
│  indices/ss/output│  entities: 41,539 | relations: 26,182
│  576 MB (559 MB   │  text_units: 6,934 | communities: 3,224
│  lancedb 向量)    │  community_reports: 占位（社区上下文禁用）
└──────┬───────────┘
       │
       │  ┌─────────────────────────────────────┐
       │  │         dataset.json (818 chains)    │  每条: hazard, exposure, scenario,
       │  │          267 papers × 3~4 chains     │        issue + reference(VI+DR)
       │  └──────────────┬──────────────────────┘
       │                 │
       ▼                 ▼
┌──────────────────────────────────────────────────────┐
│                    预 测                              │
│                                                      │
│  每条 chain:                                          │
│    VI prompt ← hevi_vuln_impact.txt + 论文+chain       │
│    → local_search() 检索+生成合一                      │
│         │  检索: entities · relationships · text_units  │
│         │  社区上下文: community_prop=0.0 → 不参与     │
│         ▼                                             │
│    ┌─────────────────────────────┐                    │
│    │  Step 1: VI 生成 (with KG)   │                    │
│    │   生成 vulnerability+impact │                    │
│    └─────────────┬───────────────┘                    │
│                  │ 门控: VI 为空?                      │
│                  ├─ YES → ✗ 跳过                       │
│                  └─ NO  → 进入 Step 2                  │
│                  │                                    │
│    ┌─────────────▼───────────────┐                    │
│    │  Step 2: DR 生成 (with KG)  │                    │
│    │   DR prompt ← hevi_dr.txt   │                    │
│    │   → local_search()          │                    │
│    │   合成 dose_response         │                    │
│    └─────────────┬───────────────┘                    │
│                  │                                    │
│                  ▼ 原子写入 (.tmp → rename)            │
│    graph_result/{pid}_chain{N}.json                    │
│    {vulnerability, impact, dose_response, reference}  │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│                    评 价     --watch                   │
│                                                      │
│  每 5s 扫描 graph_result/                              │
│    → 发现新 .json → 立刻评价                           │
│         │                                             │
│         ▼                                             │
│    LLM (t=0): REFERENCE vs GENERATED                  │
│    → {vulnerability, impact, dose_response}           │
│      每字段: covered / partial / not + reason          │
│         │                                             │
│         ▼ 原子写入                                     │
│    evaluation/{pid}_chain{N}.json                      │
│                                                      │
│  818 条满额 → 汇总统计 → 🛑 自动退出                    │
└──────────────────────────────────────────────────────┘
```

## 5. 与 LightRAG 版的架构差异

| | LightRAG 版 | GraphRAG 版（本项目） |
|---|---|---|
| **检索** | `aquery_data()` 显式调用，返回文档列表 | `local_search()` 内嵌，检索+生成合一 |
| **查询构造** | `he_query()` + `si_query()` 两条独立检索，手动合并 | VI/DR prompt 直接作为 query，GraphRAG 内部检索 |
| **模型配置** | LLM 参数散落在 Python 代码中 | 集中在 `settings.yaml`，`load_config()` 统一加载 |
| **向量存储** | NanoVectorDB × 3 | LanceDB（实体描述嵌入） |
| **图存储** | NetworkX | Parquet DataFrame + LanceDB |
| **知识库构建** | `build_kg.py` → `ainsert_custom_kg()` | `build_index.py --fast` |
| **社区上下文** | 无 | 可选（当前 community_prop=0.0，禁用） |
| **并发控制** | Python ThreadPoolExecutor | settings.yaml `concurrent_requests` + ThreadPoolExecutor |
