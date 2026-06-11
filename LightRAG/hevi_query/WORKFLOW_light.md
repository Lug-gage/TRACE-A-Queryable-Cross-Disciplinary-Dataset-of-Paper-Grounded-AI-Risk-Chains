# HEVI 实验工作流程

## 基础信息

### 项目结构

```
LightRAG-main/
├── indices/
│   ├── ss/openie_results_ner_deepseek-v4-pro.json   # OpenIE 提取产物（6,934 docs）
│   └── build_kg.py                                   # 知识库构建脚本
├── rag_storage_ss/                                    # 构建完成的知识库（1.8 GB）
│   ├── vdb_entities.json                              # 39,395 实体向量（917 MB）
│   ├── vdb_relationships.json                         # 25,844 关系向量（601 MB）
│   ├── vdb_chunks.json                                # 6,933 文本块向量（171 MB）
│   ├── graph_chunk_entity_relation.graphml            # 知识图谱（41,539 节点 / 25,844 边）
│   └── kv_store_text_chunks.json                      # 文本块原始内容（14 MB）
├── api_key.txt
└── hevi_query/
    ├── dataset.json                                   # 267 篇论文 × 818 条 chain
    ├── README.md
    ├── WORKFLOW.md                                    # 本文件
    ├── api_key.txt                                    # 自包含分发副本
    ├── scripts/
    │   ├── hevi_query.py                              # 预测
    │   ├── eval_hevi.py                               # 评价
    │   ├── hevi_vuln_impact.txt                       # VI prompt
    │   ├── hevi_dr.txt                                # DR prompt
    │   └── eval_prompt.txt                            # 评价 prompt
    ├── lightrag_results/                              # 预测输出
    └── evaluation/                                    # 评价输出
```

### 数据规模

| 项目 | 数值 |
|------|------|
| 论文 | 267 篇（ICML 2024: 129, ICML 2025: 138） |
| workflow chains | 818 条（全部含 hazard+exposure） |
| 知识库实体 | 39,395 |
| 知识库关系 | 25,844 |
| 知识库文本块 | 6,933 |
| 向量维度 | 3,072（text-embedding-3-large） |
| 知识库大小 | 1.8 GB |

### 模型配置

| 组件 | 选型 |
|------|------|
| LLM | deepseek-v4-pro |
| Embedding | text-embedding-3-large（dim=3072） |
| RAG 框架 | LightRAG v1.5.1 |
| 向量存储 | NanoVectorDB（cosine） |
| 图存储 | NetworkX |
| 检索模式 | mix（实体 + 关系 + 文本块融合） |
| 重试 | 5 次，指数退避（2s/4s/8s/16s/30s） |

### 关键参数

| 参数 | 值 |
|------|-----|
| LLM timeout | 120s |
| Embedding timeout | 120s |
| max_tokens | 384,000 |
| 检索 top_k | 40（LightRAG 默认） |
| 检索 chunk_top_k | 6（LightRAG 默认） |
| Rerank | 关闭 |
| 文本块截断上限 | 10,000 字符（按条目边界） |
| 预测 temperature | 模型默认 |
| 评价 temperature | 0（确保可复现） |

### 评价标准（LLM-as-a-Judge，三级标签）

按字段逐条语义对比，每个字段输出 `covered` / `partial` / `not`：

- **vulnerability** = SENSITIVITY：covered = 参考中的敏感性条件语义存在；partial = 相关但不同；not = 遗漏或虚构
- **impact** = CONSEQUENCE：covered = 受影响方 **AND** 损害类型均语义存在；partial = 仅一方；not = 均缺失
- **dose_response** = CAUSAL TRANSLATION ARC：covered = 因果逻辑语义一致；partial = 缺失某环节；not = 错误或颠倒

### 输出文件命名

```
lightrag_results/{paper_id}_chain{N}.json    # 预测
evaluation/{paper_id}_chain{N}.json          # 评价
```

---

## 1. 检索库搭建

```
Step 1: OpenIE 提取
  论文标题+摘要 → deepseek-v4-pro NER + 三元组提取
  ↓
  indices/ss/openie_results_ner_deepseek-v4-pro.json（6,934 passages, 56,505 实体, 26,394 三元组）
  
Step 2: 数据转换（indices/build_kg.py → load_indices_as_custom_kg）
  读取 JSON → 按 (名称)/(主语,关系,宾语) 去重 → 映射为 LightRAG 格式
  chunks:       [{content: "标题+摘要", source_id: chunk_id}, ...]
  entities:     [{entity_name, entity_type:"CONCEPT", description, source_id}, ...]
  relationships:[{src_id, tgt_id, description/keywords, weight:1.0, source_id}, ...]
  
Step 3: 向量化 + 图构建（LightRAG ainsert_custom_kg）
  批量插入 → text-embedding-3-large 向量化(3072d) → 存入 3 个 NanoVectorDB + 1 个 NetworkX 图
  输出: rag_storage_ss/（vdb_entities + vdb_relationships + vdb_chunks + graph）
```

**执行命令：**
```bash
python indices/build_kg.py ss
```

---

## 2. 预测

```
每条 chain：
  
  输入: {hazard, exposure, scenario, issue, title, abstract}
  
  ┌─ 检索与生成分离（防止检索上下文淹没格式约束）─┐
  │                                                    │
  │  两条聚焦查询并行检索 → 合并去重                   │
  │  he_query = "{hazard}. {exposure}."                 │
  │  si_query = "{scenario}. {issue}."                  │
  │        ↓                                            │
  │  aquery_data() × 2 → 合并实体/关系/文本块去重       │
  │        ↓                                            │
  │  按条目边界截断（≤10,000 字符）                     │
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
  │    input ← {hazard,exposure,vulnerability,impact,    │
  │             title,abstract}                          │
  │    → LLM → JSON{dose_response}                       │
  └────────────────────────────────────────────────────┘
  
  输出: lightrag_results/{paper_id}_chain{N}.json
     {
       paper_id, chain_index, title, abstract,
       query_input: {scenario, issue, hazard, exposure},
       lightrag_result: {vulnerability, impact, dose_response},
       reference: {dose_response, vulnerability, impact}
     }
```

**并行策略：** `--workers N` → 818 条链轮询分配（`i % N`）到 N 个协程，共用同一 LightRAG 实例。断点续做：启动时跳过已有 `.json` 的 chain。

**容错：** 检索超时退回纯 LLM | VI 失败跳过 DR | DR 解析失败取 raw[:300] | JSON 解析三层兜底（直接→代码块→正则）

**运行命令：**
```bash
python hevi_query/scripts/hevi_query.py --workers 5           # 5 协程并行
python hevi_query/scripts/hevi_query.py --limit 3             # 测试
python hevi_query/scripts/hevi_query.py --paper icml_2024_0001 # 单篇
python hevi_query/scripts/hevi_query.py --no-kg               # 纯 LLM
```

---

## 3. 评价

```
每条预测结果：
  
  读取 lightrag_results/{pid}_chain{N}.json
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
python hevi_query/scripts/eval_hevi.py --workers 3 --expected 818    # 批量
python hevi_query/scripts/eval_hevi.py --watch --workers 3 --expected 818  # 边预测边评
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
│   (build_kg.py)   │  → openie_results_ner_deepseek-v4-pro.json
└──────┬───────────┘
       │  Load + 去重 + 映射
       ▼
┌──────────────────┐
│   LightRAG        │  ainsert_custom_kg()
│   向量化 + 图构建   │  → text-embedding-3-large (3072d)
└──────┬───────────┘  → NanoVectorDB × 3 + NetworkX × 1
       │
       ▼
┌──────────────────┐
│  rag_storage_ss/  │  39,395 entities | 25,844 relations | 6,933 chunks | 1.8 GB
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
│    he_query = hazard + exposure                       │
│    si_query = scenario + issue                        │
│         │                    │                        │
│         ▼                    ▼                        │
│    aquery_data()        aquery_data()                 │
│         │                    │                        │
│         └──── 合并去重 ──────┘                        │
│                  │                                    │
│                  ▼ (检索上下文, ≤10K chars)            │
│    ┌─────────────────────────────┐                    │
│    │  Step 1: VI 生成 (with KG)   │                    │
│    │   生成 vulnerability+impact │                    │
│    └─────────────┬───────────────┘                    │
│                  │ 门控: VI 为空?                      │
│                  ├─ YES → ✗ 跳过                       │
│                  └─ NO  → 进入 Step 2                  │
│                  │                                    │
│    ┌─────────────▼───────────────┐                    │
│    │  Step 2: DR 生成 (no KG)    │                    │
│    │   合成 dose_response        │                    │
│    └─────────────┬───────────────┘                    │
│                  │                                    │
│                  ▼ 原子写入 (.tmp → rename)            │
│    lightrag_results/{pid}_chain{N}.json                │
│    {vulnerability, impact, dose_response, reference}  │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│                    评 价     --watch                   │
│                                                      │
│  每 5s 扫描 lightrag_results/                          │
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
