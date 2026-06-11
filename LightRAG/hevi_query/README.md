# LightRAG × HEVI 风险分析

基于 LightRAG 检索增强生成 + Turner et al. (2003) HEVI 框架的自动化风险链分析实验。

## 项目结构

```
LightRAG-main/
├── indices/
│   ├── ss/openie_results_ner_deepseek-v4-pro.json   # ① OpenIE 提取产物（6,934 docs）
│   └── build_kg.py                                   # 知识库构建脚本
├── rag_storage_ss/                                    # ② 构建完成的知识库（1.8 GB）
│   ├── vdb_entities.json                              # 39,395 实体向量
│   ├── vdb_relationships.json                         # 25,844 关系向量
│   ├── vdb_chunks.json                                # 6,933 文本块向量
│   └── graph_chunk_entity_relation.graphml            # 知识图谱
├── api_key.txt
└── hevi_query/
    ├── dataset.json                                   # ③ 267 篇论文 × 818 条 chain
    ├── README.md                                      # 本文件
    ├── WORKFLOW.md                                    # 详细实验流程
    ├── scripts/
    │   ├── hevi_query.py                              # ④ 预测：检索 + LLM 二步流水线
    │   ├── eval_hevi.py                               # ⑤ 评价：covered/partial/not 语义对比
    │   ├── hevi_vuln_impact.txt                       # VI prompt 模板
    │   ├── hevi_dr.txt                                # DR prompt 模板
    │   └── eval_prompt.txt                            # 评价 prompt 模板
    ├── lightrag_results/                              # ⑥ 预测输出
    └── evaluation/                                    # ⑦ 评价输出
```

## 数据流

```
论文标题+摘要 (267篇)
    │
    ├── deepseek-v4-pro NER + 三元组 ──→ indices/ss/*.json           # OpenIE 提取
    │
    ├── build_kg.py ──→ 去重/映射/向量化 ──→ rag_storage_ss/         # 知识库构建
    │
    ├── dataset.json (818 chains)
    │       │ 每条: hazard + exposure + scenario + issue + reference
    │       ▼
    │   hevi_query.py --workers 5                                    # 预测
    │       │
    │       ├─ he_query + si_query 并行检索 → 合并去重                 # 检索与生成分离
    │       ├─ Step 1: VI 生成 → vulnerability + impact               # Turner 定义注入 system_prompt
    │       ├─ 门控: VI 为空? ──✗ 跳过── ✓ Step 2                     # 防止误差传播
    │       └─ Step 2: DR 合成 → dose_response                        # causal arc, 不检索
    │              │
    │              ▼ 原子写入
    │       lightrag_results/{pid}_chain{N}.json
    │
    └── eval_hevi.py --watch --workers 3 --expected 818              # 评价
            │
            ├─ 每 5s 扫描新文件 → LLM(t=0) 语义对比
            ├─ 每字段: covered / partial / not + reason
            │
            ▼ 原子写入
        evaluation/{pid}_chain{N}.json
            │
            818 满额 → 汇总统计 → 🛑
```
