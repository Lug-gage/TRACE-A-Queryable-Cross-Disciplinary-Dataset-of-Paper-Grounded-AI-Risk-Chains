# GraphRAG × HEVI 风险分析

基于 Microsoft GraphRAG v3.1.0 检索增强生成 + Turner et al. (2003) HEVI 框架的自动化风险链分析实验。

## 项目结构

```
graphrag-main/
├── indices/ss/
│   ├── openie_results_ner_deepseek-v4-pro.json   # ① OpenIE 提取产物（6,934 docs）
│   ├── output/                                    # ② 索引产物（576 MB）
│   │   ├── entities.parquet                       # 41,539 实体
│   │   ├── relationships.parquet                  # 26,182 关系
│   │   ├── text_units.parquet                     # 6,934 文本块
│   │   ├── communities.parquet                    # 3,224 社区
│   │   ├── community_reports.parquet              # 3,224 社区报告（fast 模式占位）
│   │   └── lancedb/                               # 向量嵌入（558 MB）
│   └── settings.yaml                              # GraphRAG 配置
├── api_key.txt
└── hevi_query/
    ├── dataset.json                               # ③ 267 篇论文 × 818 条 chain
    ├── README.md                                  # 本文件
    ├── scripts/
    │   ├── hevi_query_graphrag.py                 # ④ 预测：GraphRAG 检索+生成二步流水线
    │   ├── eval_hevi.py                           # ⑤ 评价：covered/partial/not 语义对比
    │   ├── hevi_vuln_impact.txt                   # VI prompt 模板
    │   ├── hevi_dr.txt                            # DR prompt 模板
    │   └── eval_prompt.txt                        # 评价 prompt 模板
    ├── graph_result/                              # ⑥ 预测输出（818 个 JSON）
    └── evaluation/                                # ⑦ 评价输出（818 个 JSON）
```

## 数据流

```
论文标题+摘要 (267 篇)
    │
    ├── deepseek-v4-pro NER + 三元组 ──→ indices/ss/openie_results_ner_*.json  # OpenIE 提取
    │
    ├── build_index.py --fast ──→ 聚类/向量化 ──→ indices/ss/output/           # 索引构建
    │
    ├── dataset.json (818 chains)
    │       │ 每条: hazard + exposure + scenario + issue + reference
    │       ▼
    │   hevi_query_graphrag.py --workers 5                                    # 预测
    │       │
    │       ├─ GraphRAG local_search 检索+生成合一                            # 无显式检索步骤，
    │       │     检索上下文 = entities · relationships · text_units           # 内嵌在 local_search 中
    │       ├─ Step 1: VI 生成 → vulnerability + impact                       # Turner 定义注入 prompt
    │       ├─ 门控: VI 为空? ──✗ 跳过── ✓ Step 2                             # 防止误差传播
    │       └─ Step 2: DR 合成 → dose_response                                # causal arc
    │              │
    │              ▼ 原子写入 (.tmp → rename)
    │       graph_result/{pid}_chain{N}.json
    │
    └── eval_hevi.py --watch --expected 818                                  # 评价
            │
            ├─ 每 5s 扫描新文件 → LLM(t=0) 语义对比
            ├─ 每字段: covered / partial / not + reason
            │
            ▼ 原子写入 (.tmp → rename)
        evaluation/{pid}_chain{N}.json
            │
            818 满额 → 汇总统计 → 🛑
```
