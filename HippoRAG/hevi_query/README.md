# HippoRAG × HEVI 风险分析

基于 OSU-NLP HippoRAG 检索增强生成 + Turner et al. (2003) HEVI 框架的自动化风险链分析实验。

## 项目结构

```
HippoRAG-main/
├── indices/ss/
│   ├── openie_results_ner_deepseek-v4-pro.json   # ① OpenIE 提取产物（6,934 篇 SS 语料）
│   ├── llm_cache/
│   │   └── deepseek-v4-pro_cache.sqlite           # DSPyFilter 重排序缓存
│   └── deepseek-v4-pro_text-embedding-3-large/    # ② 索引产物（1.2 GB）
│       ├── chunk_embeddings/                       # DPR 文本块向量
│       ├── fact_embeddings/                        # 三元组向量
│       ├── entity_embeddings/                      # 实体向量
│       └── graph.pickle                            # PPR 知识图谱
├── api_key.txt
└── hevi_query/
    ├── dataset.json                               # ③ 267 篇论文 × 818 条 chain
    ├── README.md                                  # 本文件
    ├── scripts/
    │   ├── hevi_query_hipporag.py                 # ④ 预测：HippoRAG 双路检索 + LLM 二步生成
    │   ├── hevi_query_llm.py                      # ⑤ 预测：纯 LLM 基线（无 RAG）
    │   ├── eval_hevi.py                           # ⑥ 评价：covered/partial/not 语义对比
    │   ├── hevi_vuln_impact.txt                   # VI prompt 模板
    │   ├── hevi_dr.txt                            # DR prompt 模板
    │   └── eval_prompt.txt                        # 评价 prompt 模板
    ├── hipporag_results/                          # ⑦ 预测输出（818 个 JSON）— HippoRAG 版
    ├── llm_results/                               # ⑧ 预测输出（818 个 JSON）— 纯 LLM 版
    ├── evaluation/                                # ⑨ 评价输出（818 个 JSON）— HippoRAG 版
    └── evaluation_llm/                            # ⑩ 评价输出（818 个 JSON）— 纯 LLM 版
```

## 数据流

```
6,934 篇 SS 安全论文 (full text)
    │
    ├── deepseek-v4-pro OpenIE ──→ indices/ss/openie_results_ner_*.json   # 三元组提取
    │
    ├── text-embedding-3-large ──→ 3 种向量索引 + graph.pickle            # DPR + PPR 构建
    │
    ├── dataset.json (818 chains / 267 papers)
    │       │ 每条: hazard + exposure + scenario + issue + reference
    │       ▼
    │   ┌─ hevi_query_hipporag.py --workers 5                             # 预测 (HippoRAG)
    │   │     │
    │   │     ├─ 双路 DPR 检索
    │   │     │    ├─ hazard + exposure ──→ top 3 docs
    │   │     │    └─ scenario + issue   ──→ top 3 docs (去重，≤6 docs)
    │   │     ├─ DSPyFilter LLM 重排序 (deepseek-v4-pro)
    │   │     ├─ PPR 图搜索 (graph.pickle)
    │   │     ├─ Step 1: VI 生成 → vulnerability + impact                # Turner 定义注入 prompt
    │   │     ├─ 门控: VI 为空? ──✗ 跳过── ✓ Step 2                      # 防止误差传播
    │   │     └─ Step 2: DR 生成 → dose_response                          # causal arc
    │   │            │
    │   │            ▼ 原子写入 (.tmp → rename)
    │   │     hipporag_results/{pid}_chain{N}.json
    │   │
    │   └─ hevi_query_llm.py --workers 5                                  # 预测 (纯 LLM)
    │         │
    │         ├─ 无检索，仅 paper abstract + title + scenario + issue + H + E
    │         ├─ Step 1: VI 生成 → vulnerability + impact
    │         ├─ 门控: VI 为空? ──✗ 跳过── ✓ Step 2
    │         └─ Step 2: DR 生成 → dose_response
    │                │
    │                ▼ 原子写入 (.tmp → rename)
    │         llm_results/{pid}_chain{N}.json
    │
    └── eval_hevi.py --workers 3 --results-dir <dir>                     # 评价
            │
            ├─ 每字段: covered / partial / not + reason (Turner 锚定)
            ├─ LLM(t=0) 语义对比，5 次重试 + 指数退避
            ├─ 监控模式: --watch --workers 7 (1 扫描 + 7 评价)
            │
            ▼ 原子写入 (.tmp → rename)
        evaluation/{pid}_chain{N}.json  或  evaluation_llm/{pid}_chain{N}.json
            │
            汇总统计 → 覆盖率报告
```
