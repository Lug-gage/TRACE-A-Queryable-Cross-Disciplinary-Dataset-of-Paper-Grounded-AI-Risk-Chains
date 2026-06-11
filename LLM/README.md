# HEVI LLM — Hazard-Exposure-Vulnerability-Impact 纯 LLM 风险推理

基于 **Turner et al. (2003)** 脆弱性框架，从论文摘要中通过 LLM 直接推理 hazard → exposure → vulnerability → impact 的因果链条。

**无任何外部依赖（仅需 `openai`）**，不依赖知识库检索。

## 目录结构

```
hevi_llm/
├── api_key.txt
├── dataset.json
├── scripts/
│   ├── hevi_query_llm.py      # LLM 生成脚本
│   ├── eval_hevi.py           # 评测脚本
│   ├── hevi_vuln_impact.txt   # Step 1 prompt
│   ├── hevi_dr.txt            # Step 2 prompt
│   └── eval_prompt.txt        # 评测 prompt
├── llm_results/               # 生成结果（818 条 chain）
├── evaluation_llm/            # 评测结果（818 条 chain）
└── README.md
```

## 安装

```bash
pip install openai
```

## 使用方法

### 1. 生成（VI + DR）

```bash
# 全部，5 线程
python scripts/hevi_query_llm.py --workers 5

# 前 5 条测试
python scripts/hevi_query_llm.py --limit 5

# 指定论文
python scripts/hevi_query_llm.py --paper icml_2024_0001

# 多机分片
python scripts/hevi_query_llm.py --start 0 --count 200
```

结果写入 `llm_results/{paper_id}_chain{N}.json`。

### 2. 评测

```bash
# 全部，3 线程
python scripts/eval_hevi.py --workers 3

# 监控模式：1 线程扫描 + 7 线程评价（配合生成脚本边跑边评）
python scripts/eval_hevi.py --watch --workers 7

# 前 5 条测试
python scripts/eval_hevi.py --limit 5
```

结果写入 `evaluation_llm/{paper_id}_chain{N}.json`，终端自动输出覆盖率汇总。

## 推理逻辑

### Step 1: Vulnerability + Impact

| 输入 | 说明 |
|------|------|
| title | 论文标题 |
| abstract | 论文摘要 |
| scenario | 应用场景 |
| issue | 具体问题 |
| hazard | 扰动/压力源 |
| exposure | 暴露方式 |

→ 输出 vulnerability ≤30 words, impact ≤30 words（推导，非复述）

### Step 2: Dose-Response

以 Step 1 的 vulnerability + impact 为输入，加上 paper_impact 作为上下文，生成一句 ≤40 words 的因果弧线。

## 输出结构

```json
{
  "paper_id": "icml_2024_0001",
  "chain_index": 0,
  "title": "...",
  "abstract": "...",
  "impact": "...",
  "query_input": {
    "scenario": "...",
    "issue": "...",
    "hazard": ["..."],
    "exposure": ["..."]
  },
  "hipporag_result": {
    "vulnerability": "...",
    "impact": "...",
    "dose_response": "..."
  },
  "reference": {
    "dose_response": ["..."],
    "vulnerability": ["..."],
    "impact": ["..."]
  }
}
```

## 评测标准

| 字段 | covered | partial | not |
|------|---------|---------|-----|
| vulnerability (sensitivity) | 参考的敏感性条件在生成中语义存在 | 相关但不同的条件 | 完全缺失或编造 |
| impact (consequence) | 受影响方 + 损害类型均存在 | 仅一项存在 | 均不匹配 |
| dose_response (causal arc) | 相同因果逻辑 | 部分因果弧线存在 | 错误、缺失或颠倒 |

## 评测结果结构

```json
{
  "paper_id": "icml_2024_0001",
  "chain_index": 0,
  "vulnerability": {
    "match": "covered",
    "reason": "...",
    "reference": "...",
    "generated": "..."
  },
  "impact": { "match": "partial", ... },
  "dose_response": { "match": "covered", ... }
}
```

## API 配置

`api_key.txt` 需放在 `hevi_llm/` 目录下。默认使用 `deepseek-v4-pro` 模型，代理地址 `https://www.highland-api.top/v1`。

## 相关项目

- 本包切片自 [HEVI HippoRAG 版本](../hevi_query/)，去掉了 HippoRAG 知识库检索依赖
- HippoRAG 版包含双路 DPR 检索 + DSPyFilter 重排序 + PPR 图搜索，详见 `../hevi_query/README.md`
