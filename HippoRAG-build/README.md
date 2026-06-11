# HippoRAG - Paper Index Builder

基于 [HippoRAG 2](https://github.com/OSU-NLP-Group/HippoRAG) 的论文知识图谱索引构建工具。

## 环境

```bash
conda create -n hipporag python=3.10
conda activate hipporag
pip install -r requirements.txt
pip install -e .
```

配置 `.env`（已 gitignore）:

```
OPENAI_API_KEY=your_key
```

## 构建索引

```bash
# 全量
python scripts/build_paper_indexes.py --source cs --save-root indices
python scripts/build_paper_indexes.py --source ss --save-root indices

# 采样
python scripts/build_paper_indexes.py --source cs --save-root indices --sample-size 500
```

**输入**: `papers/processed/{cs,ss}_corpus.csv` — 预归一化的论文语料

**输出**: `indices/{cs,ss}/`，包含知识图谱 (`graph.pickle`)、向量库 (`*_embeddings/vdb_*.parquet`)、LLM 缓存

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--source` | `all` | `cs` \| `ss` |
| `--save-root` | `outputs/corpora_sample` | 索引输出目录 |
| `--sample-size` | `0` | `0`=全量，其他=采样条数 |
| `--llm-name` | `deepseek-v4-flash` | LLM 模型名 |
| `--llm-base-url` | `https://www.highland-api.top/v1` | LLM API 地址 |
| `--embedding-name` | `text-embedding-3-large` | Embedding 模型 |
| `--embedding-base-url` | `https://www.highland-api.top/v1` | Embedding API 地址 |
| `--ner-template-name` | `ner` | NER prompt 模板 |
| `--triple-template-name` | `triple_extraction` | 三元组抽取模板 |

## 注意事项

- CS 语料 ~13,500 篇，SS 语料 ~7,000 篇，全量构建耗时长、API 成本高
- LLM 缓存 `indices/*/llm_cache/`，中断后重跑不会重复计费
- 从头重建加 `--force-openie-from-scratch --force-index-from-scratch`
- `cs` 用 `ner_risk_cs` / `triple_extraction_risk_cs`，`ss` 用 `ner_risk_ss` / `triple_extraction_risk_ss`
