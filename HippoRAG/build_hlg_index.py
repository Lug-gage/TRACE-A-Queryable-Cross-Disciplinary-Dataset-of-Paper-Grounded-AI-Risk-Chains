"""
将 hlg/ 目录中的层级知识图谱 JSON 向量化建 HippoRAG 索引

每个 JSON 文件对应一篇论文的知识图谱，包含:
  - Level1/2/3: 具体概念 → 中层概念 → 高层主题
  - Relations: 概念间关系 (source, target, relation, explanation)
  - overall_explanation: 整体分析

输出目录: indices/<index_name>/

用法:
    # 1. 默认建库 (hlg/ -> indices/hlg/)
    python build_hlg_index.py

    # 2. 指定数据源和索引名
    python build_hlg_index.py --source-dir hlg_9336 --index-name hlg_9336

    # 3. 建库 + 测试查询
    python build_hlg_index.py --source-dir hlg_9336 --index-name hlg_9336 --query "fake news detection"

    # 4. 仅查询已建好的库
    python build_hlg_index.py --index-name hlg --query-only "adversarial attacks on NLP"
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List

# ---- 日志 ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# ---- API Key ----
API_KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_key.txt")
with open(API_KEY_FILE) as f:
    os.environ["OPENAI_API_KEY"] = f.read().strip()

BASE_URL = "https://www.highland-api.top/v1"
LLM_MODEL = "deepseek-v4-pro"
EMBEDDING_MODEL = "text-embedding-3-large"

from src.hipporag import HippoRAG
from src.hipporag.utils.config_utils import BaseConfig

# ---- 路径 ----
PROJECT_ROOT = Path(__file__).parent


def load_hlg_json_files(source_dir: Path) -> List[dict]:
    """加载所有 *_hlg.json 文件"""
    json_files = sorted(source_dir.glob("*_hlg.json"))
    papers = []
    for f in json_files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            data["_filename"] = f.stem.replace("_hlg", "")
            papers.append(data)
        except Exception as e:
            print(f"  [跳过] {f.name}: {e}")
    return papers


def build_document(paper: dict) -> str:
    """
    将一篇论文的 HLG JSON 构造为适合 HippoRAG 索引的自然语言文档。

    策略：将三层概念 + 关系解释组合成连贯文本，
          让 OpenIE 能从中提取实体和三元组。
    """
    name = paper.get("_filename", "Unknown")
    level1 = paper.get("Level1", [])
    level2 = paper.get("Level2", [])
    level3 = paper.get("Level3", [])
    relations = paper.get("Relations", [])
    overall_explanation = paper.get("overall_explanation", "")

    parts = [f"Paper: {name}"]

    # 概念层级
    if level1:
        parts.append("Key Concepts: " + "; ".join(level1) + ".")
    if level2:
        parts.append("Mid-level Themes: " + "; ".join(level2) + ".")
    if level3:
        parts.append("High-level Topics: " + "; ".join(level3) + ".")

    # 关系解释（自然语言核心）
    if relations:
        rel_lines = []
        for r in relations:
            src = r.get("source", "")
            tgt = r.get("target", "")
            rel = r.get("relation", "")
            exp = r.get("explanation", "")
            confidence = r.get("confidence", 0)
            rel_lines.append(
                f"Relation: \"{src}\" {rel} \"{tgt}\" (confidence={confidence}). "
                f"Explanation: {exp}"
            )
        parts.append("Relations:\n" + "\n".join(rel_lines))

    # 整体解释
    if overall_explanation:
        parts.append(f"Overall Analysis: {overall_explanation}")

    return "\n\n".join(parts)


def build_documents(papers: List[dict]) -> List[str]:
    """将所有论文 JSON 转为文档字符串列表"""
    return [build_document(p) for p in papers]


def create_index(papers: List[dict], index_dir: Path, source_label: str):
    """从 hlg JSON 构建 HippoRAG 索引"""
    print(f"\n{'='*60}")
    print(f"从 {source_label} 目录加载了 {len(papers)} 篇论文 HLG JSON")
    print(f"索引输出目录: {index_dir}")
    print(f"{'='*60}\n")

    docs = build_documents(papers)
    print(f"构造了 {len(docs)} 条文档，总长度 {sum(len(d) for d in docs):,} 字符")

    config = BaseConfig(
        save_dir=str(index_dir),
        llm_base_url=BASE_URL,
        llm_name=LLM_MODEL,
        embedding_model_name=EMBEDDING_MODEL,
        embedding_base_url=BASE_URL,
        force_index_from_scratch=True,   # 首次建库从头开始
        force_openie_from_scratch=True,
        retrieval_top_k=20,
        linking_top_k=5,
        qa_top_k=5,
        max_new_tokens=384000,
        openie_mode="online",
        save_openie=True,
    )

    hipporag = HippoRAG(global_config=config)

    print("\n开始索引 (OpenIE + Embedding + Graph Construction)...")
    hipporag.index(docs=docs)

    print("\n索引完成! 图谱信息:")
    info = hipporag.get_graph_info()
    for k, v in info.items():
        print(f"  {k}: {v}")

    return hipporag


def load_index(index_dir: Path) -> HippoRAG:
    """加载已构建的索引"""
    if not (index_dir / "deepseek-v4-pro_text-embedding-3-large").exists():
        print(f"索引不存在: {index_dir}")
        sys.exit(1)

    config = BaseConfig(
        save_dir=str(index_dir),
        llm_base_url=BASE_URL,
        llm_name=LLM_MODEL,
        embedding_model_name=EMBEDDING_MODEL,
        embedding_base_url=BASE_URL,
        force_index_from_scratch=False,
        force_openie_from_scratch=False,
        retrieval_top_k=20,
        linking_top_k=5,
        qa_top_k=5,
        max_new_tokens=384000,
        openie_mode="online",
    )

    hipporag = HippoRAG(global_config=config)
    print(f"已加载索引: {hipporag.get_graph_info()}")
    return hipporag


def query(hipporag: HippoRAG, queries: List[str], top_k: int = 10):
    """对索引执行查询"""
    for q in queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print(f"{'='*60}")

        results = hipporag.rag_qa(queries=[q])
        qs = results[0][0]

        print(f"\n答案: {qs.answer}")
        print(f"\n相关文档 (Top {top_k}):")
        for i, doc in enumerate(qs.docs[:top_k]):
            preview = doc[:200].replace("\n", " ")
            print(f"  [{i+1}] {preview}...")


def main():
    parser = argparse.ArgumentParser(description="HLG JSON -> HippoRAG 索引构建与查询")
    parser.add_argument("--source-dir", "-s", type=str, default="hlg",
                        help="HLG JSON 数据目录 (默认 hlg)")
    parser.add_argument("--index-name", "-n", type=str, default=None,
                        help="索引名称，输出到 indices/<name>/ (默认与 --source-dir 同名)")
    parser.add_argument("--query", "-q", type=str, default=None,
                        help="建库后执行一次查询")
    parser.add_argument("--query-only", type=str, default=None,
                        help="仅查询已有索引，不重建")
    parser.add_argument("--rebuild", action="store_true",
                        help="强制重建索引（清空已有数据）")
    parser.add_argument("--top-k", "-k", type=int, default=10,
                        help="查询返回文档数")
    args = parser.parse_args()

    source_dir = PROJECT_ROOT / args.source_dir
    index_name = args.index_name or args.source_dir
    index_dir = PROJECT_ROOT / "indices" / index_name

    # 纯查询模式
    if args.query_only:
        hipporag = load_index(index_dir)
        query(hipporag, [args.query_only], top_k=args.top_k)
        return

    # 建库模式
    papers = load_hlg_json_files(source_dir)
    if not papers:
        print(f"错误: {source_dir} 中没有找到 *_hlg.json 文件")
        sys.exit(1)

    # 如果索引已存在且不强制重建
    index_exists = (index_dir / "deepseek-v4-pro_text-embedding-3-large").exists()
    if index_exists and not args.rebuild:
        print("索引已存在。使用 --rebuild 强制重建，或用 --query-only 直接查询。")
        hipporag = load_index(index_dir)
    else:
        if args.rebuild:
            import shutil
            if index_dir.exists():
                shutil.rmtree(index_dir)
                print("已清空旧索引。")
        hipporag = create_index(papers, index_dir, args.source_dir)

    # 查询
    if args.query:
        query(hipporag, [args.query], top_k=args.top_k)


if __name__ == "__main__":
    main()
