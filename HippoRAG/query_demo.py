"""
基于预建 HippoRAG 索引进行检索问答

用法:
    python query_demo.py                          # 交互式查询 ss 知识库
    python query_demo.py --index cs               # 查询 cs 知识库
    python query_demo.py --query "你的问题"         # 单次查询
"""
import os
import sys
import json
import argparse

# 读取 API Key
API_KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_key.txt")
with open(API_KEY_FILE) as f:
    os.environ["OPENAI_API_KEY"] = f.read().strip()

BASE_URL = "https://www.highland-api.top/v1"
LLM_MODEL = "deepseek-v4-pro"
EMBEDDING_MODEL = "text-embedding-3-large"

from src.hipporag import HippoRAG
from src.hipporag.utils.config_utils import BaseConfig


def load_index(index_name: str = "ss") -> HippoRAG:
    """加载预建的 HippoRAG 索引"""
    index_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indices", index_name)

    config = BaseConfig(
        save_dir=index_dir,
        llm_base_url=BASE_URL,
        llm_name=LLM_MODEL,
        embedding_model_name=EMBEDDING_MODEL,
        embedding_base_url=BASE_URL,
        force_index_from_scratch=False,
        force_openie_from_scratch=False,
        retrieval_top_k=10,
        linking_top_k=5,
        qa_top_k=5,
        max_qa_steps=1,
        max_new_tokens=384000,
        openie_mode="online",
    )

    hipporag = HippoRAG(global_config=config)
    print(f"已加载索引 [{index_name}]: {hipporag.get_graph_info()}")
    return hipporag


def main():
    parser = argparse.ArgumentParser(description="HippoRAG 预建索引查询")
    parser.add_argument("--index", "-i", default="ss", choices=["ss", "cs"], help="选择知识库 (默认 ss)")
    parser.add_argument("--query", "-q", type=str, default=None, help="单次查询")
    parser.add_argument("--limit", "-l", type=int, default=5, help="返回文档数 (默认 5)")
    args = parser.parse_args()

    print(f"正在加载 [{args.index}] 知识库...")
    hipporag = load_index(args.index)

    if args.query:
        queries = [args.query]
    else:
        print("\n输入查询（输入 quit 退出）:")
        queries = []
        while True:
            try:
                q = input("> ").strip()
                if q.lower() in ("quit", "exit", "q"):
                    break
                if q:
                    queries.append(q)
            except (EOFError, KeyboardInterrupt):
                break

    if not queries:
        print("没有查询，退出。")
        return

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        # 检索 + QA
        results = hipporag.rag_qa(queries=[query])

        print(f"\n答案: {results[0][0].answer}")
        print(f"\n相关文档 (Top {args.limit}):")
        for i, doc in enumerate(results[0][0].docs[:args.limit]):
            print(f"  [{i+1}] {doc[:150]}...")


if __name__ == "__main__":
    main()
