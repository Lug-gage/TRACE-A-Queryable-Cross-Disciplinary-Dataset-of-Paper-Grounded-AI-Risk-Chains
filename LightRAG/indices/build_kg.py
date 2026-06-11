"""
LightRAG 知识图谱构建脚本
用法:
    python indices/build_kg.py cs          # 只建 CS
    python indices/build_kg.py ss          # 只建 SS
    python indices/build_kg.py both        # 两个都建（不同目录）

    python build_kg.py cs --query          # 建完后进入交互查询
    python build_kg.py both --skip-import  # 跳过导入，直接查询
    python build_kg.py cs --mode global    # 指定查询模式
"""
import json
import asyncio
import argparse
from functools import partial
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

# ===== 配置 =====
API_KEY = "sk-OZZbAcjqc9JNOeW9JTTtuluJuXpq4Djf8urFyyMW9r6OqcJL"
BASE_URL = "https://www.highland-api.top/v1"
LLM_MODEL = "deepseek-v4-pro"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072

# 每个索引对应的配置
KG_CONFIGS = {
    "cs": {
        "json_path": "indices/cs/openie_results_ner_deepseek-v4-pro.json",
        "working_dir": "./rag_storage_cs",
        "label": "Computer Science",
    },
    "ss": {
        "json_path": "indices/ss/openie_results_ner_deepseek-v4-pro.json",
        "working_dir": "./rag_storage_ss",
        "label": "Social Science",
    },
}

# 查询模式选项
QUERY_MODES = ["local", "global", "hybrid", "naive", "mix"]


# ===== 1. 数据转换 =====
def load_indices_as_custom_kg(json_path: str) -> dict:
    with open(json_path) as f:
        data = json.load(f)

    chunks, entities, relationships = [], [], []
    seen_entities = set()
    seen_relations = set()

    for doc in data["docs"]:
        chunk_id = doc["idx"]
        title = doc["passage"]["title"]
        abstract = doc["passage"]["abstract"]

        chunks.append({
            "content": f"{title}\n{abstract}",
            "source_id": chunk_id,
        })

        for ent in doc["extracted_entities"]:
            if ent not in seen_entities:
                seen_entities.add(ent)
                entities.append({
                    "entity_name": ent,
                    "entity_type": "CONCEPT",
                    "description": f"From: {title}",
                    "source_id": chunk_id,
                })

        for s, r, o in doc["extracted_triples"]:
            rel_key = (s, r, o)
            if rel_key not in seen_relations:
                seen_relations.add(rel_key)
                relationships.append({
                    "src_id": s,
                    "tgt_id": o,
                    "description": r,
                    "keywords": r,
                    "weight": 1.0,
                    "source_id": chunk_id,
                })

    return {"chunks": chunks, "entities": entities, "relationships": relationships}


# ===== 2. LLM 函数 =====
async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await openai_complete_if_cache(
        LLM_MODEL, prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY, base_url=BASE_URL,
        max_tokens=384000,
        **kwargs,
    )


# ===== 3. 创建 RAG 实例 =====
def make_rag(working_dir: str) -> LightRAG:
    return LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_func,
        embedding_func_max_async=1,   # 单 worker，避免 API 限流
        embedding_batch_num=50,       # 每批 50 条，减少 API 请求次数
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=8192,
            func=partial(
                openai_embed.func,
                model=EMBEDDING_MODEL,
                api_key=API_KEY,
                base_url=BASE_URL,
            ),
        ),
    )


# ===== 4. 建库 =====
async def build_kg(kg_name: str):
    cfg = KG_CONFIGS[kg_name]
    rag = make_rag(cfg["working_dir"])
    await rag.initialize_storages()

    print(f"Loading {cfg['json_path']} ...")
    kg = load_indices_as_custom_kg(cfg["json_path"])
    print(f"  [{cfg['label']}] chunks: {len(kg['chunks'])}, "
          f"entities: {len(kg['entities'])}, "
          f"relationships: {len(kg['relationships'])}")

    print(f"Inserting into {cfg['working_dir']} (纯存储，不调用LLM)...")
    await rag.ainsert_custom_kg(kg)

    await rag.finalize_storages()
    print(f"[{cfg['label']}] 建库完成.\n")


# ===== 5. 交互查询 =====
async def query_loop(kg_name: str, mode: str = "mix"):
    cfg = KG_CONFIGS[kg_name]
    rag = make_rag(cfg["working_dir"])
    await rag.initialize_storages()

    print(f"\n{'='*50}")
    print(f"查询库: [{cfg['label']}]  |  模式: {mode}")
    print(f"输入问题回车查询，输入 :mode <模式名> 切换模式")
    print(f"可用模式: {', '.join(QUERY_MODES)}")
    print(f"输入 :cs / :ss 切换库，输入 :quit 退出")
    print(f"{'='*50}\n")

    current_mode = mode
    current_kg = kg_name

    while True:
        try:
            user_input = input(f"[{current_kg}:{current_mode}] >>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue

        # 命令处理
        if user_input == ":quit":
            break
        elif user_input.startswith(":mode"):
            parts = user_input.split()
            if len(parts) > 1 and parts[1] in QUERY_MODES:
                current_mode = parts[1]
                print(f"  切换到模式: {current_mode}")
            else:
                print(f"  可用模式: {', '.join(QUERY_MODES)}")
        elif user_input in (":cs", ":ss"):
            if user_input[1:] != current_kg:
                current_kg = user_input[1:]
                await rag.finalize_storages()
                cfg = KG_CONFIGS[current_kg]
                rag = make_rag(cfg["working_dir"])
                await rag.initialize_storages()
                print(f"  切换到库: [{cfg['label']}]")
        else:
            # 执行查询
            print(f"  检索中...")
            result = await rag.aquery(
                user_input,
                param=QueryParam(mode=current_mode, enable_rerank=False),
            )
            print(f"\n{result}\n")

    await rag.finalize_storages()


# ===== 6. 主入口 =====
async def main():
    parser = argparse.ArgumentParser(description="LightRAG 知识图谱构建与查询")
    parser.add_argument(
        "command", nargs="?", default="both",
        choices=["cs", "ss", "both"],
        help="建哪个库: cs | ss | both (默认both)"
    )
    parser.add_argument(
        "--skip-import", action="store_true",
        help="跳过导入，直接进入查询"
    )
    parser.add_argument(
        "--query", "-q", action="store_true",
        help="导入后进入交互查询"
    )
    parser.add_argument(
        "--mode", "-m", default="mix",
        choices=QUERY_MODES,
        help="查询模式 (默认mix)"
    )
    parser.add_argument(
        "--query-kg", default="cs",
        choices=["cs", "ss"],
        help="查询哪个库 (默认cs)"
    )
    parser.add_argument(
        "--run", "-r", type=str, nargs="*",
        help="直接执行一条查询（不进入交互模式）"
    )
    args = parser.parse_args()

    # 如果 command 是 cs/ss，且用户没显式指定 --query-kg，则默认用 command
    if args.command in ("cs", "ss"):
        # argparse 的 default 会覆盖，所以需要判断用户是否改了
        # 简单处理：如果 query_kg 是默认值 "cs" 但 command 是 "ss"，则用 "ss"
        pass  # 下面有更干净的处理
    if args.command in ("cs", "ss"):
        args.query_kg = args.command  # command 决定默认查哪个库

    # ---- 导入阶段 ----
    if not args.skip_import:
        targets = ["cs", "ss"] if args.command == "both" else [args.command]
        for name in targets:
            await build_kg(name)
    else:
        print("跳过导入.\n")

    # ---- 查询阶段 ----
    if args.run:
        # 单条查询模式
        queries = args.run if isinstance(args.run, list) else [args.run]
        cfg = KG_CONFIGS[args.query_kg]
        rag = make_rag(cfg["working_dir"])
        await rag.initialize_storages()

        for q in queries:
            print(f"{'='*50}")
            print(f"[{cfg['label']}:{args.mode}] {q}")
            print(f"{'='*50}")
            result = await rag.aquery(
                q,
                param=QueryParam(mode=args.mode, enable_rerank=False),
            )
            print(result)
            print()

        await rag.finalize_storages()

    elif args.query:
        await query_loop(args.query_kg, args.mode)


if __name__ == "__main__":
    asyncio.run(main())
