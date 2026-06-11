"""
GraphRAG 版 HEVI 查询脚本 (v2 — 两阶段 VI/DR)

从 dataset.json 读取论文和 workflow_chains，用 GraphRAG 检索 + 两阶段 LLM 生成
dose_response / vulnerability / impact。

用法:
    python hevi_query/scripts/hevi_query_graphrag.py                          # 全部串行，ss 库
    python hevi_query/scripts/hevi_query_graphrag.py --workers 5              # 5 线程并行，自动分片
    python hevi_query/scripts/hevi_query_graphrag.py --limit 3                # 前 3 条
    python hevi_query/scripts/hevi_query_graphrag.py --paper icml_2024_0001   # 指定论文
    python hevi_query/scripts/hevi_query_graphrag.py --no-kg                  # 纯 LLM
    python hevi_query/scripts/hevi_query_graphrag.py --mode global            # global 查询模式
    python hevi_query/scripts/hevi_query_graphrag.py --start 0 --count 100    # 手动分片
"""
import asyncio
import json, argparse, sys, logging, os, time, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.WARNING, force=True)
for _name in ("graphrag", "graphrag_llm", "lancedb", "openai", "httpx", "urllib3",
              "aiohttp", "litellm", "azure", "msal", "httpcore", "asyncio", "LiteLLM",
              "graphrag.query.context_builder.community_context"):
    logging.getLogger(_name).setLevel(logging.ERROR)

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["TQDM_DISABLE"] = "1"

import warnings
warnings.filterwarnings("ignore", message=".*Unclosed.*")

import pandas as pd
from graphrag.api.query import local_search, global_search
from graphrag.config.load_config import load_config

# ---- 路径 & 配置 ----
BASE_DIR       = Path(__file__).parent.parent
DATASET_PATH   = BASE_DIR / "dataset.json"
OUTPUT_DIR     = BASE_DIR / "graph_result"; OUTPUT_DIR.mkdir(exist_ok=True)
VI_PROMPT      = (Path(__file__).parent / "hevi_vuln_impact.txt").read_text(encoding="utf-8")
DR_PROMPT      = (Path(__file__).parent / "hevi_dr.txt").read_text(encoding="utf-8")

PROJECT_ROOT   = Path(__file__).parent.parent.parent
_api_key       = (PROJECT_ROOT / "api_key.txt").read_text().strip()
os.environ["OPENAI_API_KEY"] = _api_key

BASE_URL       = "https://www.highland-api.top/v1"
LLM_MODEL      = "deepseek-v4-pro"
RETRY_MAX      = 5

# 全局共享的 GraphRAG config 和 data（主线程加载一次）
_graphrag_config = None
_graphrag_data   = None
_search_mode     = "local"
_community_level = 0


# ============================================================
#  helpers
# ============================================================

def _parse_json(text: str) -> dict:
    """多层兜底 --- 直接解析 -> 代码块 -> 最外层 {}"""
    for source in [text,
                   re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL),
                   re.search(r'\{.*\}', text, re.DOTALL)]:
        try:
            src = source.group(1) if isinstance(source, re.Match) else source
            return json.loads(src)
        except: pass
    return {}


def _retry_llm(fn, *args, **kwargs):
    """5 次重试 + 指数退避"""
    for attempt in range(1, RETRY_MAX + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == RETRY_MAX: raise
            wait = min(2 ** attempt, 30)
            print(f"    [LLM重试 {attempt}/{RETRY_MAX}] {type(e).__name__}, {wait}s后重试...")
            time.sleep(wait)


# ============================================================
#  prompt 构造
# ============================================================

def _first(chain, key):
    v = chain.get(key, [])
    return v[0] if isinstance(v, list) and v else str(v)


def _vi_prompt(paper, chain):
    return VI_PROMPT.format(
        title=paper.get("title", ""), abstract=paper.get("abstract", ""),
        scenario=chain.get("scenario", ""), issue=chain.get("issue", ""),
        hazard=_first(chain, "hazard"), exposure=_first(chain, "exposure"),
    )


def _dr_prompt(paper, chain, vulnerability, impact):
    return DR_PROMPT.format(
        abstract=paper.get("abstract", ""), paper_impact=paper.get("impact", ""),
        scenario=chain.get("scenario", ""), issue=chain.get("issue", ""),
        hazard=_first(chain, "hazard"), exposure=_first(chain, "exposure"),
        vulnerability=vulnerability, impact=impact,
    )


# ============================================================
#  GraphRAG 检索 + 生成（替代 HippoRAG retrieve + LLM）
# ============================================================

def _graphrag_search(query_text: str) -> str:
    """用 GraphRAG 执行一次检索+生成，返回响应文本。"""
    config = _graphrag_config
    data = _graphrag_data

    if _search_mode == "global":
        response, _ = asyncio.run(
            global_search(
                config=config,
                entities=data["entities"],
                communities=data["communities"],
                community_reports=data["community_reports"],
                community_level=_community_level,
                dynamic_community_selection=False,
                response_type="multiple paragraphs",
                query=query_text,
            )
        )
    else:
        # local / drift 都走 local_search
        response, _ = asyncio.run(
            local_search(
                config=config,
                entities=data["entities"],
                communities=data["communities"],
                community_reports=data["community_reports"],
                text_units=data["text_units"],
                relationships=data["relationships"],
                covariates=data.get("covariates"),
                community_level=_community_level,
                response_type="multiple paragraphs",
                query=query_text,
            )
        )
    return response if isinstance(response, str) else str(response)


# ============================================================
#  纯 LLM 调用（--no-kg 模式）
# ============================================================

def _make_openai_llm():
    from openai import OpenAI
    client = OpenAI(api_key=_api_key, base_url=BASE_URL)
    def call(prompt):
        resp = _retry_llm(
            client.chat.completions.create,
            model=LLM_MODEL, messages=[{"role": "user", "content": prompt}],
            max_tokens=384000, timeout=120)
        return resp.choices[0].message.content
    return call


# ============================================================
#  单条 chain 处理
# ============================================================

def _process_chain(paper, ci, chain, use_kg, openai_llm):
    """处理单条 chain：检索 + VI 生成 + DR 生成"""
    pid = paper["paper_id"]

    if use_kg:
        # ---- GraphRAG 路径：两步 local_search ----
        try:
            raw_vi = _graphrag_search(_vi_prompt(paper, chain))
        except Exception as e:
            print(f"    ✗ VI查询失败: {e}")
            return pid, ci, None, True

        vi = _parse_json(raw_vi)
        vuln = vi.get("vulnerability", "")
        imp = vi.get("impact", "")
        if not vuln or not imp:
            return pid, ci, None, True  # VI 失败

        # Step 2: DR
        try:
            raw_dr = _graphrag_search(_dr_prompt(paper, chain, vuln, imp))
        except Exception as e:
            print(f"    ✗ DR查询失败: {e}")
            return pid, ci, None, True

        dr = _parse_json(raw_dr)
        dr_text = dr.get("dose_response", "")
        if not dr_text:
            dr_text = raw_dr[:300]
    else:
        # ---- 纯 LLM 路径（无检索）----
        llm = openai_llm

        raw = llm(_vi_prompt(paper, chain))
        vi = _parse_json(raw)
        vuln = vi.get("vulnerability", "")
        imp = vi.get("impact", "")
        if not vuln or not imp:
            return pid, ci, None, True

        raw = llm(_dr_prompt(paper, chain, vuln, imp))
        dr = _parse_json(raw)
        dr_text = dr.get("dose_response", "")
        if not dr_text:
            dr_text = raw[:300]

    out = {
        "paper_id": pid, "chain_index": ci,
        "title": paper.get("title", ""), "abstract": paper.get("abstract", ""),
        "impact": paper.get("impact", ""),
        "query_input": {
            "scenario": chain.get("scenario", ""), "issue": chain.get("issue", ""),
            "hazard": chain.get("hazard", []), "exposure": chain.get("exposure", []),
        },
        "graphrag_result": {
            "vulnerability": vuln,
            "impact": imp,
            "dose_response": dr_text,
        },
        "reference": {
            "dose_response": chain.get("dose_response", []),
            "vulnerability": chain.get("vulnerability", []),
            "impact": chain.get("impact", []),
        },
    }

    dest = OUTPUT_DIR / f"{pid}_chain{ci}.json"
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(dest)

    return pid, ci, out, False


# ============================================================
#  Worker
# ============================================================

def _run_worker(my_chunk, use_kg, wid):
    """单个 worker：串行处理分到的 chains"""
    openai_llm = _make_openai_llm() if not use_kg else None

    n_done = 0
    n_failed = 0
    for paper, ci, chain in my_chunk:
        pid, ci, _, failed = _process_chain(paper, ci, chain, use_kg, openai_llm)
        if failed:
            n_failed += 1
            print(f"  [W{wid}] {pid}_chain{ci} ✗ VI失败，跳过", flush=True)
        else:
            n_done += 1
            print(f"  [W{wid}] {pid}_chain{ci} ✓", flush=True)

    return n_done, n_failed


# ============================================================
#  GraphRAG 数据加载
# ============================================================

def _load_graphrag(dataset: str):
    """加载 GraphRAG 配置和索引数据（主线程调用一次）"""
    global _graphrag_config, _graphrag_data

    root_dir = PROJECT_ROOT / "indices" / dataset
    if not (root_dir / "settings.yaml").exists():
        print(f"错误: 找不到 {root_dir}/settings.yaml")
        sys.exit(1)

    config = load_config(root_dir=root_dir)
    output_dir = root_dir / "output"
    config.output_storage.base_dir = str(output_dir.resolve())

    from graphrag_storage import create_storage
    from graphrag_storage.tables.table_provider_factory import create_table_provider
    from graphrag.data_model.data_reader import DataReader

    storage = create_storage(config.output_storage)
    provider = create_table_provider(config.table_provider, storage=storage)
    reader = DataReader(provider)

    data = {}
    for name in ["entities", "relationships", "communities", "community_reports",
                  "text_units", "documents"]:
        df = asyncio.run(getattr(reader, name)())
        data[name] = df
        print(f"  Loaded {name}: {len(df)} rows")

    try:
        data["covariates"] = asyncio.run(reader.covariates())
    except Exception:
        data["covariates"] = pd.DataFrame()

    _graphrag_config = config
    _graphrag_data = data


# ============================================================
#  主流程
# ============================================================

def run(index="ss", limit=0, paper_id=None, no_kg=False,
        start=-1, count=0, workers=1):

    papers = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    use_kg = not no_kg
    parallel = start >= 0 and count > 0

    # ---- 加载 GraphRAG（KG 模式）----
    if use_kg:
        print(f"加载 {index} 知识库...")
        _load_graphrag(index)

    # ---- 收集 chains ----
    tasks = []
    for p in papers:
        if paper_id and p["paper_id"] != paper_id:
            continue
        for ci, chain in enumerate(p.get("workflow_chains", [])):
            if not chain.get("hazard") or not chain.get("exposure"):
                continue
            tasks.append((p, ci, chain))

    total = len(tasks)

    if parallel:
        tasks = tasks[start:start + count]
    else:
        tasks = [t for t in tasks
                 if not (OUTPUT_DIR / f"{t[0]['paper_id']}_chain{t[1]}.json").exists()]
        skipped_by_cache = total - len(tasks)
        skipped_by_limit = 0
        if limit and len(tasks) > limit:
            skipped_by_limit = len(tasks) - limit
            tasks = tasks[:limit]

    # ---- 打印任务概况 ----
    kg_label = f"GraphRAG ({_search_mode})" if use_kg else "纯 LLM (无检索)"
    print(f"知识库: {index}  |  模式: {kg_label}")
    if parallel:
        print(f"手动分片: start={start} count={count}  →  {len(tasks)} 条")
    elif workers > 1:
        print(f"并行 worker: {workers}")
        print(f"跳过 {skipped_by_cache} 条（已有结果），待查询 {len(tasks)} 条")
    else:
        parts = []
        if skipped_by_cache:
            parts.append(f"跳过 {skipped_by_cache} 条（已有结果）")
        if skipped_by_limit:
            parts.append(f"因 limit 忽略 {skipped_by_limit} 条")
        if parts:
            print(", ".join(parts) + f"，待查询 {len(tasks)} 条")
        else:
            print(f"共 {len(tasks)} 条")
    print(flush=True)

    if not tasks:
        return

    # ---- 执行 ----
    if workers > 1 and not parallel:
        # 轮询分片
        chunks = [[] for _ in range(workers)]
        for i, t in enumerate(tasks):
            chunks[i % workers].append(t)
        print(f"分组: {' | '.join(f'W{w}={len(c)}条' for w, c in enumerate(chunks))}")
        print(flush=True)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_worker, chunk, use_kg, wid): wid
                for wid, chunk in enumerate(chunks)
            }
            n_total = 0
            n_failed_total = 0
            for f in as_completed(futures):
                n_d, n_f = f.result()
                n_total += n_d
                n_failed_total += n_f
    else:
        # 串行
        n_total, n_failed_total = _run_worker(tasks, use_kg, 0)

    summary_parts = [f"{n_total} 个文件  →  {OUTPUT_DIR}"]
    if n_failed_total:
        summary_parts.append(f"（{n_failed_total} 条 VI 失败跳过）")
    print("".join(summary_parts), flush=True)


# ============================================================
def main():
    p = argparse.ArgumentParser(description="HEVI 查询 (GraphRAG)")
    p.add_argument("--index", "-i", default="ss", choices=["ss", "cs"])
    p.add_argument("--limit", "-l", type=int, default=0)
    p.add_argument("--paper", "-p")
    p.add_argument("--no-kg", action="store_true")
    p.add_argument("--mode", "-m", default="local", choices=["local", "global", "drift"],
                   help="GraphRAG 查询模式 (默认 local)")
    p.add_argument("--start", "-s", type=int, default=-1)
    p.add_argument("--count", "-c", type=int, default=0)
    p.add_argument("--workers", "-w", type=int, default=1,
                   help="并行 worker 数（自动分片，与 --start/--count 互斥）")
    args = p.parse_args()

    global _search_mode, _community_level
    _search_mode = args.mode
    _community_level = 0

    run(index=args.index, limit=args.limit, paper_id=args.paper,
        no_kg=args.no_kg,
        start=args.start, count=args.count,
        workers=args.workers)


if __name__ == "__main__":
    main()
