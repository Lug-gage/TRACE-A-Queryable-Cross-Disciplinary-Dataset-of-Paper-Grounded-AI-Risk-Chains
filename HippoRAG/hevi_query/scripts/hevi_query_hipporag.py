"""
HEVI 查询脚本 — HippoRAG 版本

从 dataset.json 读取论文和 workflow_chains，用 HippoRAG 检索 + LLM 生成
dose_response / vulnerability / impact。

用法:
    python hevi_query/scripts/hevi_query_hipporag.py                          # 全部串行，ss 库
    python hevi_query/scripts/hevi_query_hipporag.py --workers 5              # 5 线程并行，自动分片
    python hevi_query/scripts/hevi_query_hipporag.py --limit 3                # 前 3 条
    python hevi_query/scripts/hevi_query_hipporag.py --paper icml_2024_0001   # 指定论文
    python hevi_query/scripts/hevi_query_hipporag.py --no-kg                  # 纯 LLM
    python hevi_query/scripts/hevi_query_hipporag.py --dpr-only               # DPR only
    python hevi_query/scripts/hevi_query_hipporag.py --start 0 --count 100    # 手动分片
"""
import json, argparse, sys, logging, os, time, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.WARNING, force=True)
for _name in ("hipporag", "openai", "httpx", "asyncio", "urllib3", "requests", "httpcore"):
    logging.getLogger(_name).setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
os.environ["TQDM_DISABLE"] = "1"

from src.hipporag import HippoRAG
from src.hipporag.utils.config_utils import BaseConfig

# ---- 路径 & 配置 ----
BASE_DIR       = Path(__file__).resolve().parent.parent   # hevi_query/
SCRIPTS_DIR    = Path(__file__).resolve().parent           # hevi_query/scripts/
DATASET_PATH   = BASE_DIR / "dataset.json"
OUTPUT_DIR     = BASE_DIR / "hipporag_results"; OUTPUT_DIR.mkdir(exist_ok=True)
VI_PROMPT      = (SCRIPTS_DIR / "hevi_vuln_impact.txt").read_text(encoding="utf-8")
DR_PROMPT      = (SCRIPTS_DIR / "hevi_dr.txt").read_text(encoding="utf-8")

# api_key.txt: 按 hevi_query/ → 上级目录 → CWD 顺序查找
_key_path = None
for _candidate in (BASE_DIR / "api_key.txt",
                   BASE_DIR.parent / "api_key.txt",
                   Path("api_key.txt")):
    if _candidate.exists():
        _key_path = _candidate
        break
if _key_path is None:
    raise FileNotFoundError("api_key.txt 未找到，请放在 hevi_query/ 或项目根目录或当前工作目录")
_api_key = _key_path.read_text().strip()
os.environ["OPENAI_API_KEY"] = _api_key

# 项目根目录是 hevi_query 的上级（HippoRAG-main/）
_PROJECT_ROOT = BASE_DIR.parent

BASE_URL       = "https://www.highland-api.top/v1"
LLM_MODEL      = "deepseek-v4-pro"
EMBEDDING_MODEL = "text-embedding-3-large"
RETRY_MAX      = 5


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
#  检索 query 构造
# ============================================================

def _first(chain, key):
    v = chain.get(key, [])
    return v[0] if isinstance(v, list) and v else str(v)


def he_query(chain):
    return ". ".join(filter(None, [_first(chain, "hazard"), _first(chain, "exposure")])) + "."


def si_query(chain):
    return ". ".join(filter(None, [chain.get("scenario", ""), chain.get("issue", "")])) + "."


# ============================================================
#  prompt 构造
# ============================================================

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
#  LLM 调用
# ============================================================

def _make_llm(hipporag, use_kg):
    if use_kg and hipporag:
        def call(prompt, docs_text=""):
            if docs_text:
                prompt = f"=== RETRIEVED CONTEXT ===\n{docs_text}\n\n{prompt}"
            msg, _, _ = _retry_llm(
                hipporag.llm_model.infer,
                [{"role": "user", "content": prompt}], max_completion_tokens=384000)
            return msg
    else:
        from openai import OpenAI
        client = OpenAI(api_key=_api_key, base_url=BASE_URL)
        def call(prompt, docs_text=""):
            if docs_text:
                prompt = f"=== RETRIEVED CONTEXT ===\n{docs_text}\n\n{prompt}"
            resp = _retry_llm(
                client.chat.completions.create,
                model=LLM_MODEL, messages=[{"role": "user", "content": prompt}],
                max_tokens=384000, timeout=120)
            return resp.choices[0].message.content
    return call


# ============================================================
#  单条 chain 处理
# ============================================================

def _process_chain(paper, ci, chain, hipporag, use_kg, llm):
    """处理单条 chain：检索 + VI 生成 + DR 生成"""
    pid = paper["paper_id"]

    # 检索
    docs_text = ""
    if use_kg and hipporag:
        he = hipporag.retrieve([he_query(chain)], num_to_retrieve=3)
        si = hipporag.retrieve([si_query(chain)], num_to_retrieve=3)
        docs = list(he[0].docs)
        for d in si[0].docs:
            if d not in docs:
                docs.append(d)
        docs_text = "\n\n".join(f"[Document {j+1}]: {d}" for j, d in enumerate(docs))

    # Step 1: VI
    raw = llm(_vi_prompt(paper, chain), docs_text)
    vi = _parse_json(raw)

    vuln = vi.get("vulnerability", "")
    imp = vi.get("impact", "")
    if not vuln or not imp:
        return pid, ci, None, True  # VI 失败

    # Step 2: DR
    raw = llm(_dr_prompt(paper, chain, vuln, imp), "")
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
        "hipporag_result": {
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

def _run_worker(my_chunk, index_name, use_kg, wid):
    """单个 worker：创建独立 HippoRAG 实例，串行处理分到的 chains"""
    hipporag = load_hipporag(index_name) if use_kg else None
    llm = _make_llm(hipporag, use_kg)

    n_done = 0
    n_failed = 0
    for paper, ci, chain in my_chunk:
        pid, ci, _, failed = _process_chain(paper, ci, chain, hipporag, use_kg, llm)
        if failed:
            n_failed += 1
            print(f"  [W{wid}] {pid}_chain{ci} ✗ VI失败，跳过", flush=True)
        else:
            n_done += 1
            print(f"  [W{wid}] {pid}_chain{ci} ✓", flush=True)

    return n_done, n_failed


# ============================================================
#  主流程
# ============================================================

def run(index="ss", limit=0, paper_id=None, no_kg=False, dpr_only=False,
        start=-1, count=0, workers=1):

    papers = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    use_kg = not no_kg
    parallel = start >= 0 and count > 0

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
    print(f"知识库: {index}")
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
                pool.submit(_run_worker, chunk, index, use_kg, wid): wid
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
        n_total, n_failed_total = _run_worker(tasks, index, use_kg, 0)

    summary_parts = [f"{n_total} 个文件  →  {OUTPUT_DIR}"]
    if n_failed_total:
        summary_parts.append(f"（{n_failed_total} 条 VI 失败跳过）")
    print("".join(summary_parts), flush=True)


def load_hipporag(name):
    return HippoRAG(global_config=BaseConfig(
        save_dir=str(_PROJECT_ROOT / "indices" / name),
        llm_base_url=BASE_URL, llm_name=LLM_MODEL,
        embedding_model_name=EMBEDDING_MODEL, embedding_base_url=BASE_URL,
        retrieval_top_k=10, linking_top_k=5, qa_top_k=5,
        max_new_tokens=384000, openie_mode="online",
    ))


# ============================================================
def main():
    p = argparse.ArgumentParser(description="HEVI 查询 (HippoRAG)")
    p.add_argument("--index", "-i", default="ss", choices=["ss", "cs"])
    p.add_argument("--limit", "-l", type=int, default=0)
    p.add_argument("--paper", "-p")
    p.add_argument("--no-kg", action="store_true")
    p.add_argument("--dpr-only", action="store_true")
    p.add_argument("--start", "-s", type=int, default=-1)
    p.add_argument("--count", "-c", type=int, default=0)
    p.add_argument("--workers", "-w", type=int, default=1,
                   help="并行 worker 数（自动分片，与 --start/--count 互斥）")
    args = p.parse_args()
    run(index=args.index, limit=args.limit, paper_id=args.paper,
        no_kg=args.no_kg, dpr_only=args.dpr_only,
        start=args.start, count=args.count,
        workers=args.workers)


if __name__ == "__main__":
    main()
