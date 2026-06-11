"""
HEVI 查询脚本 — 纯 LLM 版本（无 RAG 检索）

仅使用论文自身的 title + abstract + scenario + issue + hazard + exposure
通过 LLM 直接生成 vulnerability / impact / dose_response。

用法:
    python hevi_query_llm.py                          # 全部串行
    python hevi_query_llm.py --workers 5              # 5 线程并行
    python hevi_query_llm.py --limit 3                # 前 3 条
    python hevi_query_llm.py --paper icml_2024_0001   # 指定论文
    python hevi_query_llm.py --start 0 --count 100    # 手动分片
"""
import json, argparse, logging, os, time, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.WARNING, force=True)
for _name in ("openai", "httpx", "asyncio", "urllib3", "requests", "httpcore"):
    logging.getLogger(_name).setLevel(logging.WARNING)

from openai import OpenAI

# ---- 路径 & 配置 ----
SCRIPTS_DIR = Path(__file__).resolve().parent           # hevi_llm/scripts/
BASE_DIR    = SCRIPTS_DIR.parent                         # hevi_llm/
OUTPUT_DIR  = BASE_DIR / "llm_results"; OUTPUT_DIR.mkdir(exist_ok=True)

# prompt 文件（在 scripts/ 内）
VI_PROMPT = (SCRIPTS_DIR / "hevi_vuln_impact.txt").read_text(encoding="utf-8")
DR_PROMPT = (SCRIPTS_DIR / "hevi_dr.txt").read_text(encoding="utf-8")

# api_key.txt
_key_path = BASE_DIR / "api_key.txt"
if not _key_path.exists():
    raise FileNotFoundError(f"api_key.txt 未找到，请放在 {BASE_DIR}/")
API_KEY = _key_path.read_text().strip()

# 数据集
DATASET_PATH = BASE_DIR / "dataset.json"

BASE_URL  = "https://www.highland-api.top/v1"
LLM_MODEL = "deepseek-v4-pro"
RETRY_MAX = 5


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


def _first(chain, key):
    v = chain.get(key, [])
    return v[0] if isinstance(v, list) and v else str(v)


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
#  单条 chain 处理
# ============================================================

def _process_chain(paper, ci, chain, client):
    """纯 LLM：VI 生成 → DR 生成"""
    pid = paper["paper_id"]

    # Step 1: VI
    raw = _retry_llm(
        client.chat.completions.create,
        model=LLM_MODEL, messages=[{"role": "user", "content": _vi_prompt(paper, chain)}],
        max_tokens=384000, timeout=120)
    raw_text = raw.choices[0].message.content or ""
    vi = _parse_json(raw_text)

    vuln = vi.get("vulnerability", "")
    imp = vi.get("impact", "")
    if not vuln or not imp:
        return pid, ci, None, True  # VI 失败

    # Step 2: DR
    raw = _retry_llm(
        client.chat.completions.create,
        model=LLM_MODEL, messages=[{"role": "user", "content": _dr_prompt(paper, chain, vuln, imp)}],
        max_tokens=384000, timeout=120)
    raw_text = raw.choices[0].message.content or ""
    dr = _parse_json(raw_text)
    dr_text = dr.get("dose_response", "")
    if not dr_text:
        dr_text = raw_text[:300]

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

def _run_worker(my_chunk, wid):
    """单个 worker：独立 OpenAI client，串行处理分到的 chains"""
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=120)

    n_done = 0
    n_failed = 0
    for paper, ci, chain in my_chunk:
        pid, ci, _, failed = _process_chain(paper, ci, chain, client)
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

def run(limit=0, paper_id=None, start=-1, count=0, workers=1):

    papers = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
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
    print(f"模式: 纯 LLM（无 RAG）")
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
        chunks = [[] for _ in range(workers)]
        for i, t in enumerate(tasks):
            chunks[i % workers].append(t)
        print(f"分组: {' | '.join(f'W{w}={len(c)}条' for w, c in enumerate(chunks))}")
        print(flush=True)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_worker, chunk, wid): wid
                for wid, chunk in enumerate(chunks)
            }
            n_total = 0
            n_failed_total = 0
            for f in as_completed(futures):
                n_d, n_f = f.result()
                n_total += n_d
                n_failed_total += n_f
    else:
        n_total, n_failed_total = _run_worker(tasks, 0)

    summary_parts = [f"{n_total} 个文件  →  {OUTPUT_DIR}"]
    if n_failed_total:
        summary_parts.append(f"（{n_failed_total} 条 VI 失败跳过）")
    print("".join(summary_parts), flush=True)


# ============================================================
def main():
    p = argparse.ArgumentParser(description="HEVI 查询 (纯 LLM)")
    p.add_argument("--limit", "-l", type=int, default=0)
    p.add_argument("--paper", "-p")
    p.add_argument("--start", "-s", type=int, default=-1)
    p.add_argument("--count", "-c", type=int, default=0)
    p.add_argument("--workers", "-w", type=int, default=1,
                   help="并行 worker 数（自动分片，与 --start/--count 互斥）")
    args = p.parse_args()
    run(limit=args.limit, paper_id=args.paper,
        start=args.start, count=args.count,
        workers=args.workers)


if __name__ == "__main__":
    main()
