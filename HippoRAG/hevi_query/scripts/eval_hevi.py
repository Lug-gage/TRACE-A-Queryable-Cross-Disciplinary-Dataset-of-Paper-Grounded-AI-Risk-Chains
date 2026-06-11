"""
HEVI 结果评价

将 hipporag_result 与 reference 逐条对比，LLM 评判 covered/partial/not，
每条 chain 独立 eval 文件，最后覆盖率汇总。

用法:
    python hevi_query/scripts/eval_hevi.py                          # 全部串行
    python hevi_query/scripts/eval_hevi.py --workers 3              # 3 线程并行
    python hevi_query/scripts/eval_hevi.py --limit 5                # 前 5 条
    python hevi_query/scripts/eval_hevi.py --paper icml_2024_0001   # 指定论文
    python hevi_query/scripts/eval_hevi.py --start 0 --count 100    # 分片并行
    python hevi_query/scripts/eval_hevi.py --watch                  # 监控模式
"""
import json, argparse, logging, os, re, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.WARNING, force=True)
for _n in ("openai", "httpx", "asyncio", "urllib3", "requests", "httpcore"):
    logging.getLogger(_n).setLevel(logging.WARNING)

from openai import OpenAI

# ---- 配置 ----
BASE_DIR     = Path(__file__).resolve().parent.parent   # hevi_query/
SCRIPTS_DIR  = Path(__file__).resolve().parent           # hevi_query/scripts/

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
API_KEY = _key_path.read_text().strip()

BASE_URL     = "https://www.highland-api.top/v1"
LLM_MODEL    = "deepseek-v4-pro"

RESULTS_DIR  = BASE_DIR / "hipporag_results"   # 默认，可被 --results-dir 覆盖
EVAL_DIR     = BASE_DIR / "evaluation"          # 默认，随 --results-dir 自动调整
EVAL_PROMPT  = (SCRIPTS_DIR / "eval_prompt.txt").read_text(encoding="utf-8")

FIELDS       = ("vulnerability", "impact", "dose_response")


# ============================================================
#  helpers
# ============================================================

def _esc(s):
    return s.replace("{", "{{").replace("}", "}}")


def _set_eval_dir(results_dir):
    """根据 results_dir 推导 eval 输出目录"""
    global EVAL_DIR
    if results_dir:
        # llm_results → evaluation_llm, hipporag_results → evaluation
        name = Path(results_dir).name
        if name == "hipporag_results":
            EVAL_DIR = BASE_DIR / "evaluation"
        else:
            # llm_results → evaluation_llm
            suffix = name.replace("_results", "")
            EVAL_DIR = BASE_DIR / f"evaluation_{suffix}"
    else:
        EVAL_DIR = BASE_DIR / "evaluation"
    EVAL_DIR.mkdir(exist_ok=True)


def _parse_json(text):
    for src in [text,
                re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL),
                re.search(r'\{.*\}', text, re.DOTALL)]:
        try:
            s = src.group(1) if isinstance(src, re.Match) else src
            return json.loads(s)
        except: pass
    return {}


def _ref(r, field):
    v = r.get(field, [])
    return v[0] if isinstance(v, list) and v else str(v)


# ============================================================
#  单条评价
# ============================================================

def _eval_item(pid, ci, data, client):
    """评价一条 chain，返回 (pid, ci, result)"""
    gen  = data["hipporag_result"]
    ref  = data["reference"]
    abst = data.get("abstract", "")
    pimp = data.get("impact", "")

    gen_text = {f: gen.get(f, "") for f in FIELDS}
    ref_text = {f: _ref(ref, f) for f in FIELDS}

    query = EVAL_PROMPT.format(
        abstract=_esc(abst), paper_impact=_esc(pimp),
        ref_dose_response=_esc(ref_text["dose_response"]),
        ref_vulnerability=_esc(ref_text["vulnerability"]),
        ref_impact=_esc(ref_text["impact"]),
        gen_dose_response=_esc(gen_text["dose_response"]),
        gen_vulnerability=_esc(gen_text["vulnerability"]),
        gen_impact=_esc(gen_text["impact"]),
    )

    for attempt in range(1, 6):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL, messages=[{"role": "user", "content": query}],
                max_tokens=384000, temperature=0, timeout=120)
            raw = resp.choices[0].message.content or ""
            data_json = _parse_json(raw)
            result = {}
            for f in FIELDS:
                result[f] = {
                    "match": data_json.get(f, {}).get("match", "not"),
                    "reason": data_json.get(f, {}).get("reason", ""),
                    "reference": ref_text[f],
                    "generated": gen_text[f],
                }
            return pid, ci, result
        except Exception as e:
            if attempt == 5:
                return pid, ci, {f: {"match": "not", "reason": f"failed: {e}",
                                     "reference": ref_text[f], "generated": gen_text[f]}
                                 for f in FIELDS}
            time.sleep(min(2 ** attempt, 30))


# ============================================================
#  Worker
# ============================================================

def _run_worker(chunk, wid):
    """每个 worker 有自己的 OpenAI client，处理分到的链"""
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=120)
    n = 0
    for pid, ci, data in chunk:
        pid, ci, r = _eval_item(pid, ci, data, client)
        r = {"paper_id": pid, "chain_index": ci, **r}

        dest = EVAL_DIR / f"{pid}_chain{ci}.json"
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(dest)

        n += 1
        print(f"  [W{wid}] {pid}_chain{ci}  "
              f"V={r['vulnerability']['match']}  "
              f"I={r['impact']['match']}  "
              f"DR={r['dose_response']['match']}", flush=True)
    return n


# ============================================================
#  汇总
# ============================================================

def summary(eval_files, total_available=0):
    if not eval_files:
        return
    counts = {f: {"covered": 0, "partial": 0, "not": 0} for f in FIELDS}
    for fp in eval_files:
        r = json.loads(fp.read_text(encoding="utf-8"))
        for f in FIELDS:
            m = r.get(f, {}).get("match", "not")
            counts[f][m] = counts[f].get(m, 0) + 1

    n = len(eval_files)
    tc = 0
    info = f"  ({n}" + (f" / {total_available}" if total_available else "") + " 已评)"
    print(f"\n{'='*60}")
    print(f" 汇总{info}")
    print(f"{'='*60}")
    for f in FIELDS:
        c = counts[f]
        cov = c["covered"] / n * 100 if n else 0
        tc += c["covered"]
        print(f"  {f:16s}  covered={c['covered']:2d}  partial={c['partial']:2d}  "
              f"not={c['not']:2d}  → {cov:.1f}%")
    print(f"  {'整体':16s}  {tc / (n * 3) * 100:.1f}%")


# ============================================================
#  主流程
# ============================================================

def _resolve_src_dir(results_dir):
    """解析结果源目录，支持相对于 BASE_DIR 的名称"""
    if not results_dir:
        return RESULTS_DIR
    p = Path(results_dir)
    if not p.is_absolute() and not p.exists():
        p = BASE_DIR / results_dir
    return p


def run_batch(limit=0, paper=None, start=-1, count=0, workers=1, results_dir=None):
    _set_eval_dir(results_dir)
    src_dir = _resolve_src_dir(results_dir)
    all_files = sorted(src_dir.glob("*.json"))
    if paper:
        all_files = [f for f in all_files if f.stem.startswith(paper)]

    if not all_files:
        print("无结果文件。"); return

    parallel = start >= 0 and count > 0
    if parallel:
        all_files = all_files[start:start + count]
        pending = []
        for f in all_files:
            d = json.loads(f.read_text(encoding="utf-8"))
            pid = f.stem.rsplit("_chain", 1)[0]
            ci = d["chain_index"]
            pending.append((pid, ci, d))
        skipped = 0
    else:
        pending = []; skipped = 0
        for f in all_files:
            if (EVAL_DIR / f.name).exists():
                skipped += 1; continue
            d = json.loads(f.read_text(encoding="utf-8"))
            pid = f.stem.rsplit("_chain", 1)[0]
            ci = d["chain_index"]
            pending.append((pid, ci, d))
        if limit:
            pending = pending[:limit]

    if parallel:
        print(f"分片: start={start} count={count}  →  {len(pending)} 条")
    elif workers > 1:
        print(f"并行 worker: {workers}")
        if skipped:
            print(f"跳过 {skipped} 条  待评 {len(pending)} 条")
        else:
            print(f"共 {len(pending)} 条")
    elif skipped:
        print(f"跳过 {skipped} 条  待评 {len(pending)} 条")
    else:
        print(f"共 {len(pending)} 条")
    print(flush=True)

    if not pending:
        summary(sorted(EVAL_DIR.glob("*.json")))
        return

    # 执行
    if workers > 1 and not parallel:
        chunks = [[] for _ in range(workers)]
        for i, t in enumerate(pending):
            chunks[i % workers].append(t)
        print(f"分组: {' | '.join(f'W{w}={len(c)}条' for w, c in enumerate(chunks))}")
        print(flush=True)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run_worker, chunk, wid) for wid, chunk in enumerate(chunks)]
            n_total = sum(f.result() for f in as_completed(futures))
    else:
        n_total = _run_worker(pending, 0)

    print(f"\n{n_total} 个文件  →  {EVAL_DIR}", flush=True)
    summary(sorted(EVAL_DIR.glob("*.json")))


# ============================================================
#  监控模式
# ============================================================

def run_watch(interval=5, expected_total=0, results_dir=None, eval_workers=3):
    _set_eval_dir(results_dir)
    src_dir = _resolve_src_dir(results_dir)

    import threading
    from queue import Queue

    task_queue = Queue()
    seen = set()   # 已入队的文件名
    seen_lock = threading.Lock()
    stop_flag = threading.Event()

    # ---- 预扫已有文件 ----
    for f in sorted(src_dir.glob("*.json")):
        if not (EVAL_DIR / f.name).exists():
            with seen_lock:
                seen.add(f.name)
            d = json.loads(f.read_text(encoding="utf-8"))
            pid = f.stem.rsplit("_chain", 1)[0]
            ci = d["chain_index"]
            task_queue.put((f.name, pid, ci, d))

    # ---- 监控线程 ----
    def _monitor():
        while not stop_flag.is_set():
            time.sleep(interval)
            if stop_flag.is_set():
                break
            all_files = sorted(src_dir.glob("*.json"))
            for f in all_files:
                if stop_flag.is_set():
                    break
                if (EVAL_DIR / f.name).exists():
                    continue
                with seen_lock:
                    if f.name in seen:
                        continue
                    seen.add(f.name)
                d = json.loads(f.read_text(encoding="utf-8"))
                pid = f.stem.rsplit("_chain", 1)[0]
                ci = d["chain_index"]
                task_queue.put((f.name, pid, ci, d))

            n_evaled = len(list(EVAL_DIR.glob("*.json")))
            n_total = len(all_files)
            print(f"[{time.strftime('%H:%M:%S')}] 已评 {n_evaled}/{n_total}", flush=True)
            if expected_total and n_evaled >= expected_total:
                stop_flag.set()
                break

        # 发终止信号给 worker
        for _ in range(eval_workers):
            task_queue.put(None)

    # ---- 评价 worker ----
    def _eval_worker(wid):
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=120)
        n = 0
        while True:
            item = task_queue.get()
            if item is None:
                task_queue.task_done()
                break
            fname, pid, ci, data = item
            pid, ci, r = _eval_item(pid, ci, data, client)
            r = {"paper_id": pid, "chain_index": ci, **r}

            dest = EVAL_DIR / f"{pid}_chain{ci}.json"
            tmp = dest.with_suffix(".tmp")
            tmp.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(dest)

            n += 1
            print(f"  [W{wid}] {pid}_chain{ci}  "
                  f"V={r['vulnerability']['match']}  "
                  f"I={r['impact']['match']}  "
                  f"DR={r['dose_response']['match']}", flush=True)
            task_queue.task_done()
        return n

    # ---- 启动 ----
    print(f"监控模式: 1 扫描 + {eval_workers} 评价（间隔 {interval}s）", flush=True)
    if expected_total:
        print(f"期望总数: {expected_total}，满额自动退出", flush=True)
    print(flush=True)

    monitor_thread = threading.Thread(target=_monitor, daemon=True)
    monitor_thread.start()

    n_total = 0
    with ThreadPoolExecutor(max_workers=eval_workers) as pool:
        futures = [pool.submit(_eval_worker, wid) for wid in range(eval_workers)]
        monitor_thread.join()
        n_total = sum(f.result() for f in as_completed(futures))

    result_files = sorted(EVAL_DIR.glob("*.json"))
    n = len(result_files)
    if n:
        print(f"\n✅ 全部完成 ({n}/{expected_total if expected_total else n})", flush=True)
        summary(result_files, expected_total)


# ============================================================
def main():
    p = argparse.ArgumentParser(description="HEVI 评价")
    p.add_argument("--limit", "-l", type=int, default=0)
    p.add_argument("--paper", "-p")
    p.add_argument("--start", "-s", type=int, default=-1)
    p.add_argument("--count", "-c", type=int, default=0)
    p.add_argument("--workers", "-w", type=int, default=1,
                   help="并行 worker 数（自动分片，与 --start/--count 互斥）")
    p.add_argument("--watch", action="store_true",
                   help="监控模式：持续扫描新结果文件")
    p.add_argument("--expected", type=int, default=818,
                   help="期望评价总数，监控模式满额自动退出")
    p.add_argument("--results-dir", "-r", default=None,
                   help="结果目录名（默认 hipporag_results，可用 llm_results）")
    args = p.parse_args()

    if args.watch:
        run_watch(interval=5, expected_total=args.expected,
                  results_dir=args.results_dir, eval_workers=args.workers)
    else:
        run_batch(limit=args.limit, paper=args.paper,
                  start=args.start, count=args.count, workers=args.workers,
                  results_dir=args.results_dir)


if __name__ == "__main__":
    main()
