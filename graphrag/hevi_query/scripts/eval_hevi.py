"""
HEVI 结果评价

将 graphrag_result 与 reference 逐条对比，LLM 评判 covered/partial/not，
每条 chain 独立 eval 文件，最后覆盖率汇总。

用法:
    python hevi_query/scripts/eval_hevi.py                          # 全部串行
    python hevi_query/scripts/eval_hevi.py --workers 3              # 3 线程并行
    python hevi_query/scripts/eval_hevi.py --limit 5                # 前 5 条
    python hevi_query/scripts/eval_hevi.py --paper icml_2024_0001   # 指定论文
    python hevi_query/scripts/eval_hevi.py --start 0 --count 100    # 分片并行
    python hevi_query/scripts/eval_hevi.py --watch                  # 监控模式
    python hevi_query/scripts/eval_hevi.py --workers 2 --expected 818  # 2线程，满818条后汇总
"""
import json, argparse, logging, os, re, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.WARNING, force=True)
for _n in ("openai", "httpx", "asyncio", "urllib3", "requests", "httpcore"):
    logging.getLogger(_n).setLevel(logging.WARNING)

from openai import OpenAI

# ---- 配置 ----
PROJECT_ROOT = Path(__file__).parent.parent.parent
API_KEY      = (PROJECT_ROOT / "api_key.txt").read_text().strip()

BASE_URL     = "https://www.highland-api.top/v1"
LLM_MODEL    = "deepseek-v4-pro"

BASE_DIR     = Path(__file__).parent.parent
RESULTS_DIR  = BASE_DIR / "graph_result"
EVAL_DIR     = BASE_DIR / "evaluation"; EVAL_DIR.mkdir(exist_ok=True)
EVAL_PROMPT  = (Path(__file__).parent / "eval_prompt.txt").read_text(encoding="utf-8")

FIELDS       = ("vulnerability", "impact", "dose_response")


# ============================================================
#  helpers
# ============================================================

def _esc(s):
    return s.replace("{", "{{").replace("}", "}}")


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
    gen  = data["graphrag_result"]
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

def summary(eval_files, expected=0):
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
    info = f"  ({n}" + (f" / {expected}" if expected else "") + " 已评)"
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

def run_batch(limit=0, paper=None, start=-1, count=0, workers=1, expected=0):
    parallel = start >= 0 and count > 0

    if parallel:
        all_files = sorted(RESULTS_DIR.glob("*.json"))
        if paper:
            all_files = [f for f in all_files if f.stem.startswith(paper)]
        all_files = all_files[start:start + count]
        pending = []
        for f in all_files:
            d = json.loads(f.read_text(encoding="utf-8"))
            pid = f.stem.rsplit("_chain", 1)[0]
            ci = d["chain_index"]
            pending.append((pid, ci, d))
        skipped = 0
        print(f"分片: start={start} count={count}  →  {len(pending)} 条")
        print(flush=True)
        if workers > 1:
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
        return

    # ---- 等待 + 轮询模式 ----
    poll_interval = 5
    if expected:
        print(f"等待 {expected} 条结果就绪（每 {poll_interval}s 检查一次）...")
        print(flush=True)
        while len(sorted(RESULTS_DIR.glob("*.json"))) < expected:
            time.sleep(poll_interval)
            n_now = len(sorted(RESULTS_DIR.glob("*.json")))
            print(f"[{time.strftime('%H:%M:%S')}] 当前 {n_now}/{expected}", flush=True)

    # ---- 收集待评 ----
    all_files = sorted(RESULTS_DIR.glob("*.json"))
    if paper:
        all_files = [f for f in all_files if f.stem.startswith(paper)]

    if not all_files:
        print("无结果文件。"); return

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

    if not pending:
        summary(sorted(EVAL_DIR.glob("*.json")), expected)
        return

    if workers > 1:
        print(f"并行 worker: {workers}")
    if skipped:
        print(f"跳过 {skipped} 条  待评 {len(pending)} 条")
    else:
        print(f"共 {len(pending)} 条")
    print(flush=True)

    # 执行
    if workers > 1:
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
    summary(sorted(EVAL_DIR.glob("*.json")), expected)


# ============================================================
#  监控模式
# ============================================================

def run_watch(interval=5, expected_total=0):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=120)
    print(f"监控模式（间隔 {interval}s）", flush=True)
    if expected_total:
        print(f"期望总数: {expected_total}，满额自动退出", flush=True)
    print(flush=True)

    while True:
        all_files = sorted(RESULTS_DIR.glob("*.json"))
        pending_files = [f for f in all_files if not (EVAL_DIR / f.name).exists()]

        if not pending_files:
            n = len(list(EVAL_DIR.glob("*.json")))
            print(f"[{time.strftime('%H:%M:%S')}] 已评 {n}/{len(all_files)}", flush=True)
            if expected_total and n >= expected_total:
                break
            time.sleep(interval)
            continue

        for f in pending_files:
            d = json.loads(f.read_text(encoding="utf-8"))
            pid = f.stem.rsplit("_chain", 1)[0]
            ci = d["chain_index"]
            pid, ci, r = _eval_item(pid, ci, d, client)
            r = {"paper_id": pid, "chain_index": ci, **r}

            dest = EVAL_DIR / f.name
            tmp = dest.with_suffix(".tmp")
            tmp.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(dest)

            print(f"[{time.strftime('%H:%M:%S')}] {pid}_chain{ci}  "
                  f"V={r['vulnerability']['match']}  "
                  f"I={r['impact']['match']}  "
                  f"DR={r['dose_response']['match']}", flush=True)

        n = len(list(EVAL_DIR.glob("*.json")))
        if expected_total and n >= expected_total:
            break
        time.sleep(interval)

    result_files = sorted(EVAL_DIR.glob("*.json"))
    n = len(result_files)
    if n:
        print(f"\n✅ 全部完成 ({n}/{expected_total})", flush=True)
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
    args = p.parse_args()

    if args.watch:
        run_watch(interval=5, expected_total=args.expected)
    else:
        run_batch(limit=args.limit, paper=args.paper,
                  start=args.start, count=args.count, workers=args.workers,
                  expected=args.expected)


if __name__ == "__main__":
    main()
