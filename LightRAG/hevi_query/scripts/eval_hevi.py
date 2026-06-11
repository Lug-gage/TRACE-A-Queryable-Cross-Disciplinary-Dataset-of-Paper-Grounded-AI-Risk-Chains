"""
HEVI 结果评价

将 lightrag_result 与 reference 逐条对比，LLM 评判 covered/partial/not，
每条 chain 独立 eval 文件，最后覆盖率汇总。

用法:
    python hevi_query/scripts/eval_hevi.py                          # 全部串行
    python hevi_query/scripts/eval_hevi.py --workers 3              # 3 线程并行
    python hevi_query/scripts/eval_hevi.py --limit 5                # 前 5 条
    python hevi_query/scripts/eval_hevi.py --paper icml_2024_0001   # 指定论文
    python hevi_query/scripts/eval_hevi.py --start 0 --count 100    # 分片并行
    python hevi_query/scripts/eval_hevi.py --watch                  # 监控模式：出来一篇评一篇
    python hevi_query/scripts/eval_hevi.py --watch --workers 3      # 监控 + 3 线程并行评价
    python hevi_query/scripts/eval_hevi.py --watch --expected 818   # 满额自动退出
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

RESULTS_DIR  = BASE_DIR / "lightrag_results"
EVAL_DIR     = BASE_DIR / "evaluation"; EVAL_DIR.mkdir(exist_ok=True)
EVAL_PROMPT  = (SCRIPTS_DIR / "eval_prompt.txt").read_text(encoding="utf-8")

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
    gen  = data["lightrag_result"]
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
#  主流程（一次性）
# ============================================================

def run_batch(limit=0, paper=None, start=-1, count=0, workers=1):
    all_files = sorted(RESULTS_DIR.glob("*.json"))
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
            pending.append((f.stem.rsplit("_chain", 1)[0], d["chain_index"], d))
        skipped = 0
    else:
        pending = []; skipped = 0
        for f in all_files:
            if (EVAL_DIR / f.name).exists():
                skipped += 1; continue
            d = json.loads(f.read_text(encoding="utf-8"))
            pending.append((f.stem.rsplit("_chain", 1)[0], d["chain_index"], d))
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
        summary(sorted(EVAL_DIR.glob("*.json")), len(all_files))
        return

    # ---- 执行 ----
    if workers > 1 and not parallel:
        chunks = [[] for _ in range(workers)]
        for i, item in enumerate(pending):
            chunks[i % workers].append(item)
        print(f"分组: {' | '.join(f'W{w}={len(c)}条' for w, c in enumerate(chunks))}")
        print(flush=True)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_worker, chunk, wid): wid
                       for wid, chunk in enumerate(chunks)}
            n_total = 0
            for f in as_completed(futures):
                n_total += f.result()
    else:
        n_total = _run_worker(pending, 0)

    summary(sorted(EVAL_DIR.glob("*.json")), len(all_files))


# ============================================================
#  监控模式（持续轮询，出来一篇评一篇，支持多线程并行评价）
# ============================================================

def run_watch(interval=5, expected_total=0, workers=1):
    print(f"🔍 监控模式启动（扫描间隔 {interval}s）")
    if workers > 1:
        print(f"   评价线程: {workers}（并行评价新文件）")
    if expected_total:
        print(f"   期望总数: {expected_total}，达到后自动退出")
    print(f"   结果目录: {RESULTS_DIR}")
    print(f"   评价输出: {EVAL_DIR}")
    print()

    while True:
        all_files = sorted(RESULTS_DIR.glob("*.json"))
        if not all_files:
            print(f"[{time.strftime('%H:%M:%S')}] 暂无结果文件，等待...", flush=True)
            time.sleep(interval)
            continue

        # 找出未评价的
        pending_files = [f for f in all_files if not (EVAL_DIR / f.name).exists()]

        if not pending_files:
            n_evaled = len(list(EVAL_DIR.glob("*.json")))
            msg = f"[{time.strftime('%H:%M:%S')}] 全部已评 ({n_evaled}/{len(all_files)})"
            if not expected_total or n_evaled < expected_total:
                msg += " → 等待新结果..."
            print(msg, flush=True)
            if expected_total and n_evaled >= expected_total:
                print("\n✅ 全部评价完成！")
                summary(sorted(EVAL_DIR.glob("*.json")), expected_total)
                break
            time.sleep(interval)
            continue

        # 解析 pending 文件
        pending = []
        for f in pending_files:
            d = json.loads(f.read_text(encoding="utf-8"))
            pending.append((f.stem.rsplit("_chain", 1)[0], d["chain_index"], d))

        # 单线程 or 多线程
        if workers > 1:
            chunks = [[] for _ in range(workers)]
            for i, item in enumerate(pending):
                chunks[i % workers].append(item)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_run_worker, chunk, wid): wid
                           for wid, chunk in enumerate(chunks)}
                for f in as_completed(futures):
                    f.result()
        else:
            _run_worker(pending, 0)

        # 检查是否全完成
        eval_files = sorted(EVAL_DIR.glob("*.json"))
        if expected_total and len(eval_files) >= expected_total:
            print(f"\n✅ 全部评价完成！（{len(eval_files)}/{expected_total}）")
            summary(eval_files, expected_total)
            break

        time.sleep(interval)


# ============================================================
def main():
    p = argparse.ArgumentParser(description="HEVI 评价")
    p.add_argument("--limit", "-l", type=int, default=0)
    p.add_argument("--paper", "-p")
    p.add_argument("--start", "-s", type=int, default=-1)
    p.add_argument("--count", "-c", type=int, default=0)
    p.add_argument("--workers", "-w", type=int, default=1,
                   help="并行线程数（分批模式自动分片，监控模式并行评新文件）")
    p.add_argument("--watch", action="store_true",
                   help="监控模式：持续扫描新结果文件，出来一篇评一篇")
    p.add_argument("--interval", type=int, default=5,
                   help="监控模式扫描间隔（秒），默认 5")
    p.add_argument("--expected", type=int, default=818,
                   help="期望评价总数，达到后自动退出（默认 818）")
    args = p.parse_args()

    if args.watch:
        run_watch(interval=args.interval, expected_total=args.expected,
                  workers=args.workers)
    else:
        run_batch(limit=args.limit, paper=args.paper, start=args.start,
                  count=args.count, workers=args.workers)


if __name__ == "__main__":
    main()
