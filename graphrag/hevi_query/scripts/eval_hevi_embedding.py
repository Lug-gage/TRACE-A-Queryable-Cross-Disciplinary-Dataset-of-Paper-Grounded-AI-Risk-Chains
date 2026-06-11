"""
HEVI 结果评价 — Embedding 相似度版（确定性，无随机性）

用 text-embedding-3-large 对 reference 和 generated 分别向量化，
计算 cosine similarity（0~1），后续可设阈值分档。

用法:
    python hevi_query/scripts/eval_hevi_embedding.py                          # 全部串行
    python hevi_query/scripts/eval_hevi_embedding.py --workers 5              # 5 线程并行
    python hevi_query/scripts/eval_hevi_embedding.py --limit 5                # 前 5 条
    python hevi_query/scripts/eval_hevi_embedding.py --paper icml_2024_0001   # 指定论文
    python hevi_query/scripts/eval_hevi_embedding.py --results-dir llm_results
"""
import json, argparse, logging, math, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.WARNING, force=True)
for _n in ("openai", "httpx", "asyncio", "urllib3", "requests", "httpcore"):
    logging.getLogger(_n).setLevel(logging.WARNING)

from openai import OpenAI

# ---- 配置 ----
SCRIPTS_DIR = Path(__file__).resolve().parent           # hevi_query/scripts/
BASE_DIR    = SCRIPTS_DIR.parent                         # hevi_query/

# api_key.txt
_key_path = BASE_DIR / "api_key.txt"
if not _key_path.exists():
    _key_path = BASE_DIR.parent / "api_key.txt"
if not _key_path.exists():
    raise FileNotFoundError("api_key.txt 未找到")
API_KEY = _key_path.read_text().strip()

BASE_URL       = "https://www.highland-api.top/v1"
EMBED_MODEL    = "text-embedding-3-large"

RESULTS_DIR    = BASE_DIR / "graph_result"   # 默认
EVAL_DIR       = BASE_DIR / "evaluation_embedding"

FIELDS = ("vulnerability", "impact", "dose_response")


# ============================================================
#  helpers
# ============================================================

def _ref(r, field):
    v = r.get(field, [])
    return v[0] if isinstance(v, list) and v else str(v)


def cosine(a, b):
    """cosine similarity between two vectors"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ============================================================
#  Embedding 调用（批量，5 次重试）
# ============================================================

def _embed_batch(client, texts):
    """批量获取 embedding，自动重试"""
    for attempt in range(1, 6):
        try:
            resp = client.embeddings.create(
                model=EMBED_MODEL, input=texts, timeout=60)
            return [d.embedding for d in resp.data]
        except Exception as e:
            if attempt == 5: raise
            time.sleep(min(2 ** attempt, 30))


# ============================================================
#  单条评价
# ============================================================

def _eval_item(pid, ci, data, client):
    """评价一条 chain：6 个文本批量 embedding → cosine"""
    gen = data["graphrag_result"]
    ref = data["reference"]

    ref_texts = {f: _ref(ref, f) for f in FIELDS}
    gen_texts = {f: gen.get(f, "") for f in FIELDS}

    # 收集所有需要 embedding 的文本
    texts = []
    order = []
    for f in FIELDS:
        texts.append(ref_texts[f])
        texts.append(gen_texts[f])
        order.append(f)

    # 批量调用（6 个文本一次请求）
    embeddings = _embed_batch(client, texts)

    result = {}
    for i, f in enumerate(order):
        ref_vec = embeddings[i * 2]
        gen_vec = embeddings[i * 2 + 1]
        sim = round(cosine(ref_vec, gen_vec), 4)
        result[f] = {
            "similarity": sim,
            "reference": ref_texts[f],
            "generated": gen_texts[f],
        }

    return pid, ci, result


# ============================================================
#  Worker
# ============================================================

def _run_worker(chunk, wid):
    """每个 worker 独立 OpenAI client，处理分到的链"""
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=60)
    n = 0
    for pid, ci, data in chunk:
        try:
            pid, ci, r = _eval_item(pid, ci, data, client)
        except Exception as e:
            gen = data["graphrag_result"]
            ref = data["reference"]
            r = {f: {"similarity": 0.0, "reference": _ref(ref, f),
                     "generated": gen.get(f, ""), "error": str(e)}
                 for f in FIELDS}

        r = {"paper_id": pid, "chain_index": ci, **r}

        dest = EVAL_DIR / f"{pid}_chain{ci}.json"
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(dest)

        n += 1
        sims = "  ".join(f"{f[0].upper()}={r[f]['similarity']:.3f}" for f in FIELDS)
        print(f"  [W{wid}] {pid}_chain{ci}  {sims}", flush=True)
    return n


# ============================================================
#  汇总
# ============================================================

def summary(eval_files, total_available=0):
    if not eval_files:
        return
    all_sims = {f: [] for f in FIELDS}
    buckets = {f: [0] * 10 for f in FIELDS}
    n = len(eval_files)
    for fp in eval_files:
        r = json.loads(fp.read_text(encoding="utf-8"))
        for f in FIELDS:
            sim = r.get(f, {}).get("similarity", 0)
            all_sims[f].append(sim)
            b = min(int(sim * 10), 9)
            buckets[f][b] += 1

    print(f"\n{'─'*65}")
    info = f" ({n}" + (f" / {total_available}" if total_available else "") + " 已评)"
    print(f" 汇总{info}")
    print(f"{'─'*65}")

    labels = ("vulnerability", "impact", "dose_response")
    label_w = max(len("dose_response"), max(len(l) for l in labels))

    # 表头
    hdr = f"  {'':{label_w}s}  {'min':>8s}  {'Q25':>8s}  {'median':>8s}  {'Q75':>8s}  {'max':>8s}  {'mean':>8s}"
    sep = f"  {'─'*label_w}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}"
    print(hdr)
    print(sep)

    for f in FIELDS:
        vals = sorted(all_sims[f])
        nf = len(vals)
        q25 = vals[nf // 4]
        q50 = vals[nf // 2]
        q75 = vals[nf * 3 // 4]
        mean = sum(vals) / nf
        print(f"  {f:{label_w}s}  {vals[0]:8.4f}  {q25:8.4f}  {q50:8.4f}  {q75:8.4f}  {vals[-1]:8.4f}  {mean:8.4f}")

    all_vals = sorted(s for f in FIELDS for s in all_sims[f])
    nt = len(all_vals)
    overall_label = "overall"
    print(sep)
    print(f"  {overall_label:{label_w}s}  {all_vals[0]:8.4f}  {all_vals[nt//4]:8.4f}  {all_vals[nt//2]:8.4f}  {all_vals[nt*3//4]:8.4f}  {all_vals[-1]:8.4f}  {sum(all_vals)/nt:8.4f}")

    # ---- 柱状图 ----
    _plot_distribution(buckets, n, EVAL_DIR)


def _plot_distribution(buckets, n, eval_dir):
    """生成相似度分布柱状图，输出到 evaluation_embedding/png/"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    png_dir = eval_dir / "png"
    png_dir.mkdir(exist_ok=True)

    labels = [f"[{i/10:.1f},{i/10+.1:.1f})" for i in range(9)] + ["[0.9,1.0]"]
    x = np.arange(len(labels))
    width = 0.25
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    names = {"vulnerability": "Vulnerability", "impact": "Impact", "dose_response": "Dose-Response"}

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, f in enumerate(FIELDS):
        bars = ax.bar(x + i * width, buckets[f], width, label=names[f], color=colors[i],
                      edgecolor="white", linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + max(buckets[f]) * 0.01,
                        str(h), ha="center", va="bottom", fontsize=7, color="#333")

    ax.set_xlabel("Cosine Similarity", fontsize=12, labelpad=10)
    ax.set_ylabel("Count", fontsize=12, labelpad=10)
    ax.set_title(f"HEVI Embedding Similarity Distribution (n={n})", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=11, loc="upper left")
    ax.set_ylim(0, max(max(buckets[f]) for f in FIELDS) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    # avg 标注
    for i, f in enumerate(FIELDS):
        vals = [buckets[f][j] for j in range(10)]
        centers = [j * 10 + 5 for j in range(10)]  # bin center in 0-100 scale
        total = sum(vals)
        weighted = sum(v * (j * 10 + 5) for j, v in enumerate(vals))
        avg = weighted / total if total else 0
        ax.axvline(x=avg / 10 - 0.5 + i * width + width / 2, color=colors[i],
                   linestyle="--", linewidth=1, alpha=0.6)
        ax.text(avg / 10 - 0.5 + i * width + width / 2, max(buckets[f]) * 0.95,
                f"avg={avg/100:.2f}", color=colors[i], fontsize=8,
                ha="center", va="top", rotation=90)

    plt.tight_layout()
    fig.savefig(png_dir / "distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  📊 柱状图 → {png_dir / 'distribution.png'}")


# ============================================================
#  主流程
# ============================================================

def _resolve_src_dir(results_dir):
    if not results_dir:
        return RESULTS_DIR
    p = Path(results_dir)
    if not p.is_absolute() and not p.exists():
        p = BASE_DIR / results_dir
    return p


def run_batch(limit=0, paper=None, start=-1, count=0, workers=1, results_dir=None):
    global EVAL_DIR
    # 推导 eval 输出目录
    if results_dir:
        name = Path(results_dir).name
        if name == "graph_result":
            EVAL_DIR = BASE_DIR / "evaluation_embedding"
        else:
            suffix = name.replace("_result", "").replace("graph_", "").strip("_")
            EVAL_DIR = BASE_DIR / f"evaluation_embedding_{suffix}" if suffix else BASE_DIR / "evaluation_embedding"
    else:
        EVAL_DIR = BASE_DIR / "evaluation_embedding"
    EVAL_DIR.mkdir(exist_ok=True)

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
    print(f"模型: {EMBED_MODEL}  |  输出: {EVAL_DIR}", flush=True)
    print(flush=True)

    if not pending:
        summary(sorted(EVAL_DIR.glob("*.json")))
        return

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
def main():
    p = argparse.ArgumentParser(description="HEVI 评价 (Embedding 相似度)")
    p.add_argument("--limit", "-l", type=int, default=0)
    p.add_argument("--paper", "-p")
    p.add_argument("--start", "-s", type=int, default=-1)
    p.add_argument("--count", "-c", type=int, default=0)
    p.add_argument("--workers", "-w", type=int, default=1,
                   help="并行 worker 数")
    p.add_argument("--results-dir", "-r", default=None,
                   help="结果目录（默认 graph_result）")
    args = p.parse_args()
    run_batch(limit=args.limit, paper=args.paper,
              start=args.start, count=args.count, workers=args.workers,
              results_dir=args.results_dir)


if __name__ == "__main__":
    main()
