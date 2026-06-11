"""
联合分布分析：LLM 评价 (covered/partial/not) × Embedding 相似度 (0~1)

读取 evaluation/ + evaluation_embedding/，按 paper_id+chain_index 配对，
分析每个字段的相似度在各 match 档位下的分布，生成箱线图。

用法:
    python hevi_query/scripts/analyze_joint.py
    python hevi_query/scripts/analyze_joint.py --eval-dir evaluation
    python hevi_query/scripts/analyze_joint.py --emb-dir evaluation_embedding_llm
"""
import json, argparse, sys
from pathlib import Path
from collections import defaultdict

SCRIPTS_DIR = Path(__file__).resolve().parent
BASE_DIR    = SCRIPTS_DIR.parent

FIELDS = ("vulnerability", "impact", "dose_response")
MATCHES = ("covered", "partial", "not")


# ============================================================
#  加载 & 配对
# ============================================================

def load_pairs(eval_dir, emb_dir):
    """加载两个目录的结果，按 paper_id+chain_index 配对"""
    pairs = []
    eval_files = sorted(eval_dir.glob("*.json"))
    if not eval_files:
        print(f"无文件: {eval_dir}")
        sys.exit(1)

    for ef in eval_files:
        ef_data = json.loads(ef.read_text(encoding="utf-8"))
        pid = ef_data["paper_id"]
        ci = ef_data["chain_index"]

        emb_path = emb_dir / f"{pid}_chain{ci}.json"
        if not emb_path.exists():
            continue
        emb_data = json.loads(emb_path.read_text(encoding="utf-8"))

        pairs.append((pid, ci, ef_data, emb_data))

    return pairs


# ============================================================
#  分析
# ============================================================

def analyze(pairs):
    # 每个字段各 match 档位的相似度列表
    data = {f: {m: [] for m in MATCHES} for f in FIELDS}

    for pid, ci, ef, emb in pairs:
        for f in FIELDS:
            match = ef.get(f, {}).get("match", "not")
            sim = emb.get(f, {}).get("similarity", 0)
            if match not in data[f]:
                match = "not"
            data[f][match].append(sim)

    return data


# ============================================================
#  打印
# ============================================================

def print_stats(data, n_pairs):
    print(f" 配对: {n_pairs} 条")
    print()

    # 表头
    print(f"  {'字段':16s}  {'match':>8s}  {'n':>5s}  {'mean':>7s}  {'median':>7s}  {'std':>7s}  {'min':>7s}  {'max':>7s}")
    print("  " + "-" * 65)

    for f in FIELDS:
        for m in MATCHES:
            vals = data[f][m]
            if not vals:
                continue
            n = len(vals)
            mean = sum(vals) / n
            median = sorted(vals)[n // 2]
            std = (sum((v - mean)**2 for v in vals) / n) ** 0.5
            print(f"  {f:16s}  {m:>8s}  {n:>5d}  {mean:>7.4f}  {median:>7.4f}  {std:>7.4f}  {min(vals):>7.4f}  {max(vals):>7.4f}")
        print()

    # 阈值建议
    print(f"  {'='*60}")
    print("  阈值校准建议（median 加权）")
    print(f"  {'='*60}")
    for f in FIELDS:
        covered_vals = data[f]["covered"]
        partial_vals = data[f]["partial"]
        not_vals    = data[f]["not"]

        if covered_vals and partial_vals:
            cv = sorted(covered_vals)[len(covered_vals) // 2]
            pv = sorted(partial_vals)[len(partial_vals) // 2]
            # covered_cutoff: (median_partial + median_covered) / 2
            # partial_cutoff: (median_not + median_partial) / 2
            cc = round((pv + cv) / 2, 4)
        else:
            cc = None

        if partial_vals and not_vals:
            nv = sorted(not_vals)[len(not_vals) // 2]
            pc = round((nv + pv) / 2, 4) if covered_vals and partial_vals else None
        else:
            pc = None

        print(f"  {f:16s}  covered ≥ {cc}  |  partial ≥ {pc}  |  not < {pc}" if cc and pc else
              f"  {f:16s}  (数据不足)")

    # 重叠情况
    print(f"\n  {'='*60}")
    print("  档位重叠度 (covered min ~ not max)")
    print(f"  {'='*60}")
    for f in FIELDS:
        cv = data[f]["covered"]
        nv = data[f]["not"]
        if cv and nv:
            print(f"  {f:16s}  covered=[{min(cv):.4f}, {max(cv):.4f}]  "
                  f"partial=[{min(data[f]['partial']):.4f}, {max(data[f]['partial']):.4f}]  "
                  f"not=[{min(nv):.4f}, {max(nv):.4f}]")


# ============================================================
#  箱线图
# ============================================================

def plot(data, n_pairs, eval_dir, emb_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    png_dir = eval_dir.parent / "analysis_joint"
    png_dir.mkdir(exist_ok=True)

    colors = {"covered": "#55A868", "partial": "#F0C571", "not": "#C44E52"}

    # ---- 箱线图 ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, f in zip(axes, FIELDS):
        all_data = [data[f]["covered"], data[f]["partial"], data[f]["not"]]
        bp = ax.boxplot(all_data, patch_artist=True, widths=0.5,
                        medianprops={"color": "#333", "linewidth": 1.5})
        for patch, m in zip(bp["boxes"], MATCHES):
            patch.set_facecolor(colors[m])
            patch.set_alpha(0.85)
        for m, vals in zip(MATCHES, all_data):
            if vals:
                ax.scatter(np.ones(len(vals)) * (MATCHES.index(m) + 1) + np.random.uniform(-0.12, 0.12, len(vals)),
                           vals, s=3, alpha=0.15, color=colors[m])

        ax.set_title(f.replace("_", "\n"), fontsize=12, fontweight="bold")
        ax.set_xticklabels(["covered", "partial", "not"], fontsize=9)
        ax.set_ylabel("Cosine Similarity" if ax == axes[0] else "", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.2)

        # 标注 n
        for i, vals in enumerate(all_data):
            if vals:
                mean = sum(vals) / len(vals)
                ax.text(i + 1, 1.02, f"n={len(vals)}\nμ={mean:.3f}",
                        ha="center", va="bottom", fontsize=7, color="#555")

    fig.suptitle(f"Joint Distribution: LLM Match × Embedding Similarity (n={n_pairs})",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(png_dir / "boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 相似度阈值映射 ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, f in zip(axes, FIELDS):
        bins = np.linspace(0, 1, 21)  # 0, 0.05, ..., 1.0
        for m, color in zip(MATCHES, [colors["covered"], colors["partial"], colors["not"]]):
            vals = data[f][m]
            if vals:
                ax.hist(vals, bins=bins, alpha=0.5, color=color, label=f"{m} (n={len(vals)})",
                        edgecolor="white", linewidth=0.3)
        ax.set_title(f.replace("_", "\n"), fontsize=12, fontweight="bold")
        ax.set_xlabel("Cosine Similarity", fontsize=9)
        ax.set_ylabel("Count" if ax == axes[0] else "", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle(f"Similarity Distribution by Match Category (n={n_pairs})",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(png_dir / "hist_by_match.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n  📊 图表 → {png_dir}/")


# ============================================================
def main():
    p = argparse.ArgumentParser(description="联合分布分析")
    p.add_argument("--eval-dir", default="evaluation",
                   help="LLM 评价目录名（默认 evaluation）")
    p.add_argument("--emb-dir", default="evaluation_embedding",
                   help="Embedding 评价目录名（默认 evaluation_embedding）")
    args = p.parse_args()

    eval_dir = BASE_DIR / args.eval_dir
    emb_dir  = BASE_DIR / args.emb_dir

    print(f"LLM 评价:     {eval_dir}")
    print(f"Embedding:  {emb_dir}")

    pairs = load_pairs(eval_dir, emb_dir)
    data = analyze(pairs)
    print_stats(data, len(pairs))
    plot(data, len(pairs), eval_dir, emb_dir)


if __name__ == "__main__":
    main()
