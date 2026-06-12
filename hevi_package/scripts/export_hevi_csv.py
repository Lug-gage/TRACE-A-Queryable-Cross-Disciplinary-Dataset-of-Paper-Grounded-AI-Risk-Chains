"""
Export HEVI comparison CSV: ref_hevi (icml JSON) vs workflow hevi (chains from 4_consensus.json).

Supports group_*/ subdirectories from parallel pipeline runs.
"""
import argparse
import csv
import json
from pathlib import Path


def _find_subdir(parent, prefix):
    """Find first subdirectory matching prefix, also looking inside group_* subdirs."""
    if not parent.exists():
        return parent / "NOT_FOUND"
    # Collect from both direct and group_*
    all_dirs = sorted(d for d in parent.iterdir() if d.is_dir() and d.name.startswith(prefix))
    for g in sorted(d for d in parent.iterdir() if d.is_dir() and d.name.startswith("group_")):
        all_dirs.extend(sorted(d for d in g.iterdir() if d.is_dir() and d.name.startswith(prefix)))
    return all_dirs[0] if all_dirs else parent / "NOT_FOUND"


def _find_paper_dirs(wf_dir, groups=None):
    """List all icml_* paper dirs inside wf_dir, traversing group_* subdirs."""
    result = []
    for d in wf_dir.iterdir():
        if d.is_dir():
            if d.name.startswith("icml_"):
                result.append(d)
            elif d.name.startswith("group_"):
                if groups and d.name not in groups:
                    continue
                result.extend(sorted(
                    sd for sd in d.iterdir() if sd.is_dir() and sd.name.startswith("icml_")
                ))
    return sorted(result)


def _find_extract_json(icml_dir, paper_id):
    """Find extraction JSON by paper_id, may be nested under group_*/."""
    p = icml_dir / f"{paper_id}.json"
    if p.exists():
        return p
    for g in sorted(d for d in icml_dir.iterdir() if d.is_dir() and d.name.startswith("group_")):
        p = g / f"{paper_id}.json"
        if p.exists():
            return p
    return None


def _resolve_dirs():
    """Resolve input/output directories lazily."""
    workflow_dir = Path("outputs/hevi_workflow")
    icml_dir = _find_subdir(workflow_dir, "hevi_icml_")
    wf_dir = _find_subdir(workflow_dir, "hevi_")
    if icml_dir == wf_dir:
        wf_dir = workflow_dir / "hevi_deepseek-v4-pro"
    output = workflow_dir / "hevi_comparison.csv"
    return icml_dir, wf_dir, output

HEVI_SLOTS = ["hazard", "exposure", "dose_response", "vulnerability", "impact", "key_control_nodes"]


def fmt_items(items):
    """Format list items as '1. xxx\\n2. xxx'."""
    if not items:
        return ""
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1))


def main():
    parser = argparse.ArgumentParser(description="Export HEVI comparison CSV")
    parser.add_argument("--group", type=int, default=0,
                        help="Only include this group number (1-6, 0 = all)")
    args = parser.parse_args()
    groups = [f"group_{args.group}"] if args.group else None

    ICML_DIR, WF_DIR, OUTPUT = _resolve_dirs()
    rows = []
    paper_dirs = _find_paper_dirs(WF_DIR, groups=groups)
    paper_ids = sorted(d.name for d in paper_dirs)

    for pipe_dir in paper_dirs:
        pid = pipe_dir.name
        row = {"paper_id": pid}

        # ── Paper info + ref_hevi ──
        icml_file = _find_extract_json(ICML_DIR, pid)
        if icml_file:
            icml = json.loads(icml_file.read_text(encoding="utf-8"))
            row["title"] = icml.get("title", "")
            row["abstract"] = icml.get("abstract", "")
            row["impact"] = icml.get("impact", "")
            ref_hevi = icml.get("ref_hevi", {})
            for slot in HEVI_SLOTS:
                row[f"ref_{slot}"] = fmt_items(ref_hevi.get(slot, []))
        else:
            row["title"] = row["abstract"] = row["impact"] = ""
            for slot in HEVI_SLOTS:
                row[f"ref_{slot}"] = ""

        # ── Workflow HEVI (from chains, deduplicated) ──
        cons_file = pipe_dir / "4_consensus.json"
        if cons_file.exists():
            cons = json.loads(cons_file.read_text(encoding="utf-8"))
            chains = cons.get("chains", [])

            for slot in HEVI_SLOTS:
                items = []
                for ch in chains:
                    for v in ch.get(slot, []):
                        if isinstance(v, str):
                            items.append(v)
                        elif isinstance(v, dict):
                            items.append(v.get(slot, v.get("hazard", str(v))))
                # Dedup preserving order
                seen = set()
                deduped = []
                for item in items:
                    key = item.strip().lower()
                    if key not in seen:
                        seen.add(key)
                        deduped.append(item)
                row[f"wf_{slot}"] = fmt_items(deduped)

            # ── Chains detail ──
            chain_parts = []
            for i, ch in enumerate(chains, 1):
                lines = [f"=== Chain {i} ==="]
                lines.append(f"场景: {ch.get('scenario', '')}")
                lines.append(f"问题: {ch.get('issue', '')}")
                for slot in HEVI_SLOTS:
                    val = ch.get(slot, [])
                    if val:
                        lines.append(f"\n{slot}:")
                        for j, v in enumerate(val, 1):
                            lines.append(f"  {j}. {v}")
                lines.append(f"\n置信度: {ch.get('confidence', '')}")
                chain_parts.append("\n".join(lines))
            row["chains"] = "\n\n".join(chain_parts)
        else:
            for slot in HEVI_SLOTS:
                row[f"wf_{slot}"] = ""
            row["chains"] = ""

        # ── Recall ──
        cmp_file = pipe_dir / "5_compare.json"
        if cmp_file.exists():
            cmp = json.loads(cmp_file.read_text(encoding="utf-8"))
            row["item_recall"] = cmp.get("item_recall", "")
        else:
            row["item_recall"] = ""

        rows.append(row)

    columns = (
        ["paper_id", "title", "abstract", "impact", "item_recall"]
        + [f"ref_{s}" for s in HEVI_SLOTS]
        + [f"wf_{s}" for s in HEVI_SLOTS]
        + ["chains"]
    )

    with open(OUTPUT, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} papers to {OUTPUT}")


if __name__ == "__main__":
    main()
