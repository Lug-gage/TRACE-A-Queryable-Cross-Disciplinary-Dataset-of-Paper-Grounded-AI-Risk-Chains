"""
Export HEVI comparison CSV: ref_hevi (icml JSON) vs workflow hevi (chains from 4_consensus.json).
"""
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _find_subdir(parent, prefix):
    """Find first subdirectory matching prefix."""
    if not parent.exists():
        return parent / "NOT_FOUND"
    dirs = sorted([d for d in parent.iterdir() if d.is_dir() and d.name.startswith(prefix)])
    return dirs[0] if dirs else parent / "NOT_FOUND"


def _resolve_dirs():
    """Resolve input/output directories lazily."""
    workflow_dir = REPO_ROOT / "outputs" / "hevi_workflow"
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
    ICML_DIR, WF_DIR, OUTPUT = _resolve_dirs()
    rows = []
    paper_ids = sorted(
        [d.name for d in WF_DIR.iterdir() if d.is_dir() and d.name.startswith("icml_")],
    )

    for pid in paper_ids:
        row = {"paper_id": pid}

        # ── Paper info + ref_hevi ──
        icml_file = ICML_DIR / f"{pid}.json"
        if icml_file.exists():
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
        cons_file = WF_DIR / pid / "4_consensus.json"
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
        cmp_file = WF_DIR / pid / "5_compare.json"
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
