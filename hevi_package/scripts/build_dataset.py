#!/usr/bin/env python3
"""Merge extraction outputs + pipeline consensus chains into a single dataset JSON.

Base: outputs/hevi_workflow/hevi_deepseek-v4-pro/  (pipeline results)
Lookup: outputs/hevi_workflow/hevi_icml_deepseek-v4-pro/  (extraction: title, abstract, impact, ref_hevi)

Output: outputs/dataset.json — one record per paper that has pipeline results.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

HEVI_KEYS = ["hazard", "exposure", "dose_response", "vulnerability", "impact", "key_control_nodes"]

PKG_ROOT = Path(__file__).resolve().parents[1]
EXTRACT_DIR = PKG_ROOT / "outputs" / "hevi_workflow" / "hevi_icml_deepseek-v4-pro"
PIPELINE_DIR = PKG_ROOT / "outputs" / "hevi_workflow" / "hevi_deepseek-v4-pro"
OUT_PATH = PKG_ROOT / "outputs" / "dataset.json"


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize_hevi_six(value: Dict[str, Any]) -> Dict[str, List[str]]:
    """Ensure all six HEVI keys are present as lists of strings."""
    out = {}
    for key in HEVI_KEYS:
        items = value.get(key, [])
        if isinstance(items, list):
            out[key] = [str(item).strip() for item in items if str(item).strip()]
        else:
            out[key] = []
    return out


def extract_chain(chain: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only the HEVI fields + scenario/issue. Drop pipeline metadata."""
    out: Dict[str, Any] = {
        "scenario": str(chain.get("scenario", "")).strip(),
        "issue": str(chain.get("issue", "")).strip(),
    }
    for key in HEVI_KEYS:
        items = chain.get(key, [])
        if isinstance(items, list):
            out[key] = [str(item).strip() for item in items if str(item).strip()]
        else:
            out[key] = []
    return out


def build_dataset() -> None:
    pipeline_dirs = sorted(
        d for d in PIPELINE_DIR.iterdir()
        if d.is_dir() and d.name.startswith("icml_")
    )
    print(f"Pipeline papers: {len(pipeline_dirs)}")

    dataset: List[Dict[str, Any]] = []
    skipped_no_extraction = 0
    skipped_no_consensus = 0

    for pipe_dir in pipeline_dirs:
        paper_id = pipe_dir.name

        # Look up extraction data
        ext_path = EXTRACT_DIR / f"{paper_id}.json"
        if not ext_path.exists():
            skipped_no_extraction += 1
            print(f"  {paper_id} — no extraction JSON, skipped")
            continue

        # Look up consensus
        consensus_path = pipe_dir / "4_consensus.json"
        if not consensus_path.exists():
            skipped_no_consensus += 1
            print(f"  {paper_id} — no 4_consensus.json, skipped")
            continue

        ext_data = load_json(ext_path)
        consensus = load_json(consensus_path)

        chains_raw = consensus.get("chains", [])
        chains = [extract_chain(c) for c in chains_raw if isinstance(c, dict)]

        record: Dict[str, Any] = {
            "paper_id": paper_id,
            "title": ext_data.get("title", ""),
            "abstract": ext_data.get("abstract", ""),
            "impact": ext_data.get("impact", ""),
            "ref_hevi": normalize_hevi_six(ext_data.get("ref_hevi", {})),
            "workflow_chains": chains,
        }
        dataset.append(record)
        print(f"  {paper_id} — {len(chains)} chains")

    # Write
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    # Summary
    total_chains = sum(len(r["workflow_chains"]) for r in dataset)
    total_ref_items = sum(sum(len(v) for v in r["ref_hevi"].values()) for r in dataset)
    total_wf_items = sum(
        sum(sum(len(v) for k, v in chain.items() if k in HEVI_KEYS) for chain in r["workflow_chains"])
        for r in dataset
    )

    print(f"\n=== Dataset ===")
    print(f"  Papers:              {len(dataset)}")
    print(f"  Skipped (no extract):{skipped_no_extraction}")
    print(f"  Skipped (no consensus):{skipped_no_consensus}")
    print(f"  Total chains:        {total_chains}")
    print(f"  Total ref_hevi items:{total_ref_items}")
    print(f"  Total wf HEVI items: {total_wf_items}")
    print(f"  Output:              {OUT_PATH}")


if __name__ == "__main__":
    build_dataset()
