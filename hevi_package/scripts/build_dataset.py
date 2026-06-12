#!/usr/bin/env python3
"""Merge extraction outputs + pipeline consensus chains into a single dataset JSON.

Base: outputs/hevi_workflow/hevi_deepseek-v4-pro/  (pipeline results)
      Results may be nested inside group_*/ subdirectories (parallel runs).
Lookup: outputs/hevi_workflow/hevi_icml_deepseek-v4-pro/  (extraction: title, abstract, impact, ref_hevi)

Output: outputs/dataset.json — one record per paper that has pipeline results.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

HEVI_KEYS = ["hazard", "exposure", "dose_response", "vulnerability", "impact", "key_control_nodes"]

EXTRACT_DIR = Path("outputs/hevi_workflow/hevi_icml_deepseek-v4-pro")
PIPELINE_DIR = Path("outputs/hevi_workflow/hevi_deepseek-v4-pro")
OUT_PATH = Path("outputs/dataset.json")


def _subdirs(base: Path, prefix: str = "") -> List[Path]:
    """Collect icml_* subdirs from base, also traversing group_*/ subdirs."""
    result: List[Path] = []
    for d in base.iterdir():
        if d.is_dir():
            if d.name.startswith(prefix):
                result.append(d)
            elif d.name.startswith("group_"):
                result.extend(sorted(
                    sd for sd in d.iterdir() if sd.is_dir() and sd.name.startswith(prefix)
                ))
    return sorted(result)


def _find_extract(base: Path, paper_id: str) -> Optional[Path]:
    """Find an extraction JSON for paper_id, checking groups when needed."""
    # Direct
    direct = base / f"{paper_id}.json"
    if direct.exists():
        return direct
    # Under group_*
    for g in sorted(d for d in base.iterdir() if d.is_dir() and d.name.startswith("group_")):
        p = g / f"{paper_id}.json"
        if p.exists():
            return p
    return None


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


def extract_chain(chain: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract only the HEVI fields + scenario/issue. Drop pipeline metadata.

    Returns None when scenario, issue, hazard, or exposure is empty — the chain is dropped.
    """
    scenario = str(chain.get("scenario", "")).strip()
    issue = str(chain.get("issue", "")).strip()
    if not scenario or not issue:
        return None
    out: Dict[str, Any] = {"scenario": scenario, "issue": issue}
    for key in HEVI_KEYS:
        items = chain.get(key, [])
        if isinstance(items, list):
            out[key] = [str(item).strip() for item in items if str(item).strip()]
        else:
            out[key] = []
    # Drop chain when hazard or exposure is empty (CS-side incomplete)
    if not out["hazard"] or not out["exposure"]:
        return None
    return out


def _count_non_empty_slots(ref_hevi: Dict[str, List[str]]) -> int:
    """Count how many of the 6 HEVI slots have at least one item."""
    return sum(1 for key in HEVI_KEYS if ref_hevi.get(key))


def build_dataset(groups: Optional[List[str]] = None, min_slots: int = 3) -> None:
    """Build dataset from pipeline results.

    Args:
        groups: Optional list of group dir names to include (e.g. ['group_1', 'group_2']).
                None means all groups.
        min_slots: Minimum number of non-empty ref_hevi slots required (default 3, i.e. >2).
    """
    all_pipeline_dirs = _subdirs(PIPELINE_DIR, prefix="icml_")
    pipeline_dirs = all_pipeline_dirs
    if groups and all_pipeline_dirs:
        pipeline_dirs = sorted(
            d for d in all_pipeline_dirs
            if d.parent.name in groups
        )
    print(f"Pipeline papers: {len(all_pipeline_dirs)} total, {len(pipeline_dirs)} matching")
    print(f"Min slots: {min_slots}  (papers with < {min_slots} non-empty HEVI slots will be skipped)")
    print()

    dataset: List[Dict[str, Any]] = []
    skipped_no_extraction = 0
    skipped_no_consensus = 0
    skipped_low_slots = 0
    skipped_no_valid_chains = 0  # chains dropped for missing scenario/issue/hazard/exposure

    for pipe_dir in pipeline_dirs:
        paper_id = pipe_dir.name

        # Look up extraction data (may be under group_*/)
        ext_path = _find_extract(EXTRACT_DIR, paper_id)
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

        # ── Slot filter: skip papers with too few non-empty ref_hevi slots ──
        ref_hevi_norm = normalize_hevi_six(ext_data.get("ref_hevi", {}))
        n_slots = _count_non_empty_slots(ref_hevi_norm)
        if n_slots < min_slots:
            skipped_low_slots += 1
            print(f"  {paper_id} — {n_slots} slots < {min_slots}, skipped")
            continue

        consensus = load_json(consensus_path)

        chains_raw = consensus.get("chains", [])
        chains = [c for c in (extract_chain(ch) for ch in chains_raw if isinstance(ch, dict)) if c is not None]
        if not chains:
            skipped_no_valid_chains += 1
            print(f"  {paper_id} — 0 valid chains (s/i/h/e required), skipped")
            continue

        record: Dict[str, Any] = {
            "paper_id": paper_id,
            "title": ext_data.get("title", ""),
            "abstract": ext_data.get("abstract", ""),
            "impact": ext_data.get("impact", ""),
            "ref_hevi": ref_hevi_norm,
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
    print(f"  Skipped (low slots): {skipped_low_slots}")
    print(f"  Skipped (no valid chains): {skipped_no_valid_chains}  (missing scenario/issue/hazard/exposure)")
    print(f"  Total chains:        {total_chains}")
    print(f"  Total ref_hevi items:{total_ref_items}")
    print(f"  Total wf HEVI items: {total_wf_items}")
    print(f"  Output:              {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build HEVI dataset JSON from pipeline outputs")
    parser.add_argument("--group", type=int, default=0,
                        help="Only include this group number (1-6, 0 = all)")
    parser.add_argument("--min-slots", type=int, default=3,
                        help="Minimum non-empty ref_hevi slots required (default 3, i.e. >2)")
    args = parser.parse_args()
    groups = [f"group_{args.group}"] if args.group else None
    build_dataset(groups=groups, min_slots=args.min_slots)
