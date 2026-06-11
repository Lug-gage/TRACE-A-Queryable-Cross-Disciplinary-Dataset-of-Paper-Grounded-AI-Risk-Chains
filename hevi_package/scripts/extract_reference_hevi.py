"""
Stage 1 only: extract query + ref_hevi from ICML papers.
Two-step extraction with architecture-level information isolation —
  2 LLM calls per paper: (A) title+abstract → CS query, (B) impact → HEVI extraction.
No retrieval — fast filtering before the full pipeline.

Output: one JSON file per paper.
Filter by --min-slots to keep only papers with rich impact statements.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipporag.hevi_workflow.agents import RiskLLM
from hipporag.hevi_workflow.pipeline import HEVI_KEYS, build_workflow_input, strip_internal_ids
from hipporag.hevi_workflow.utils import load_icml_inputs, load_project_api_key, write_json


def count_nonempty_slots(hevi: Dict[str, List[str]]) -> int:
    return sum(1 for key in HEVI_KEYS if hevi.get(key))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract reference HEVI from ICML papers (Stage 1 only).")
    parser.add_argument("--input", default="data/icml_corpus_with_len.csv")
    parser.add_argument("--limit", type=int, default=0, help="Max CSV rows to scan (0 = scan all)")
    parser.add_argument("--min-impact-chars", type=int, default=500,
                        help="Skip papers with impact statement shorter than this (default: 500)")
    parser.add_argument("--min-slots", type=int, default=3,
                        help="Only output papers with at least this many non-empty HEVI slots (default: 0 = all)")
    parser.add_argument("--max-kept", type=int, default=0,
                        help="Stop after keeping this many papers (0 = no limit)")
    parser.add_argument("--output-dir", default="outputs/hevi_workflow/hevi_icml_{model}")
    parser.add_argument("--llm-name", default="deepseek-v4-pro")
    parser.add_argument("--llm-base-url", default="https://www.highland-api.top/v1")
    parser.add_argument("--max-new-tokens", type=int, default=384000)
    parser.add_argument("--llm-timeout", type=int, default=120)
    parser.add_argument("--llm-retries", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.max_new_tokens == 0:
        if "deepseek" in args.llm_name.lower():
            args.max_new_tokens = 384000
        elif "gpt" in args.llm_name.lower():
            args.max_new_tokens = 128000
        else:
            args.max_new_tokens = 128000

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    for logger_name in ("httpx", "httpcore", "openai", "openai._base_client"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    load_project_api_key()

    model_tag = args.llm_name.replace("/", "_").replace(" ", "-")
    output_dir = Path(str(args.output_dir).replace("{model}", model_tag))
    output_dir.mkdir(parents=True, exist_ok=True)

    llm = RiskLLM(
        save_dir=str(output_dir.parent),
        llm_name=args.llm_name,
        llm_base_url=args.llm_base_url,
        max_new_tokens=args.max_new_tokens,
        request_timeout=args.llm_timeout,
        max_retries=args.llm_retries,
    )

    papers = load_icml_inputs(args.input, args.limit)
    results: List[Dict[str, Any]] = []
    kept_ids: List[Dict[str, Any]] = []
    slot_dist: Dict[int, int] = {}
    kept = 0

    already = set(f.stem for f in output_dir.glob("icml_*.json"))
    skipped_short = 0
    skipped_done = 0
    print(f"Total papers to scan: {len(papers)}, min impact chars: {args.min_impact_chars}, min slots: {args.min_slots}", flush=True)
    for idx, paper in enumerate(papers, start=1):
        paper_id = paper["paper_id"]

        if paper_id in already:
            skipped_done += 1
            continue

        impact_len = int(paper.get("impact_chars", 0) or len(paper.get("impact", "").strip()))

        if impact_len < args.min_impact_chars:
            skipped_short += 1
            slot_dist[0] = slot_dist.get(0, 0) + 1
            continue

        title_short = paper.get("title", "")[:60]
        print(f"  [{idx}/{len(papers)}] {paper_id} extracting...", flush=True)
        try:
            ref_data = build_workflow_input(paper, llm)
        except Exception as exc:
            print(f"  [{idx}/{len(papers)}] {paper_id} FAILED: {exc}", flush=True)
            continue
        ref_data["paper_id"] = paper_id
        ref_data["impact_raw"] = paper.get("impact", "")
        n_slots = count_nonempty_slots(ref_data.get("ref_hevi", {}))
        ref_data["nonempty_slots"] = n_slots

        slot_dist[n_slots] = slot_dist.get(n_slots, 0) + 1

        if n_slots >= args.min_slots:
            results.append(strip_internal_ids(ref_data))
            paper_out = {
                "title": ref_data["title"],
                "abstract": ref_data["abstract"],
                "query": ref_data.get("query", ""),
                "query_terms": ref_data.get("query_terms", []),
                "impact": ref_data.get("impact_raw", ""),
                "hevi_node": n_slots,
                "ref_hevi": ref_data.get("ref_hevi", {}),
            }
            write_json(output_dir / f"{paper_id}.json", paper_out)
            kept_ids.append({
                "paper_id": paper_id,
                "title": paper["title"],
                "hevi_node": n_slots,
            })
            kept += 1
            print(f"  [{idx}/{len(papers)}] {paper_id} slots={n_slots} ✓ KEPT ({kept} total)", flush=True)
        else:
            print(f"  [{idx}/{len(papers)}] {paper_id} slots={n_slots} < {args.min_slots}, skipped", flush=True)

        if args.max_kept > 0 and kept >= args.max_kept:
            print(f"Reached --max-kept={args.max_kept}, stopping.", flush=True)
            break

    # Summary
    total = len(papers)
    summary = {
        "total_papers": total,
        "min_impact_chars": args.min_impact_chars,
        "skipped_short_impact": skipped_short,
        "skipped_already_done": skipped_done,
        "llm_calls": 2 * (total - skipped_short - skipped_done),  # 2 calls per paper: Step A (query) + Step B (HEVI)
        "min_slots_filter": args.min_slots,
        "output_papers": kept,
        "slot_distribution": {str(k): v for k, v in sorted(slot_dist.items())},
        "empty_impact": slot_dist.get(0, 0),
    }
    write_json(output_dir / "summary.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Wrote {kept} papers to {output_dir}/", flush=True)

    if kept_ids:
        print("\nKept papers:")
        for r in kept_ids:
            print(f"  {r['paper_id']}  slots={r['hevi_node']}  {r['title']}")


if __name__ == "__main__":
    main()
