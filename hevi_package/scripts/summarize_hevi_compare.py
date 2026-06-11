import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from hipporag.hevi_workflow.pipeline import HEVI_KEYS


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    slot_totals = {
        key: {"reference_items": 0, "matched": 0, "recall": None}
        for key in HEVI_KEYS
    }
    total_ref = 0
    total_matched = 0
    papers = []

    for row in rows:
        ref_count = int(row.get("reference_item_count") or 0)
        matched_count = int(row.get("matched_item_count") or 0)
        total_ref += ref_count
        total_matched += matched_count
        papers.append(
            {
                "title": row.get("title", ""),
                "reference_item_count": ref_count,
                "matched_item_count": matched_count,
                "item_recall": row.get("item_recall"),
            }
        )

        slot_matches = row.get("slot_matches", {})
        if not isinstance(slot_matches, dict):
            slot_matches = {}
        for key in HEVI_KEYS:
            items = slot_matches.get(key, [])
            if not isinstance(items, list):
                items = []
            slot_ref = len(items)
            slot_matched = sum(1 for item in items if isinstance(item, dict) and item.get("matched"))
            slot_totals[key]["reference_items"] += slot_ref
            slot_totals[key]["matched"] += slot_matched

    for key, data in slot_totals.items():
        denom = data["reference_items"]
        data["recall"] = data["matched"] / denom if denom else None

    return {
        "paper_count": len(rows),
        "overall": {
            "reference_item_count": total_ref,
            "matched_item_count": total_matched,
            "item_recall": total_matched / total_ref if total_ref else None,
        },
        "by_slot": slot_totals,
        "papers": papers,
    }


def write_csv(path: Path, papers: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "title",
                "reference_item_count",
                "matched_item_count",
                "item_recall",
            ],
        )
        writer.writeheader()
        writer.writerows(papers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize HEVI comparison results.")
    parser.add_argument("--input-dir", default="outputs/hevi_workflow/hevi_compare")
    parser.add_argument("--output", default="outputs/hevi_workflow/hevi_compare_summary.json")
    parser.add_argument("--csv-output", default="outputs/hevi_workflow/hevi_compare_summary.csv")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    rows = []
    for path in sorted(input_dir.glob("*.json")):
        row = read_json(path)
        rows.append(row)

    summary = summarize(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(Path(args.csv_output), summary["papers"])

    print(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "paper_count": summary["paper_count"],
                "summary": str(output_path),
                "csv": args.csv_output,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
