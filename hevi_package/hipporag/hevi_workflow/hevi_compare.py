import json
from typing import Any, Dict, List

from hipporag.hevi_workflow.agents import RiskLLM
from hipporag.hevi_workflow.pipeline import HEVI_KEYS


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_comparison(row: Dict[str, Any], reference: Dict[str, Any], workflow: Dict[str, Any]) -> Dict[str, Any]:
    item_matches = row.get("item_matches", {})
    if not isinstance(item_matches, dict):
        item_matches = {}
    ref_hevi = reference.get("hevi", {}) if isinstance(reference.get("hevi"), dict) else {}

    total_items = 0
    matched_items = 0
    normalized = {}

    for key in HEVI_KEYS:
        ref_items = as_list(ref_hevi.get(key))
        if not ref_items:
            normalized[key] = []
            continue

        # LLM judgments for this slot, one per reference item
        judgments = as_list(item_matches.get(key))
        slot_results = []
        for idx, ref_item in enumerate(ref_items):
            judgment = judgments[idx] if idx < len(judgments) and isinstance(judgments[idx], dict) else {}
            is_matched = bool(judgment.get("matched"))
            if is_matched:
                matched_items += 1
            total_items += 1
            slot_results.append({
                "reference_item": str(ref_item),
                "matched": is_matched,
                "matched_workflow_evidence": [
                    str(v) for v in as_list(judgment.get("matched_workflow_evidence")) if str(v).strip()
                ],
                "reason": str(judgment.get("reason") or ""),
            })
        normalized[key] = slot_results

    return {
        "title": reference.get("title") or "",
        "reference_impact": reference.get("impact", ""),
        "reference_item_count": total_items,
        "matched_item_count": matched_items,
        "item_recall": matched_items / total_items if total_items else None,
        "slot_matches": normalized,
        "notes": row.get("notes", []) if isinstance(row.get("notes"), list) else [],
    }


class HEVIComparator:
    def __init__(self, llm: RiskLLM) -> None:
        self.llm = llm

    def compare(self, workflow: Dict[str, Any], reference: Dict[str, Any]) -> Dict[str, Any]:
        system = (
            "You compare workflow-generated HEVI risk chains against an ICML-impact-derived HEVI reference. "
            "Judge each reference item independently — each item is a single risk claim within a HEVI slot. "
            "Judge semantic matches, not exact wording. "
            "Do not give credit for vague topical similarity; require the workflow chain to express the same risk element. "
            "Only slots with non-empty reference items are judged. "
            "Return valid JSON only."
        )
        user = {
            "task": "Compute item-level semantic recall of workflow HEVI against each reference HEVI item.",
            "reference": {
                "title": reference.get("title", ""),
                "impact": reference.get("impact", ""),
                "hevi": reference.get("hevi", {}),
            },
            "workflow": {
                "chains": workflow.get("chains", []),
            },
            "rules": [
                "For each HEVI slot with non-empty reference items, output one judgment per reference item.",
                "For each reference item: matched=true only when at least one workflow chain semantically covers that specific claim.",
                "Do not output internal IDs. Use matched_workflow_evidence to briefly name the matching chain content.",
                "SLOT-SPECIFIC MATCHING:",
                "  - hazard: match on the core technical capability/risk. Do not penalise the workflow for being more specific than the reference.",
                "  - exposure: match on WHO is exposed (the group/system name). The technical qualifier (e.g. 'linguistically calibrated') belongs to the hazard field, not exposure. Do NOT require it in the workflow exposure text.",
                "  - vulnerability: match on the enabling condition or gap that makes harm possible.",
                "  - impact: match on the negative social/ethical consequence.",
                "  - key_control_nodes: match on the intervention or mitigation action.",
            ],
            "output_schema": {
                "item_matches": {
                    key: [
                        {
                            "matched": True,
                            "matched_workflow_evidence": ["..."],
                            "reason": "brief reason",
                        }
                    ]
                    for key in HEVI_KEYS
                },
                "notes": ["string"],
            },
        }
        compared = self.llm.json_call(system, json.dumps(user, ensure_ascii=False))
        return normalize_comparison(compared, reference, workflow)
