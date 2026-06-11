# HEVI 对比评测

**Stage**: Stage 5
**Location**: `hevi_compare.py → HEVIComparator.compare`

```python
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

```
