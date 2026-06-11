# SS Agent 评议 CS

**Stage**: Stage 4 (共识)
**Location**: `agents.py → SSAgent.critique_cs`

```python
    def critique_cs(
        self,
        cs_proposal: Dict[str, Any],
        ss_evidence: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        system = (
            "You are the Social Science Agent in a symmetric bilateral consensus protocol. "
            "Your role in this phase is to critique the CS Agent's nexus_candidates from a social science standpoint. "
            "For each CS nexus_candidate, assess whether the exposure scenarios and risk issues are socially plausible. "
            "Point out: missing stakeholder groups, overlooked vulnerability dimensions, implausible social mechanisms, "
            "or impact pathways that social science evidence suggests would differ. "
            "Be specific and constructive — suggest concrete additions or revisions. "
            "If a candidate IS socially well-grounded, say so clearly. "
            "Return valid JSON only."
        )
        user = {
            "task": "Critique each CS nexus_candidate for social plausibility and completeness.",
            "cs_proposal": {
                "hazard": cs_proposal.get("hazard"),
                "nexus_candidates": cs_proposal.get("nexus_candidates", []),
            },
            "ss_evidence": _compact_evidence(ss_evidence),
            "output_schema": {
                "agent": "ss",
                "critiques": [
                    {
                        "nexus_index": 0,
                        "scenario": "scenario being critiqued",
                        "assessment": "supported|speculative|unsupported",
                        "social_critique": "specific social science reasoning — what is plausible or missing",
                        "suggested_change": "concrete revision suggestion, or null if fully supported",
                    }
                ],
                "overall": "one-sentence summary of SS-side social science assessment",
            },
        }
        return self.llm.json_call(system, json.dumps(user, ensure_ascii=False))

```
