# CS Agent 修订提案

**Stage**: Stage 4 (共识)
**Location**: `agents.py → CSAgent.revise_proposal`

```python
    def revise_proposal(
        self,
        cs_proposal: Dict[str, Any],
        ss_critique: Dict[str, Any],
        cs_evidence: List[Dict[str, Any]],
        cs_query: str,
    ) -> Dict[str, Any]:
        system = (
            "You are the CS Agent in a symmetric bilateral consensus protocol. "
            "Your role in this phase is to revise your CS proposal based on the SS Agent's critique. "
            "For each critique point: either accept the suggestion and revise your proposal accordingly, "
            "or explain why you maintain your original position (only if you have strong technical evidence). "
            "Do NOT simply dismiss valid social science concerns — if the SS Agent points out a missing stakeholder "
            "or an implausible exposure pathway, adjust your nexus_candidates to address it. "
            "Update your self_score to reflect the revised proposal quality. "
            "REVISION PRINCIPLE: prefer precision over length. If a point is already adequately expressed, "
            "refine it rather than adding new text. When addressing a critique about a missing stakeholder, "
            "add that stakeholder's NAME to the existing exposure item — do NOT add vulnerability conditions "
            "or impact descriptions (those belong to the SS Agent). "
            "Do NOT create a new paragraph or duplicate the scenario. "
            "Consolidate near-duplicate items into a single concise entry. "
            "Making claims MORE SPECIFIC is an improvement. Adding hedging words "
            "(may, might, hypothesized, could potentially, requires further study) is a DOWNGRADE. "
            "A confident assertion backed by evidence is the goal. "
            "If uncertain about a point, reflect that in self_score, not by weakening the claim language. "
            "Return valid JSON only."
        )
        user = {
            "task": "Revise the CS proposal absorbing the SS Agent's critique.",
            "original_proposal": {
                "hazard": cs_proposal.get("hazard"),
                "nexus_candidates": cs_proposal.get("nexus_candidates", []),
                "self_score": cs_proposal.get("self_score"),
            },
            "ss_critique": ss_critique.get("critiques", []),
            "ss_overall": ss_critique.get("overall", ""),
            "cs_evidence": _compact_evidence(cs_evidence),
            "cs_query": cs_query,
            "output_schema": {
                "agent": "cs",
                "self_score": "0.0-1.0 updated confidence after revision",
                "revision_notes": "one sentence summarizing what was changed and why",
                "hazard": [
                    {
                        "hazard": "technical hazard description",
                        "confidence": "low|medium|high",
                    }
                ],
                "nexus_candidates": [
                    {
                        "scenario": "deployment or encounter setting",
                        "issue": "risk issue created by the technical mechanism",
                        "exposure": "who or what could be exposed",
                        "confidence": "low|medium|high",
                    }
                ],
            },
        }
        return self.llm.json_call(system, json.dumps(user, ensure_ascii=False))

```
