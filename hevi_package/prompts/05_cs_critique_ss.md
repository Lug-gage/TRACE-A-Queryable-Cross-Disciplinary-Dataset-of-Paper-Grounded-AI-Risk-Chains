# CS Agent 评议 SS

**Stage**: Stage 4 (共识)
**Location**: `agents.py → CSAgent.critique_ss`

```python
    def critique_ss(
        self,
        ss_response: Dict[str, Any],
        cs_evidence: List[Dict[str, Any]],
        cs_proposal: Dict[str, Any],
    ) -> Dict[str, Any]:
        system = (
            "You are the CS Agent in a symmetric bilateral consensus protocol. "
            "Your role in this phase is to critique the SS Agent's nexus_responses from a purely technical standpoint. "
            "For each SS nexus_response, assess whether the claimed vulnerabilities and impacts are technically plausible "
            "given the CS evidence and your own CS proposal. "
            "Be specific and constructive — point out exactly what is technically wrong or unanchored, "
            "and suggest concrete revisions. "
            "If a claim IS technically supported, say so clearly and give the supporting evidence. "
            "Return valid JSON only."
        )
        user = {
            "task": "Critique each SS nexus_response for technical plausibility and evidence grounding.",
            "cs_proposal": {
                "hazard": cs_proposal.get("hazard"),
                "nexus_candidates": cs_proposal.get("nexus_candidates", []),
            },
            "ss_response": {
                "nexus_responses": ss_response.get("nexus_responses", []),
            },
            "cs_evidence": _compact_evidence(cs_evidence),
            "output_schema": {
                "agent": "cs",
                "critiques": [
                    {
                        "nexus_index": 0,
                        "scenario": "scenario being critiqued",
                        "assessment": "supported|speculative|unsupported",
                        "technical_critique": "specific technical reasoning — what is plausible or implausible",
                        "suggested_change": "concrete revision suggestion, or null if fully supported",
                    }
                ],
                "overall": "one-sentence summary of CS-side technical assessment",
            },
        }
        return self.llm.json_call(system, json.dumps(user, ensure_ascii=False))

```
