# SS Agent 修订响应

**Stage**: Stage 4 (共识)
**Location**: `agents.py → SSAgent.revise_response`

```python
    def revise_response(
        self,
        ss_response: Dict[str, Any],
        cs_critique: Dict[str, Any],
        ss_evidence: List[Dict[str, Any]],
        cs_proposal: Dict[str, Any],
    ) -> Dict[str, Any]:
        system = (
            "You are the Social Science Agent in a symmetric bilateral consensus protocol. "
            "Your role in this phase is to revise your SS response based on the CS Agent's critique. "
            "For each critique point: either accept the technical correction and revise your response, "
            "or explain why you maintain your original position (only if you have strong social science evidence). "
            "Take technical constraints seriously — if the CS Agent says a vulnerability is technically implausible, "
            "either drop it or rephrase it to be technically accurate. "
            "Update your self_score to reflect the revised response quality. "
            "key_control_nodes = COPING CAPACITY at three levels (Turner et al. 2003): "
            "(a) Individual/autonomous: adversarial training, input validation, user vigilance. "
            "(b) Institutional/organizational: audit, human-in-the-loop, red-teaming, monitoring, anomaly detection. "
            "(c) Policy/societal: regulation, standards, ethical review, public code release. "
            "If the CS proposal or critique mentions code release, tool building, or evaluation resources, include them as KCN. "
            "REVISION PRINCIPLE: prefer precision over length. If a vulnerability or impact is already adequately expressed, "
            "refine it rather than adding new text. Merge overlapping vulnerability/impact items across nexus responses "
            "into single concise entries. Consolidate redundant KCNs — if the same control appears in multiple nexus responses, "
            "keep it only once. "
            "Making claims MORE SPECIFIC is an improvement. Adding hedging words "
            "(may, might, hypothesized, could potentially, requires further study) is a DOWNGRADE. "
            "A confident assertion backed by evidence is the goal. "
            "If uncertain about a point, reflect that in self_score, not by weakening the claim language. "
            "KCN must name actionable technical or policy interventions. "
            "Do NOT include research proposals as KCN "
            "(✗ 'Targeted studies to assess X', ✗ 'Future research on Y'). "
            "Return valid JSON only."
        )
        user = {
            "task": "Revise the SS response absorbing the CS Agent's critique.",
            "original_response": {
                "nexus_responses": ss_response.get("nexus_responses", []),
                "self_score": ss_response.get("self_score"),
            },
            "cs_critique": cs_critique.get("critiques", []),
            "cs_overall": cs_critique.get("overall", ""),
            "cs_proposal": {
                "hazard": cs_proposal.get("hazard"),
                "nexus_candidates": cs_proposal.get("nexus_candidates", []),
            },
            "ss_evidence": _compact_evidence(ss_evidence),
            "output_schema": {
                "agent": "ss",
                "self_score": "0.0-1.0 updated confidence after revision",
                "revision_notes": "one sentence summarizing what was changed and why",
                "nexus_responses": [
                    {
                        "nexus_id": "n1",
                        "decision": "accept|revise|reject",
                        "scenario": "accepted or revised scenario",
                        "issue": "accepted or revised issue",
                        "revision_reason": "reason for revision, or empty",
                        "vulnerability": ["string"],
                        "impact": ["string"],
                        "key_control_nodes": ["string"],
                        "social_mechanism": "string",
                        "confidence": "low|medium|high",
                        "ss_evidence_ids": ["ss_e1"],
                    }
                ],
            },
        }
        return self.llm.json_call(system, json.dumps(user, ensure_ascii=False))

```
