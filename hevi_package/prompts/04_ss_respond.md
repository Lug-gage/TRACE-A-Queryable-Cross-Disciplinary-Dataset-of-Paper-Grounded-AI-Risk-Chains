# SS Agent 响应

**Stage**: Stage 3
**Location**: `agents.py → SSAgent.respond_bilateral`

```python
    def respond_bilateral(
        self,
        cs_proposal: Dict[str, Any],
        ss_evidence: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        system = (
            "You are the Social Science Agent in a symmetric bilateral consensus protocol. "
            "Respond to the CS Agent's Nexus candidates. You may accept, reject, or revise each Nexus. "
            "You are responsible only for Nexus -> Vulnerability -> Impact and Key Control Nodes. "
            "Do not produce Exposure; Exposure belongs to the CS Agent's Hazard -> Exposure segment. "
            "Vulnerability = system SENSITIVITY to the hazard (what conditions make harm more likely/severe). "
            "Operates at multiple scales: individual, institutional, infrastructural, societal. "
            "Impact = CONSEQUENCES when hazard meets exposure under vulnerable conditions. "
            "Consequences ripple through coupled human-technical systems. "
            "Key Control Nodes = COPING CAPACITY at three levels: "
            "(a) Individual/autonomous: adversarial training, input validation, user vigilance. "
            "(b) Institutional/organizational: audit, human-in-the-loop, red-teaming, monitoring, anomaly detection, rate limiting. "
            "(c) Policy/societal: regulation, standards, ethical review, public code release. "
            "Use retrieved SS evidence for social mechanisms and impacts. "
            "Your self_score must reflect: (a) how well your vulnerability claims are grounded in SS evidence, "
            "(b) how plausible your social mechanisms are, (c) how complete your impact coverage is. "
            "SCORING GUIDE: A SHORT, SPECIFIC, evidence-anchored claim scores HIGH (0.7–0.9). "
            "A vague, generic, or unsupported claim scores LOW (0.3–0.5). "
            "Length is not a virtue — one precise sentence that names the actual risk "
            "is BETTER than a paragraph of hedging. "
            "Make risk claims as ASSERTIONS. State what the risk IS ('X leads to Y'), "
            "not what it MIGHT BE ('X could potentially lead to Y'). "
            "Reserve uncertainty for the confidence and self_score fields, not the claim text. "
            "If the CS proposal or its context mentions code release, tool building, or evaluation resources, include them as KCN. "
            "BREVITY: Each vulnerability and impact entry MUST be one concise phrase (≤30 words), not a paragraph. "
            "Each key_control_node MUST be ≤10 words — name the control only, NO explanation. "
            "social_mechanism MUST be 2-3 sentences (≤100 words total). "
            "Do not create near-duplicate vulnerability/impact/KCN entries across nexus responses; consolidate if items overlap. "
            "KCN must name actionable technical or policy interventions. "
            "Do NOT include research proposals as KCN: "
            "✗ 'Targeted clinical studies to assess X', ✗ 'Human-subject studies of Y', ✗ 'Future research on Z'. "
            "✓ 'Human-in-the-loop review', ✓ 'Confidence threshold alerts', ✓ 'Adversarial training'. "
            "Return valid JSON only."
        )
        user = {
            "task": "Respond to CS-proposed Nexus candidates and add the SS-side segment.",
            "cs_proposal": cs_proposal,
            "ss_evidence": _compact_evidence(ss_evidence),
            "output_schema": {
                "agent": "ss",
                "round": 1,
                "self_score": "0.0-1.0 confidence in the SS-side response",
                "nexus_responses": [
                    {
                        "nexus_id": "n1",
                        "decision": "accept|revise|reject",
                        "scenario": "accepted or revised scenario (≤80 chars label)",
                        "issue": "accepted or revised issue (≤100 chars)",
                        "revision_reason": "one sentence reason if revised, empty if accepted",
                        "vulnerability": ["ONE concise phrase each (≤30 words) — a risk condition, not an essay"],
                        "impact": ["ONE concise phrase each (≤30 words) — a negative consequence, not an essay"],
                        "key_control_nodes": ["≤10 words each — name the control only, no explanation"],
                        "social_mechanism": "2-3 sentences (≤100 words) describing the causal social mechanism",
                        "confidence": "low|medium|high",
                        "ss_evidence_ids": ["ss_e1"],
                    }
                ],
            },
        }
        return self.llm.json_call(system, json.dumps(user, ensure_ascii=False))

```
