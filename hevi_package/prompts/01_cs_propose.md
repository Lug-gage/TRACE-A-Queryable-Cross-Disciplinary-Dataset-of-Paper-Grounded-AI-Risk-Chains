# CS Agent 提案

**Stage**: Stage 2
**Location**: `agents.py → CSAgent.propose_bilateral`

```python
    def propose_bilateral(
        self,
        paper_id: str,
        cs_query: str,
        cs_evidence: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        system = (
            "You are the CS Agent in a symmetric bilateral consensus protocol. "
            "You are the only agent allowed to propose the technical Hazard -> Exposure segment. "
            "Use only the CS retrieval query and retrieved CS evidence. "
            "Do not use the target title, abstract, or impact statement. "
            "Hazard = perturbation/stress introduced or amplified by the technical capability "
            "(can originate INSIDE the system — new capability — or OUTSIDE — amplifying existing risk). "
            "Exposure = who/what system elements face the hazard. EXPOSURE ≠ SENSITIVITY. "
            "Propose PLACE-BASED Nexus candidates as (scenario, issue) — vulnerability varies by deployment setting. "
            "Keep scenarios anchored in the retrieval query or CS evidence. "
            "Do not frame a Nexus as malicious/adversarial/attacker-driven unless the query or CS evidence explicitly "
            "mentions attack, adversary, malicious use, misuse, security, poisoning, or evasion. "
            "If no concrete application domain is stated, use a neutral generic scenario such as deployment of the retrieved capability "
            "in the stated task or system type. "
            "Your self_score must reflect: (a) how well your hazard is grounded in CS evidence, "
            "(b) how well your exposure scenarios are anchored in the query/evidence, "
            "(c) how complete your coverage of the technical risk surface is. "
            "SCORING GUIDE: A SHORT, SPECIFIC, evidence-anchored claim scores HIGH (0.7–0.9). "
            "A vague, generic, or unsupported claim scores LOW (0.3–0.5). "
            "Length is not a virtue — one precise sentence that names the actual risk "
            "is BETTER than a paragraph of hedging. "
            "Make risk claims as ASSERTIONS. State what the risk IS ('X enables Y'), "
            "not what it MIGHT BE ('X could potentially enable Y'). "
            "Reserve uncertainty for the confidence and self_score fields, not the claim text. "
            "BREVITY: hazard MUST be one concise phrase (≤30 words), not a paragraph. "
            "State the hazard at the paper's core technical capability level. "
            "Do NOT narrow it to a single scenario or application domain — hazard is about the capability itself, "
            "not how it manifests in one specific setting. "
            "✓ 'Calibrated confidence estimates that may differ from true likelihoods'. "
            "✗ 'Domain shifts degrade calibration in biomedicine' (too scenario-specific). "
            "Each scenario MUST be ≤80 characters — a label, not a description. "
            "Each issue MUST be ≤100 characters. "
            "Each exposure: name the exposed group + brief context of exposure, ≤15 words. "
            "Format: 'who' + 'doing what / in what setting'. "
            "No vulnerability conditions (that is SS Agent's job). No impact description (SS Agent's job). "
            "✓ 'Patients and clinicians relying on AI-assisted diagnosis'. "
            "✓ 'Researchers using LLM-powered scientific QA tools'. "
            "✗ 'Patients who face compounded harm due to healthcare disparities' (impact leaked into exposure). "
            "✗ 'Scientific researchers' (too bare — missing context of use). "
            "Keep nexus_candidates to strictly distinct scenarios; do not create near-duplicate entries. "
            "Return valid JSON only."
        )
        user = {
            "task": "Propose the CS-side risk segment and Nexus candidates for later SS Agent response.",
            "paper_id": paper_id,
            "cs_query": cs_query,
            "cs_evidence": _compact_evidence(cs_evidence),
            "output_schema": {
                "agent": "cs",
                "round": 1,
                "self_score": "0.0-1.0 confidence in the CS-side proposal",
                "hazard": [
                    {
                        "hazard": "ONE concise phrase (≤30 words): the technical hazard introduced or sharpened",
                        "confidence": "low|medium|high",
                    }
                ],
                "nexus_candidates": [
                    {
                        "scenario": "≤80 chars: deployment or encounter setting label, anchored in cs_query or cs_evidence",
                        "issue": "≤100 chars: the risk issue created by the technical mechanism",
                        "exposure": "≤15 words: who + brief context of use (e.g. 'Researchers using LLM QA tools'). No vulnerability conditions or impact descriptions",
                        "confidence": "low|medium|high",
                    }
                ],
            },
        }
        return self.llm.json_call(system, json.dumps(user, ensure_ascii=False))

```
