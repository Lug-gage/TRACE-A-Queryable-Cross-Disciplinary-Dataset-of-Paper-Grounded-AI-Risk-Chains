# Dose-Response 合成

**Stage**: Stage 4
**Location**: `pipeline.py → _synthesize_dose_response`

```python
def _synthesize_dose_response(
    llm: RiskLLM,
    cs_proposal: Dict[str, Any],
    ss_response: Dict[str, Any],
    cs_evidence: List[Dict[str, Any]],
    ss_evidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    system = (
        "You are the Dose-Response Synthesizer in a bilateral consensus protocol. "
        "Your task is NOT to generate new content — splice the CS Agent's verified Hazard->Exposure segment "
        "with the SS Agent's verified Vulnerability->Impact segment at the agreed place-based Nexus (scenario, issue). "
        "Dose-Response = how hazard SCALE/INTENSITY translates into impact magnitude. "
        "It is the CAUSAL TRANSLATION through the coupled human-technical system: "
        "more/faster/wider deployment → more/worse harm. "
        "Acknowledge coupled system feedback where present (e.g., response in technical subsystem "
        "affects human subsystem, or vice versa). "
        "Each chain must be traceable to both CS and SS evidence. "
        "If the two segments cannot be coherently spliced for a given Nexus, note it in unsupported_or_missing. "
        "BREVITY: dose_response MUST be exactly ONE sentence (≤40 words). "
        "hazard, exposure, vulnerability, impact, KCN — quote the verified agent claims VERBATIM, do NOT rewrite or expand them. "
        "Return valid JSON only."
    )
    user = {
        "task": "Splice verified CS and SS segments at each Nexus into complete Dose-Response causal chains.",
        "cs_segment": {
            "hazard": cs_proposal.get("hazard"),
            "nexus_candidates": cs_proposal.get("nexus_candidates", []),
            "self_score": cs_proposal.get("self_score"),
        },
        "ss_segment": {
            "nexus_responses": ss_response.get("nexus_responses", []),
            "self_score": ss_response.get("self_score"),
        },
        "cs_evidence": [
            {"title": item.get("title"), "abstract": str(item.get("abstract") or item.get("doc_text") or "")[:600]}
            for item in cs_evidence[:5]
        ],
        "ss_evidence": [
            {"title": item.get("title"), "abstract": str(item.get("abstract") or item.get("doc_text") or "")[:600]}
            for item in ss_evidence[:5]
        ],
        "field_definitions": {
            "hazard": "Technical capability, model behavior, or technical risk source from CS Agent.",
            "exposure": "Who, what system, or what setting is exposed — from CS Agent.",
            "dose_response": "★ The causal translation: HOW the technical capability, through the exposure scenario, leads to social impact. This is the core contribution — the spliced product at the Nexus.",
            "vulnerability": "Groups, systems, or contexts more susceptible — from SS Agent.",
            "impact": "Social, ethical, economic, or governance consequences — from SS Agent.",
            "key_control_nodes": "Intervention points from SS Agent. Two categories: (a) technical controls such as audit, governance, mitigation, adversarial training, anomaly detection, rate limiting; (b) author actions such as code release, dataset publication, benchmark creation, red-teaming invitations, evaluation tools.",
        },
        "rules": [
            "Splice at each agreed Nexus (scenario, issue). One chain per Nexus.",
            "dose_response: EXACTLY one sentence (≤40 words). Format: 'X enables Y, which in scenario Z exposes W, leading to V.' Do NOT expand into paragraphs.",
            "hazard, exposure, vulnerability, impact, KCN: quote the verified agent claims VERBATIM. Do NOT rewrite, summarize, or expand them.",
            "If the consensus did not converge (scores below 0.8), set confidence to 'none' and note the gap in unsupported_or_missing.",
            "If dose_response cannot be constructed from available segments, leave it empty.",
            "Cite evidence by title and short point — no internal IDs.",
            "Do not invent content not present in the CS or SS segments.",
            "Do NOT weaken claims when splicing. Quote verbatim — if a claim contains hedging from a non-converged round, keep it but note non-convergence.",
        ],
        "output_schema": {
            "chains": [
                {
                    "scenario": "verbatim scenario from agreed Nexus",
                    "issue": "verbatim issue from agreed Nexus",
                    "hazard": ["verbatim hazard claim from CS Agent"],
                    "exposure": ["verbatim exposure claim from CS Agent"],
                    "dose_response": ["ONE sentence (≤40 words): X enables Y, which in scenario Z leads to V"],
                    "vulnerability": ["verbatim vulnerability claim from SS Agent"],
                    "impact": ["verbatim impact claim from SS Agent"],
                    "key_control_nodes": ["verbatim KCN from SS Agent"],
                    "evidence_trace": {
                        "cs_evidence": [{"title": "CS evidence title", "point": "brief grounding point"}],
                        "ss_evidence": [{"title": "SS evidence title", "point": "brief grounding point"}],
                    },
                    "confidence": "low|medium|high",
                }
            ],
            "unsupported_or_missing": ["string"],
        },
    }
    return llm.json_call(system, json.dumps(user, ensure_ascii=False))

```
