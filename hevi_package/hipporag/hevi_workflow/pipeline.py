import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from hipporag.hevi_workflow.agents import CSAgent, RiskLLM, SSAgent
from hipporag.hevi_workflow.hit_report import (
    compact_cs_output,
    compact_hit,
    compact_ss_output,
    judge_hits_with_llm,
)

HEVI_KEYS = [
    "hazard",
    "exposure",
    "dose_response",
    "vulnerability",
    "impact",
    "key_control_nodes",
]

INTERNAL_ID_KEYS = {
    "paper_id",
    "evidence_id",
    "nexus_id",
    "source_nexus_id",
    "cs_evidence_ids",
    "ss_evidence_ids",
    "evidence_ids",
    "included_nexus_ids",
    "unanchored_nexus_ids",
    "speculative_nexus_ids",
}
INTERNAL_ID_RE = re.compile(r"\s*\((?:cs|ss)_e\d+(?:\s*,\s*(?:cs|ss)_e\d+)*\)|\b(?:cs|ss)_e\d+\b")


def clean_text(value: Any) -> str:
    return " ".join(INTERNAL_ID_RE.sub("", str(value or "")).split())


def strip_internal_ids(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if key in INTERNAL_ID_KEYS or key.startswith("_"):
                continue
            cleaned[key] = strip_internal_ids(item)
        return cleaned
    if isinstance(value, list):
        return [strip_internal_ids(item) for item in value]
    if isinstance(value, str):
        return clean_text(value)
    return value


def normalize_hevi(value: Any) -> Dict[str, List[str]]:
    raw = value if isinstance(value, dict) else {}
    return {
        key: [clean_text(item) for item in raw.get(key, []) if clean_text(item)]
        if isinstance(raw.get(key, []), list)
        else []
        for key in HEVI_KEYS
    }


def build_workflow_input(paper: Dict[str, str], llm: RiskLLM) -> Dict[str, Any]:
    """Stage 1: Two-step extraction with architecture-level information isolation.

    Step A: Build CS retrieval query from title + abstract ONLY (blind to impact).
    Step B: Extract HEVI from impact statement (with title and abstract for disambiguation).
    """
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    impact = paper.get("impact", "")

    # ============================================================
    # Step A: CS Query — BLIND to impact statement
    # ============================================================
    query_system = (
        "You are a CS literature retrieval query builder. "
        "Given ONLY a paper's title and abstract, build a retrieval query to find "
        "technically related CS papers in a research corpus. "
        "You do NOT have access to the impact statement — work purely from technical content. "
        "Return valid JSON only."
    )
    query_user = {
        "task": "Build a CS retrieval query from title and abstract only.",
        "paper": {"title": title, "abstract": abstract},
        "query_rules": [
            "CRITICAL: 'query' MUST be a JSON array of strings, e.g. [\"term one\", \"term two\"]. Do NOT output a flat space-separated string.",
            "Use exact method names, task names, mechanisms, model architectures, datasets, and problem settings from title/abstract.",
            "Keep multi-word concepts as single array elements (e.g. 'Llama 2 7B', not 'Llama', '2', '7B').",
            "Include 2-3 SYNONYMS or alternative phrasings for the core technical concept (e.g. 'adversarial attack' AND 'adversarial example' AND 'evasion attack').",
            "Include 1-2 BROADER technical paradigm terms (e.g. for a specific attack method, also include 'AI security' or 'model robustness') to improve retrieval recall.",
            "Do NOT add social impact, fairness, privacy, governance, or safety terms unless EXPLICITLY present in title/abstract.",
            "Do NOT include paper title filler words like 'Revisiting', 'Towards', 'A Novel Approach', 'Rethinking'.",
            "Prefer specific named entities (method names, model names, dataset names) over broad category labels.",
            "Do not include terms for approaches or methods the paper explicitly argues against.",
            "Use 12-20 concise terms or phrases.",
        ],
        "output_schema": {
            "query": ["array of strings — each element is a term or multi-word phrase"],
            "rationale": "one sentence explaining the query construction strategy",
        },
    }
    query_result = llm.json_call(query_system, json.dumps(query_user, ensure_ascii=False))
    raw_query = query_result.get("query")
    if not isinstance(raw_query, list):
        raise ValueError(f"CS query must be a list of strings, got {type(raw_query).__name__}: {str(raw_query)[:200]}")
    query_terms = [clean_text(t) for t in raw_query if clean_text(t)]
    if not query_terms:
        raise ValueError("LLM returned an empty CS retrieval query.")
    query = " ".join(query_terms)

    # ============================================================
    # Step B: HEVI Extraction — from impact statement only
    # ============================================================
    hevi_system = (
        "You are an HEVI risk extraction specialist. "
        "Extract acknowledged risks from a paper's impact statement into the HEVI schema. "
        "HEVI is a RISK assessment framework — identify risks the author ACKNOWLEDGES, "
        "NOT benefits, contributions, or problems the paper solves. "
        "The title and abstract provide technical context ONLY for disambiguation "
        "(e.g., resolving what 'our algorithm' or 'the proposed method' concretely refers to). "
        "Do NOT use title/abstract to invent risk claims the impact does not mention. "
        "If the impact statement contains NO acknowledged risks — only benefits, contributions, or claims of "
        "'no negative impact' — leave ALL HEVI fields empty. "
        "CONSERVATIVE extraction (fewer, higher-quality items) is ALWAYS preferred over "
        "complete extraction with hallucinated items. "
        "Return valid JSON only."
    )
    hevi_user = {
        "task": "Extract reference HEVI schema from the impact statement. Use the ANCHORING CHECKLIST to filter every candidate item before extraction.",
        "paper": {"title": title, "abstract": abstract, "impact": impact},
        "anchoring_checklist": {
            "purpose": "EXECUTE THIS CHECKLIST BEFORE EXTRACTING ANY ITEM. Items that fail are DISCARDED.",
            "steps": [
                "STEP 1: Identify the paper's SPECIFIC technical contribution in one sentence (from the title and abstract). For survey/position papers without a novel method, identify the paper's core analytical framework or organizing principle.",
                "STEP 2: For each candidate risk item in the impact statement, perform the SUBSTITUTION TEST:",
                "  a. Write the risk claim with the specific method/technique name from this paper.",
                "  b. Replace the method name with '[another method in the same AI subfield]'.",
                "  c. Replace the method name with '[a method from a DIFFERENT AI subfield]'.",
                "  d. If the claim still reads as valid after BOTH substitutions → REJECT as generic template. Do NOT extract.",
                "  e. For survey/position papers: replace the paper's analytical framing with a different framing from the same domain. If the risk applies equally → REJECT.",
                "STEP 3: Only items causally linked to this paper's specific technical contribution (or analytical framework) may be extracted.",
            ],
            "pass_example": {
                "method": "Charmer query-based character-level adversarial attack",
                "claim": "Charmer adversarial attack enabling malicious evasion of NLP systems",
                "sub_1": "[PGD gradient-based attack] enabling malicious evasion → still plausible (both are attacks — partial overlap in subfield is expected)",
                "sub_2": "[Contrastive learning for sentence embeddings] enabling malicious evasion → NONSENSICAL (contrastive learning doesn't enable evasion)",
                "verdict": "PASS — cross-subfield test is decisive: the risk is causally tied to adversarial attack capability."
            },
            "fail_example": {
                "method": "CodeIt prioritized hindsight replay for program synthesis",
                "claim": "Language models exacerbating biases present in small training datasets",
                "sub_1": "[Fine-tuned LMs] exacerbating biases in small training datasets → reads the same",
                "sub_2": "[Prompt-engineered LMs] exacerbating biases in small training datasets → reads the same",
                "verdict": "FAIL — the risk (bias amplification in small datasets) is a property of ANY LM training regime, not causally linked to CodeIt's specific mechanism. This is generic template. DO NOT extract."
            }
        },
        "differential_diagnosis": {
            "purpose": "When a risk claim is ambiguous between two adjacent slots, use this matrix to decide.",
            "hazard_vs_vulnerability": {
                "rule": "hazard = what the method MAKES possible (capability). vulnerability = what ALLOWS harm (gap/condition).",
                "test": "Does this describe something the method actively DOES, or something that is MISSING? If MISSING → vulnerability, not hazard.",
                "hazard_done_wrong": "'Algorithm may produce biased predictions' → this describes a limitation, not a capability. Correct hazard must name the SPECIFIC mechanism enabling harm.",
                "vulnerability_done_wrong": "'The method lacks robustness testing' → describes a missing safeguard → belongs in vulnerability, NOT hazard."
            },
            "exposure_vs_impact": {
                "rule": "exposure = WHO + in what CONTEXT (situation). impact = WHAT negative OUTCOME results.",
                "test": "Does the phrase name a person/group/setting → exposure. Does it describe a harm/outcome → impact.",
                "exposure_done_wrong": "'Patients suffer misdiagnosis' → 'suffer misdiagnosis' is impact leaking into exposure. Correct: exposure='Patients receiving AI-assisted diagnosis', impact='Misdiagnosis leading to inappropriate treatment'.",
                "impact_done_wrong": "'Healthcare and finance sectors' → these are domains/settings (exposure territory), not actual consequences."
            },
            "kcn_vs_contribution": {
                "rule": "KCN = action that BLOCKS/REDUCES the risk. Contribution = what the paper BUILDS as its core offering.",
                "test": "Would this action exist even if the risk didn't? If YES → contribution, not KCN. KCN must be causally linked to mitigating the specific identified risk.",
                "kcn_done_wrong": "'Benchmark for explanation evaluation' → contribution. 'Future research on safety measures' or 'Targeted clinical studies to assess X' → research proposals, not KCN."
            }
        },
        "hevi_rules": [
            "CRITICAL: Only extract claims where the author acknowledges potential HARM, NEGATIVE consequence, or DANGER. If the impact statement only describes benefits, improvements, or contributions, leave ALL fields EMPTY.",
            "The title and abstract provide technical context ONLY for disambiguation. Do NOT use them to add risk claims the impact does not mention.",
            "Every extracted item MUST pass the ANCHORING CHECKLIST above. Generic template risks are WORSE than empty fields — they would poison the downstream evaluation.",
            "Every HEVI item must be a concise phrase grounded in the impact statement. Prefer EMPTY over hallucination.",
            "A single fact from the impact must appear in exactly ONE slot. Never duplicate across slots.",
            "BREVITY: hazard ≤30 words, exposure ≤15 words (who + brief context), vulnerability ≤30 words, impact ≤30 words, KCN ≤10 words each.",
        ],
        "field_definitions": {
            "hazard": "A PERTURBATION or STRESS introduced or amplified by the paper's method. Can originate INSIDE the system (new technical capability) or OUTSIDE (method amplifies existing risk). ≤30 words. State at capability level — not narrowed to a single scenario. NOT the problem solved. NOT a limitation. Example: 'Character-level adversarial attack enabling evasion of NLP systems' IS a hazard. 'Hallucination in transformers' is a pre-existing problem, NOT a hazard introduced by this paper.",
            "exposure": "WHO or WHAT system elements face the hazard. Format: 'who' + 'doing what / in what setting'. ≤15 words. EXPOSURE ≠ SENSITIVITY: describe who is in harm's way, NOT how easily they're affected. No vulnerability/impact leakage. If author doesn't specify WHO, leave EMPTY rather than writing 'society at large'. Example: 'Clinicians relying on AI-assisted diagnosis' IS exposure. 'Patients who face compounded harm' is NOT (impact leaked in). 'Healthcare' is NOT (too bare).",
            "dose_response": "How SCALE/FREQUENCY/INTENSITY of the hazard translates into impact MAGNITUDE through the coupled human-technical system. Almost never in impact statements — leave EMPTY unless author explicitly discusses thresholds or scaling relationships.",
            "vulnerability": "System SENSITIVITY to the hazard — conditions that make harm MORE LIKELY or MORE SEVERE. Operates at multiple scales: individual, institutional, infrastructural, societal. ≤30 words. NOT the gap the paper fills. NOT generic 'lack of regulation/oversight' unless author specifically mentions it. Example: 'Lack of adversarial robustness in deployed NLP models' IS a vulnerability. 'Lack of systematic debugging tools prior to this work' is NOT (research motivation).",
            "impact": "NEGATIVE CONSEQUENCES when hazard meets exposure under vulnerable conditions. Can ripple through coupled systems (e.g., biased decisions → eroded trust → reduced adoption). ≤30 words. NOT a benefit. NOT a generic category without specifics. Example: 'Malicious evasion of content filtering harming platform users' IS an impact. 'Reducing hallucination' is NOT (it's a benefit).",
            "key_control_nodes": "COPING CAPACITY at three levels. ≤10 words each, name only. (a) Individual/autonomous: adversarial training, input validation, user vigilance. (b) Institutional/organizational: audit, human-in-the-loop, red-teaming, monitoring. (c) Policy/societal: regulation, standards, ethical review, public code release. NOT the paper's method itself. NOT research proposals.",
        },
        "examples": [
            {
                "scenario": "Paper with genuine, well-specified risk acknowledgment",
                "impact": "Our Charmer attack algorithm could enable malicious evasion of NLP systems. We release code to help defenders assess robustness.",
                "correct_extraction": {
                    "hazard": ["Charmer query-based character-level adversarial attack enabling malicious evasion of NLP systems"],
                    "exposure": ["Users relying on automated sentiment analysis outputs", "Platforms using automated content moderation"],
                    "dose_response": [],
                    "vulnerability": ["Lack of adversarial robustness in deployed NLP models"],
                    "impact": ["Malicious evasion of content filtering exposing users to harmful content"],
                    "key_control_nodes": ["Public code release for defense assessment and hardening"]
                },
                "explanation": "All items pass the substitution test (Charmer→contrastive learning fails the cross-subfield check). Slots correctly assigned per differential diagnosis. Exposure is specific ('users relying on...', 'platforms using...'), not generic."
            },
            {
                "scenario": "Paper with ONLY benefits — NO risk acknowledged at all",
                "impact": "Our XAI method reduces hallucination in transformers, lowering costs and enabling adoption in healthcare. We release a benchmark for explanation evaluation.",
                "correct_extraction": {
                    "hazard": [], "exposure": [], "dose_response": [],
                    "vulnerability": [], "impact": [], "key_control_nodes": []
                },
                "explanation": "ALL empty. Impact describes benefits (reducing hallucination, lowering costs) and contributions (benchmark). 'Hallucination in transformers' is the problem being solved, not a hazard introduced. 'Benchmark release' is a contribution, not risk mitigation."
            },
            {
                "scenario": "Paper with real but VAGUE risk acknowledgment — partial extraction only. This is the HARDEST and most common case.",
                "impact": "While our method improves protein design efficiency, it could potentially be misused to design harmful biological agents. We encourage responsible use and have withheld certain implementation details.",
                "correct_extraction": {
                    "hazard": ["Diffusion-based protein design method enabling accelerated generation of protein structures"],
                    "exposure": [],
                    "dose_response": [],
                    "vulnerability": [],
                    "impact": ["Potential misuse to design harmful biological agents"],
                    "key_control_nodes": ["Withholding of implementation details to reduce misuse risk"]
                },
                "explanation": [
                    "exposure EMPTY: author says 'could be misused' but does NOT specify WHO or in WHAT context. Inventing 'Bad actors in biotech labs' would be hallucination.",
                    "vulnerability EMPTY: author does NOT identify any specific condition/gap making misuse more likely. 'Lack of regulation' would be our inference, not the author's claim.",
                    "hazard EXTRACTED: the method (diffusion-based protein design) IS causally linked to the risk (designing harmful agents). Substitution test: replace with 'graph neural network for molecular property prediction' → the risk changes direction. This PASSES.",
                    "Only 3 non-empty slots — CORRECT behavior. The author's risk acknowledgment is thin. The quality audit will evaluate whether the paper proceeds to the pipeline."
                ]
            },
            {
                "scenario": "Paper with risks that FAIL the substitution test — this is WHAT NOT TO DO",
                "impact": "Our weak-to-strong alignment method could be misused to create AI systems that bypass safety measures, potentially leading to job displacement and privacy erosion.",
                "wrong_extraction": {
                    "hazard": ["Weak-to-strong alignment enabling bypass of AI safety measures"],
                    "exposure": ["Society at large"],
                    "vulnerability": ["Lack of sufficient oversight and regulatory frameworks"],
                    "impact": ["Job loss across industries", "Privacy erosion through unauthorized data access", "Discrimination against marginalized groups"]
                },
                "why_every_item_is_wrong": [
                    "hazard FAILS substitution test: replace 'Weak-to-strong alignment' → 'RLHF' → 'RLHF enabling bypass of AI safety measures' reads identically. This is a generic AI safety concern, not anchored to the paper's specific mechanism.",
                    "exposure is pure template: 'Society at large' is the canonical example of non-specific exposure. REJECT.",
                    "vulnerability is pure template: 'Lack of oversight and regulatory frameworks' applies to virtually any AI paper. The author did NOT mention this. REJECT.",
                    "impact items are HALLUCINATED: 'Job loss', 'Privacy erosion', 'Discrimination' are NOT in the impact statement. The model has pattern-matched 'AI risk' → 'standard list of AI harms'. This is the most dangerous failure mode.",
                    "CORRECT extraction would have at most 1-2 slots: hazard (only if it passes substitution), impact=['Potential misuse to bypass safety measures']. Empty fields for everything else."
                ]
            }
        ],
        "output_schema": {
            "hevi": {
                "hazard": ["string"],
                "exposure": ["string"],
                "dose_response": ["string"],
                "vulnerability": ["string"],
                "impact": ["string"],
                "key_control_nodes": ["string"],
            },
        },
    }
    hevi_result = llm.json_call(hevi_system, json.dumps(hevi_user, ensure_ascii=False))

    return {
        "title": title,
        "abstract": abstract,
        "query": query,
        "query_terms": query_terms,
        "ref_hevi": normalize_hevi(hevi_result.get("hevi", {})),
    }


def normalize_hazard_list(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return [
            {"hazard": clean_text(item.get("hazard")), "confidence": item.get("confidence")}
            if isinstance(item, dict) else {"hazard": clean_text(item), "confidence": None}
            for item in raw
            if (isinstance(item, dict) and clean_text(item.get("hazard"))) or (isinstance(item, str) and clean_text(item))
        ]
    if isinstance(raw, str) and clean_text(raw):
        return [{"hazard": clean_text(raw), "confidence": None}]
    return []


def judge_evidence(
    llm: RiskLLM,
    side: str,
    query: str,
    agent_output: Dict[str, Any],
    evidence: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not evidence:
        return []
    judgments = judge_hits_with_llm(llm, side, query, agent_output, evidence)
    return [compact_hit(item, judgments[idx]) for idx, item in enumerate(evidence)]


def load_completed(base_dir: Path) -> set[str]:
    if not base_dir.exists():
        return set()
    completed = set()
    for item in base_dir.iterdir():
        if item.is_dir() and (item / "5_compare.json").exists():
            completed.add(item.name)
    return completed


def write_summary_csv(path: Path, papers: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["title", "reference_item_count", "matched_item_count", "item_recall"],
        )
        writer.writeheader()
        writer.writerows(papers)


def build_ss_queries(cs_proposal: Dict[str, Any], llm: Any) -> Dict[str, Any]:
    nexus_sources = []
    for nexus in cs_proposal.get("nexus_candidates", []) or []:
        nexus_sources.append({
            "scenario": str(nexus.get("scenario", "")),
            "issue": str(nexus.get("issue", "")),
        })

    system = (
        "You are a social-science literature retrieval query planner. "
        "Given the CS Agent's Hazard -> Exposure proposal, produce one retrieval query for finding social-science "
        "evidence about stakeholders, vulnerability, impacts, institutions, governance, and controls. "
        "Keep the query grounded in the CS proposal and avoid inventing new application domains. "
        "Return valid JSON only."
    )
    user = {
        "task": "Build one merged SS retrieval query from all CS Nexus candidates.",
        "cs_agent_output": {
            "hazard": cs_proposal.get("hazard"),
            "nexus_candidates": cs_proposal.get("nexus_candidates", []),
        },
        "rules": [
            "CRITICAL: the 'query' field MUST be a JSON array of strings. Do NOT output a single space-separated string.",
            "Use stakeholder, vulnerability, impact, governance, mitigation, trust, fairness, privacy, safety, or accountability terms only when useful for the CS proposal.",
            "Preserve concrete scenarios and issues from the CS Nexus candidates.",
            "Use 10-30 concise terms or phrases. Keep multi-word concepts together as single array elements.",
            "Return a single query that searches all social-risk slots together.",
        ],
        "output_schema": {
            "query": ["array of strings, each a term or phrase, not a flat string"],
            "social_anchors": ["stakeholder/vulnerability/impact/control anchors used in the query"],
            "rationale": "one short sentence",
        },
    }
    plan = llm.json_call(system, json.dumps(user, ensure_ascii=False))
    raw_query = plan.get("query")
    if not isinstance(raw_query, list):
        raise ValueError(f"SS query must be a list of strings, got {type(raw_query).__name__}: {str(raw_query)[:200]}")
    query_terms = [str(t).strip() for t in raw_query if str(t).strip()]
    if not query_terms:
        raise ValueError("LLM returned an empty SS retrieval query.")
    query = " ".join(query_terms)
    return {
        "query_terms": query_terms,
        "ss_queries": [{
            "slot": "all",
            "query": query,
            "reason": str(plan.get("rationale", "")),
            "planner": "llm",
            "source_cs_nexus": nexus_sources,
        }]
    }


def run_bilateral_consensus(
    cs_agent: CSAgent,
    ss_agent: SSAgent,
    llm: RiskLLM,
    cs_proposal: Dict[str, Any],
    ss_response: Dict[str, Any],
    cs_evidence: List[Dict[str, Any]],
    ss_evidence: List[Dict[str, Any]],
    cs_query: str,
    theta: float = 0.8,
    max_rounds: int = 3,
) -> Dict[str, Any]:
    """
    Run the bilateral consensus protocol.
    CS and SS agents critique each other's proposals in rounds,
    revise based on feedback, and converge when both self_score >= theta.
    After convergence (or max_rounds), synthesize Dose-Response chains.
    """
    rounds_log = []
    cs_score = _parse_self_score(cs_proposal.get("self_score"))
    ss_score = _parse_self_score(ss_response.get("self_score"))

    for round_num in range(1, max_rounds + 1):
        # Require at least 1 round; only check convergence from round 2 onward
        if round_num >= 2 and cs_score >= theta and ss_score >= theta:
            break

        print(f"[consensus round {round_num}] cs_score={cs_score:.2f} ss_score={ss_score:.2f} — critiquing", flush=True)

        cs_critique = cs_agent.critique_ss(ss_response, cs_evidence, cs_proposal)
        ss_critique = ss_agent.critique_cs(cs_proposal, ss_evidence)

        print(f"[consensus round {round_num}] revising", flush=True)

        cs_proposal = cs_agent.revise_proposal(cs_proposal, ss_critique, cs_evidence, cs_query)
        ss_response = ss_agent.revise_response(ss_response, cs_critique, ss_evidence, cs_proposal)

        cs_score = _parse_self_score(cs_proposal.get("self_score"))
        ss_score = _parse_self_score(ss_response.get("self_score"))

        rounds_log.append({
            "round": round_num,
            "cs_critique": cs_critique,
            "ss_critique": ss_critique,
            "cs_self_score": round(cs_score, 4),
            "ss_self_score": round(ss_score, 4),
        })

        print(f"[consensus round {round_num}] done cs_score={cs_score:.2f} ss_score={ss_score:.2f}", flush=True)

    converged = cs_score >= theta and ss_score >= theta
    print(f"[consensus] {'converged' if converged else 'max rounds reached'} after {len(rounds_log)} round(s)", flush=True)

    # Dose-Response Synthesis
    print("[consensus] synthesizing Dose-Response chains", flush=True)
    dr_chains = _synthesize_dose_response(llm, cs_proposal, ss_response, cs_evidence, ss_evidence)

    # If consensus did not converge, mark all chains as unverified
    if not converged:
        for chain in dr_chains.get("chains", []):
            chain["confidence"] = "none"
        dr_chains.setdefault("unsupported_or_missing", []).append(
            "Bilateral consensus did not converge "
            f"(cs_score={cs_score:.2f}, ss_score={ss_score:.2f}, theta={theta}). "
            "The SS-side claims below were not fully verified by the protocol."
        )

    # Extract HEVI slots from final proposals
    hazard = _extract_hazard(cs_proposal)
    exposure = _extract_exposure(cs_proposal)
    vuln = _extract_slot(ss_response, "vulnerability")
    impact = _extract_slot(ss_response, "impact")
    kcn = _extract_slot(ss_response, "key_control_nodes")

    return {
        "rounds": len(rounds_log),
        "converged": converged,
        "consensus_trace": rounds_log,
        "cs_self_score_final": round(cs_score, 4),
        "ss_self_score_final": round(ss_score, 4),
        "chains": dr_chains.get("chains", []),
        "unsupported_or_missing": dr_chains.get("unsupported_or_missing", []),
        "hazard": hazard,
        "exposure": exposure,
        "vulnerability": vuln,
        "impact": impact,
        "key_control_nodes": kcn,
    }


def _parse_self_score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _extract_hazard(cs_proposal: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = cs_proposal.get("hazard", [])
    if not isinstance(raw, list):
        raw = [raw] if raw else []
    return [
        {"hazard": clean_text(item.get("hazard")), "confidence": item.get("confidence")}
        if isinstance(item, dict) else {"hazard": clean_text(item), "confidence": None}
        for item in raw
        if isinstance(item, dict) and clean_text(item.get("hazard"))
    ]


def _extract_exposure(cs_proposal: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates = cs_proposal.get("nexus_candidates", [])
    if not isinstance(candidates, list):
        return []
    return [
        {
            "scenario": clean_text(item.get("scenario")),
            "issue": clean_text(item.get("issue")),
            "exposure": clean_text(item.get("exposure")),
            "confidence": item.get("confidence"),
        }
        for item in candidates
        if isinstance(item, dict) and (item.get("scenario") or item.get("exposure"))
    ]


def _extract_slot(ss_response: Dict[str, Any], key: str) -> List[str]:
    items = []
    for resp in ss_response.get("nexus_responses", []) or []:
        if not isinstance(resp, dict):
            continue
        for value in resp.get(key, []) or []:
            cleaned = clean_text(value)
            if cleaned and cleaned not in items:
                items.append(cleaned)
    return items


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
