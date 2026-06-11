import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from src.hipporag.hevi_workflow.agents import CSAgent, RiskLLM, SSAgent
from src.hipporag.hevi_workflow.hit_report import (
    compact_cs_output,
    compact_hit,
    compact_ss_output,
    judge_hits_with_llm,
    keywords,
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


def build_keyword_query(paper: Dict[str, str]) -> str:
    query_terms: List[str] = []
    for term in keywords(paper.get("title", ""), max_terms=12) + keywords(paper.get("abstract", ""), max_terms=28):
        if term not in query_terms:
            query_terms.append(term)
    return " ".join(query_terms)


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
    fallback_query = build_keyword_query(paper)
    system = (
        "You create the first input file for an HEVI risk workflow. "
        "This is one integrated preparation step with two strictly separated outputs. "
        "First, build a CS retrieval query using only the title and abstract. "
        "Second, extract a reference HEVI schema from the ICML impact statement. "
        "HEVI is a RISK assessment framework. Your job is to identify risks the author ACKNOWLEDGES "
        "— NOT to summarize the paper's benefits, contributions, or the problems it solves. "
        "The title and abstract provide technical context to resolve vague references in the impact statement "
        "(e.g., what 'our algorithm' concretely means), but they cannot be used to invent new risk claims. "
        "Do not use the impact statement to write the query. "
        "If the impact statement contains NO acknowledged risks — only benefits, contributions, or claims of "
        "'no negative impact' — leave ALL HEVI fields empty. "
        "Return valid JSON only."
    )
    user = {
        "task": "Build workflow_input: title, abstract, query, and impact-grounded HEVI reference.",
        "icml": {
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "impact": paper.get("impact", ""),
        },
        "query_rules": [
            "Use exact method names, task names, mechanisms, models, datasets, and problem settings from title/abstract.",
            "Use 8-24 concise terms or phrases.",
            "Do not add social impact, fairness, privacy, governance, or safety terms unless explicit in title/abstract.",
            "Prefer specific named entities over broad category labels.",
            "Do not include terms for approaches or methods the paper explicitly argues against.",
        ],
        "hevi_rules": [
            "CRITICAL — HEVI is a RISK assessment. Only extract claims where the author acknowledges a potential HARM, NEGATIVE consequence, or DANGER. If the impact statement only describes what the paper improves, solves, reduces, or contributes, leave all fields EMPTY.",
            "The title and abstract provide technical context ONLY to resolve vague terms in the impact (e.g., 'our attack' → 'Charmer query-based character-level attack'). Do NOT use title/abstract to add risk claims the impact does not mention.",
            "Every HEVI item must be a concise phrase grounded in the impact statement. Empty fields are preferred over hallucination.",
            "A single fact from the impact must appear in exactly ONE slot. Never duplicate.",
        ],
        "field_definitions": {
            "hazard": "A technical capability, behavior, or mechanism that the paper INTRODUCES or SHARPENS, which the author says COULD cause harm. NOT the problem the paper aims to solve. Example: 'Charmer query-based adversarial attack enabling malicious evasion of NLP systems' IS a hazard (the paper created the attack). 'Hallucination in transformers' is NOT a hazard (it's a pre-existing problem the paper tries to solve).",
            "exposure": "WHO (people, groups) or WHAT (systems, domains) the author says could be EXPOSED to the hazard. Must be a concrete setting, not a generic domain. Example: 'users relying on sentiment analysis for business decisions' IS exposure. 'healthcare and finance' is NOT (it's just application domains).",
            "dose_response": "How SCALE, FREQUENCY, or INTENSITY of the hazard translates into more or worse harm. Almost never mentioned in impact statements — leave empty unless the author explicitly discusses thresholds or scaling relationships.",
            "vulnerability": "A CONDITION, GAP, or WEAKNESS the author says makes harm MORE LIKELY or MORE SEVERE. NOT the gap the paper fills. Example: 'lack of adversarial robustness in deployed NLP models' IS a vulnerability. 'Lack of systematic debugging tools prior to this work' is NOT (it's the paper's motivation).",
            "impact": "A NEGATIVE social, ethical, economic, or safety CONSEQUENCE the author says COULD result from the hazard. NOT a benefit the paper provides. Example: 'malicious evasion of content filtering harming platform users' IS an impact. 'Reducing hallucination' or 'lowering energy costs' are NOT (they are benefits).",
            "key_control_nodes": "An INTERVENTION, SAFEGUARD, or MITIGATION the author mentions. Can be: (a) an action the author TAKES to reduce risk, e.g., 'code release for defense assessment', 'benchmark for safety evaluation'; (b) a technical/operational control, e.g., 'human-in-the-loop verification', 'adversarial training'. The paper's METHOD itself is NOT a KCN — 'AttnLRP debugging method' is the paper's contribution, not a control node.",
        },
        "examples": [
            {
                "scenario": "Paper with genuine risk acknowledgment",
                "impact": "Our Charmer attack algorithm could enable malicious evasion of NLP systems. We release code to help defenders assess robustness.",
                "correct_extraction": {
                    "hazard": ["Charmer query-based character-level adversarial attack enabling malicious evasion of NLP systems"],
                    "exposure": ["Users of deployed NLP systems (sentiment analysis, content moderation) relying on model outputs"],
                    "dose_response": [],
                    "vulnerability": ["Lack of adversarial robustness in deployed NLP models", "Unawareness that character-level attacks cannot easily be defended"],
                    "impact": ["Malicious evasion of content filtering exposing users to harmful content", "Manipulation of sentiment analysis leading to flawed business decisions"],
                    "key_control_nodes": ["Public code release for defense assessment and hardening"]
                }
            },
            {
                "scenario": "Paper with only benefits — NO risk acknowledged",
                "impact": "Our XAI method reduces hallucination in transformers, lowering costs and enabling adoption in healthcare. We release a benchmark for explanation evaluation.",
                "correct_extraction": {
                    "hazard": [],
                    "exposure": [],
                    "dose_response": [],
                    "vulnerability": [],
                    "impact": [],
                    "key_control_nodes": []
                },
                "explanation": "ALL empty because the impact only describes benefits (reducing hallucination, lowering costs) and contributions (benchmark release). The author does not acknowledge any risk. 'Hallucination in transformers' is the problem the paper solves, not a hazard it introduces. 'Benchmark release' is a contribution, not a risk mitigation."
            }
        ],
        "output_schema": {
            "title": "same title",
            "abstract": "same abstract",
            "query": "single CS retrieval query string",
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
    try:
        row = llm.json_call(system, json.dumps(user, ensure_ascii=False))
        query = clean_text(row.get("query")) or fallback_query
        if not query:
            query = fallback_query
    except Exception as exc:
        print(f"[warn] workflow_input LLM fallback to keyword query: {exc}", flush=True)
        query = fallback_query
        row = {"hevi": {}}

    return {
        "title": paper.get("title", ""),
        "abstract": paper.get("abstract", ""),
        "query": query,
        "ref_hevi": normalize_hevi(row.get("hevi", {})),
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


def build_ss_queries(cs_proposal: Dict[str, Any], llm: Any | None = None) -> Dict[str, Any]:
    query_terms = []
    nexus_sources = []
    for nexus in cs_proposal.get("nexus_candidates", []) or []:
        nexus_scenario = str(nexus.get("scenario", ""))
        nexus_issue = str(nexus.get("issue", ""))
        nexus_sources.append({"scenario": nexus_scenario, "issue": nexus_issue})
        for term in keywords(f"{nexus_scenario} {nexus_issue}", max_terms=18):
            if term not in query_terms:
                query_terms.append(term)

    risk_terms = (
        "exposure stakeholders vulnerability vulnerable groups social impact harm trust fairness "
        "privacy safety misinformation governance oversight audit accountability mitigation"
    )
    for term in keywords(risk_terms, max_terms=18):
        if term not in query_terms:
            query_terms.append(term)

    fallback_query = " ".join(query_terms)
    if llm is None:
        return {
            "ss_queries": [{
                "slot": "all",
                "query": fallback_query,
                "reason": "Merged SS retrieval query from all CS Nexus candidates.",
                "planner": "keyword_fallback",
                "source_cs_nexus": nexus_sources,
            }]
        }

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
            "Use stakeholder, vulnerability, impact, governance, mitigation, trust, fairness, privacy, safety, or accountability terms only when useful for the CS proposal.",
            "Preserve concrete scenarios and issues from the CS Nexus candidates.",
            "Use 10-30 concise terms or phrases.",
            "Return a single query that searches all social-risk slots together.",
        ],
        "output_schema": {
            "query": "single retrieval query string",
            "social_anchors": ["stakeholder/vulnerability/impact/control anchors used in the query"],
            "rationale": "one short sentence",
        },
    }
    try:
        plan = llm.json_call(system, json.dumps(user, ensure_ascii=False))
        query = " ".join(str(plan.get("query", "")).replace("\n", " ").split()) or fallback_query
        return {
            "ss_queries": [{
                "slot": "all",
                "query": query,
                "reason": str(plan.get("rationale", "")),
                "planner": "llm",
                "source_cs_nexus": nexus_sources,
            }]
        }
    except Exception:
        return {
            "ss_queries": [{
                "slot": "all",
                "query": fallback_query,
                "reason": "fallback_keyword_query_after_llm_error",
                "planner": "keyword_fallback",
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
        if cs_score >= theta and ss_score >= theta:
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
        "Your task is NOT to generate new content — it is to splice the CS Agent's verified Hazard->Exposure segment "
        "with the SS Agent's verified Vulnerability->Impact segment at the agreed Nexus (scenario, issue). "
        "The Dose-Response chain is the causal translation: HOW does the technical capability, through the exposure scenario, "
        "lead to the social impact? This is the spliced product of bilateral consensus, not a centrally-coordinated generation. "
        "Each chain must be traceable to both CS and SS evidence. "
        "If the two segments cannot be coherently spliced for a given Nexus, note it in unsupported_or_missing. "
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
            "dose_response must be a concrete causal narrative: 'X enables Y, which in scenario Z exposes W, leading to V'.",
            "If dose_response cannot be constructed from available segments, leave it empty.",
            "Cite evidence by title and short point — no internal IDs.",
            "Do not invent content not present in the CS or SS segments.",
        ],
        "output_schema": {
            "chains": [
                {
                    "scenario": "agreed scenario from Nexus",
                    "issue": "agreed issue from Nexus",
                    "hazard": ["string"],
                    "exposure": ["string"],
                    "dose_response": ["causal chain: technical capability → exposure → social impact"],
                    "vulnerability": ["string"],
                    "impact": ["string"],
                    "key_control_nodes": ["string"],
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
