"""
Independent quality audit for extracted ref_hevi JSON files.
Reads from hevi_icml dir, scores each paper, outputs quality_report.json.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipporag.hevi_workflow.agents import RiskLLM
from hipporag.hevi_workflow.utils import load_project_api_key, write_json


def audit_paper(llm: RiskLLM, paper: Dict[str, Any]) -> Dict[str, Any]:
    system = (
        "You are a quality auditor for HEVI risk extraction. "
        "Evaluate whether the extracted HEVI schema correctly captures risks acknowledged in the paper's impact statement. "
        "Be strict — prefer rejecting borderline cases over accepting low-quality extractions. "
        "Return valid JSON only."
    )
    user = {
        "task": "Audit the quality of this HEVI extraction. Score every dimension using the operational tests provided. Give a clear keep or reject verdict.",
        "paper": {
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", "")[:500],
            "impact": paper.get("impact", ""),
        },
        "cs_query": paper.get("query_terms", paper.get("query", "")) if isinstance(paper.get("query_terms"), list) else [],
        "extracted_hevi": paper.get("ref_hevi", {}),
        "rubric": {
            "discovery_feasibility": {
                "weight": 0.25,
                "question": "Can CS retrieval using Abstract-derived technical terms find papers discussing the Impact's risk direction? In other words: are Abstract and Impact in the same semantic space?",
                "operational_test": [
                    "1. List technical terms from the Abstract (method names, task names, model names, datasets).",
                    "2. List risk terms from the Impact (harm, attack, bias, toxic, privacy, discrimination, misuse, safety, security, etc.).",
                    "3. Check: is there semantic overlap or lexical association between the two lists?",
                    "4. If Abstract terms are purely mathematical/statistical/engineering while Impact discusses social/ethical risks → DF < 0.4 (hard reject).",
                    "5. If Abstract terms directly cover the technology that enables the Impact risk → DF ≥ 0.8.",
                ],
                "high_example": "Abstract: 'character-level adversarial attacks for language models'. Impact: 'malicious evasion of content filtering'. Technical term 'adversarial attacks' and risk term 'malicious evasion' share the AI security/safety semantic space. DF ≥ 0.8.",
                "low_example_1": "Abstract: 'dynamic survival analysis with controlled differential equations'. Impact: 'biased outcomes affecting disadvantaged groups'. Abstract terms are all math/stats — no overlap with bias/fairness. DF < 0.4.",
                "low_example_2": "Abstract: 'transformer for molecular representation learning'. Impact: 'designing toxic substances or environmentally harmful molecules'. Abstract has no dual-use/biosecurity/toxicology terms. DF < 0.4.",
            },
            "technical_anchoring": {
                "weight": 0.20,
                "question": "Are the extracted risk claims CAUSALLY LINKED to the paper's SPECIFIC technical contribution, or are they generic templates?",
                "operational_test": [
                    "For each HAZARD item: copy the text. Replace the paper's specific method name with '[Method X]'. Does the risk still read as valid? If YES → anchoring ≤ 0.3. This is the SUBSTITUTION TEST.",
                    "For each VULNERABILITY item: replace the hazard reference with a generic AI hazard. Does the sentence read as a standard AI ethics talking point? If YES → anchoring ≤ 0.3.",
                    "For each IMPACT item: check if it describes a SPECIFIC causal chain (HOW the hazard causes THIS consequence) vs. a GENERIC association (hazard X is loosely associated with harm Y). Generic → anchoring ≤ 0.3.",
                ],
                "fail_example": "Hazard='Self-improving LM policy exacerbates biases in small datasets'. Replace 'Self-improving LM policy' → 'Fine-tuned LM' → reads identically. This FAILS the substitution test — it's a property of any LM training on small data, not specific to this paper's method. Anchoring ≤ 0.3.",
                "pass_example": "Hazard='Query-based character-level adversarial attack enabling malicious evasion of NLP systems'. Replace 'Query-based character-level adversarial attack' → 'Contrastive learning for sentence embeddings' → NONSENSICAL. This PASSES — the risk is causally tied to adversarial attack capability. Anchoring ≥ 0.8.",
            },
            "direction_correctness": {
                "weight": 0.20,
                "question": "Does each HEVI item describe a RISK (harm, negative consequence, danger) rather than a benefit, contribution, or the problem being solved?",
                "operational_test": [
                    "impact items: do they describe NEGATIVE outcomes (harms) or POSITIVE outcomes (benefits)? 'Lowering costs' is a benefit — direction=0.",
                    "hazard items: do they name a capability the paper INTRODUCES that could cause harm, or the problem the paper SOLVES? 'Hallucination in transformers' is the problem being solved — direction=0.",
                    "KCN items: are they risk mitigations, or are they the paper's METHOD/CONTRIBUTION? Method names as KCN → direction=0.",
                    "exposure items: are they describing WHO is at risk, or are they describing beneficiaries?",
                ],
                "common_failures": [
                    "impact written as 'reduces X' or 'improves Y' — these are benefits, not negative consequences.",
                    "hazard written as the problem the paper solves (e.g. XAI paper's hazard = 'hallucination in transformers').",
                    "KCN written as the paper's method name rather than a risk control measure.",
                ],
            },
            "grounding": {
                "weight": 0.15,
                "question": "Can each HEVI item be traced back to something the author ACTUALLY SAID in the impact statement? Fabricated or over-inferred items score low.",
                "operational_test": [
                    "For each non-empty HEVI item: find the EXACT sentence or phrase in the impact statement that supports it.",
                    "If an item has NO clear textual anchor in the impact → grounding for that item = 0.",
                    "If the item is a reasonable inference but the author didn't explicitly say it → partial score.",
                    "If the item directly contradicts what the author said → grounding = 0.",
                ],
                "common_failure": "Impact items like 'job loss, privacy erosion, discrimination' appearing when the impact statement only says 'could be misused' — these are the model pattern-matching 'AI risk' → 'standard list of harms'. This is hallucination.",
            },
            "slot_correctness": {
                "weight": 0.10,
                "question": "Is each item placed in the CORRECT HEVI slot? Use the differential diagnosis table.",
                "slot_definitions": {
                    "hazard": "Technical capability/behavior the paper INTRODUCES that could cause harm. NOT a limitation, NOT the problem solved.",
                    "exposure": "WHO (specific people/groups/systems) + brief context of exposure. NOT a generic domain name. NOT impact/vulnerability leakage.",
                    "dose_response": "Scale/frequency/intensity → worse harm. Almost never in impact statements — should be EMPTY.",
                    "vulnerability": "CONDITION/GAP/WEAKNESS making harm more likely/severe. NOT the research gap the paper fills.",
                    "impact": "NEGATIVE social/ethical/economic CONSEQUENCE. NOT a benefit or improvement.",
                    "key_control_nodes": "INTERVENTION/SAFEGUARD/MITIGATION. Author action or technical control. NOT the paper's method. NOT research proposals.",
                },
                "differential_diagnosis": [
                    "hazard vs vulnerability: does the item describe something the method DOES (→ hazard) or something MISSING (→ vulnerability)?",
                    "exposure vs impact: does the item name a person/group/setting (→ exposure) or a harm/outcome (→ impact)?",
                    "KCN vs contribution: would this action exist even without the risk? If YES → contribution, not KCN.",
                ],
            },
            "specificity": {
                "weight": 0.10,
                "question": "Are HEVI items SPECIFIC (named methods, specific groups, concrete harms) or generic boilerplate?",
                "operational_test": [
                    "Could this EXACT item text appear in another paper's impact statement from a DIFFERENT subfield? If YES → specificity ≤ 0.3.",
                    "Does the item name a SPECIFIC method, group, harm type, or scenario? If it uses vague language ('certain groups', 'negative implications', 'various harms') → specificity ≤ 0.3.",
                ],
                "common_failures": [
                    "'Society at large' or 'General public' as exposure — no specific group identified.",
                    "'Lack of regulation/oversight/ethical guidelines' as standalone vulnerability — applies to any AI paper.",
                    "'Negative societal implications' as impact — no specific harm named.",
                    "'Bias and unfairness' without specifying type, target, or scenario.",
                ],
            },
        },
        "diagnostic_checks": {
            "purpose": "These checks are INFORMATIONAL ONLY — they do NOT affect the keep/reject verdict. They help diagnose pipeline failures.",
            "completeness": {
                "question": "Re-read the impact statement. Did the extractor MISS any risk the author clearly acknowledges?",
                "instructions": "If the author explicitly mentions a risk that was NOT captured in any HEVI slot, list it under 'missed_risks'. This does NOT penalize the extraction score (conservative is better than hallucination), but flags papers where the Ref HEVI may be incomplete.",
            },
            "query_quality": {
                "question": "Is the CS query well-constructed for retrieval?",
                "checks": [
                    "Are all query terms TECHNICAL (method, task, model, dataset names) rather than filler words?",
                    "Are there any NOISE terms (paper title fragments like 'Revisiting', 'A Novel Approach')?",
                    "Does the query include SYNONYMS or alternative phrasings for core concepts?",
                    "Would this query realistically find papers related to the risk direction? Rate as: good | adequate | poor.",
                ],
            },
        },
        "verdict_rules": {
            "keep": "overall_score ≥ 0.75 AND discovery_feasibility ≥ 0.40 AND technical_anchoring ≥ 0.40 AND ≥3 non-empty HEVI slots",
            "reject": "All cases not meeting keep criteria.",
            "hard_reject_df": "discovery_feasibility < 0.40 → REJECT (Abstract and Impact not in the same semantic space — pipeline cannot work).",
            "hard_reject_anchoring": "technical_anchoring < 0.40 → REJECT (risk claims are generic templates — extraction quality is too low).",
            "hard_reject_slots": "<3 non-empty HEVI slots → REJECT (insufficient risk information).",
        },
        "output_schema": {
            "overall_score": "0.0-1.0 weighted total (apply the weights in the rubric)",
            "verdict": "keep | reject",
            "scores": {
                "discovery_feasibility": "0.0-1.0",
                "technical_anchoring": "0.0-1.0",
                "direction_correctness": "0.0-1.0",
                "grounding": "0.0-1.0",
                "slot_correctness": "0.0-1.0",
                "specificity": "0.0-1.0",
            },
            "issues": ["specific problems found — empty list if none"],
            "strengths": ["what the extraction did well"],
            "summary": "one-sentence quality assessment",
            "diagnostics": {
                "missed_risks": ["risks the author acknowledges but the extractor missed — empty if none"],
                "query_quality": "good | adequate | poor",
                "query_notes": "brief note on query construction quality",
            },
        },
    }
    try:
        result = llm.json_call(system, json.dumps(user, ensure_ascii=False))
    except Exception as exc:
        result = {
            "overall_score": 0.0,
            "verdict": "reject",
            "scores": {},
            "issues": [f"audit LLM call failed: {exc}"],
            "strengths": [],
            "summary": "audit failed",
        }

    # Hard filter: require at least 3 non-empty HEVI slots
    ref_hevi = paper.get("ref_hevi", {})
    non_empty = sum(1 for v in ref_hevi.values() if isinstance(v, list) and len(v) > 0)
    if non_empty < 3:
        result["verdict"] = "reject"
        result.setdefault("issues", []).append(
            f"Only {non_empty} non-empty HEVI slot(s) — minimum 3 required. Paper has insufficient risk acknowledgment."
        )

    # Hard filter: technical_anchoring
    ta = result.get("scores", {}).get("technical_anchoring", 0.5)
    try:
        ta = float(ta)
    except (TypeError, ValueError):
        ta = 0.5
    if ta < 0.40:
        result["verdict"] = "reject"
        result.setdefault("issues", []).append(
            f"Technical anchoring score ({ta:.2f}) below 0.40 threshold — risk claims are generic templates."
        )

    # Normalize all score values to float (LLM may return strings)
    scores = result.get("scores", {})
    if isinstance(scores, dict):
        for key in list(scores.keys()):
            try:
                scores[key] = float(scores[key])
            except (TypeError, ValueError):
                scores[key] = 0.0
    try:
        result["overall_score"] = float(result.get("overall_score", 0))
    except (TypeError, ValueError):
        result["overall_score"] = 0.0

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit extracted HEVI reference files for quality.")
    parser.add_argument("--input-dir", default="outputs/hevi_workflow/hevi_icml_{model}")
    parser.add_argument("--output", default="outputs/hevi_workflow/hevi_icml_{model}/quality_report.json")
    parser.add_argument("--llm-name", default="deepseek-v4-pro")
    parser.add_argument("--llm-base-url", default="https://www.highland-api.top/v1")
    parser.add_argument("--max-new-tokens", type=int, default=384000)
    parser.add_argument("--llm-timeout", type=int, default=120)
    parser.add_argument("--llm-retries", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="Max papers to audit (0 = all)")
    parser.add_argument("--paper-ids", default="", help="Comma-separated paper IDs to audit (overrides --limit)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.max_new_tokens == 0:
        if "deepseek" in args.llm_name.lower():
            args.max_new_tokens = 384000
        elif "gpt" in args.llm_name.lower():
            args.max_new_tokens = 128000
        else:
            args.max_new_tokens = 128000

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    for logger_name in ("httpx", "httpcore", "openai", "openai._base_client"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    load_project_api_key()

    model_tag = args.llm_name.replace("/", "_").replace(" ", "-")
    input_dir = Path(str(args.input_dir).replace("{model}", model_tag))
    output_path = Path(str(args.output).replace("{model}", model_tag))

    if not input_dir.exists():
        print(f"Error: input dir {input_dir} not found", flush=True)
        sys.exit(1)

    # Load existing report for resume
    existing_scores: Dict[str, Any] = {}
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        existing_scores = existing.get("papers", {})
        print(f"Resume: {len(existing_scores)} papers already audited", flush=True)

    # Load papers, skip already audited
    papers = []
    paper_ids = []
    for f in sorted(input_dir.glob("icml_*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        pid = f.stem  # icml_2024_0001
        data["paper_id"] = pid
        if pid not in existing_scores:
            papers.append(data)
            paper_ids.append(pid)

    if args.paper_ids:
        ids = set(args.paper_ids.split(","))
        # Remove from existing so they get re-audited
        for pid in ids:
            existing_scores.pop(pid, None)
        # Reload papers including already-audited ones
        papers = []
        for f in sorted(input_dir.glob("icml_*.json")):
            data = json.loads(f.read_text(encoding="utf-8"))
            pid = f.stem
            if pid in ids:
                data["paper_id"] = pid
                papers.append(data)
        print(f"Re-auditing {len(papers)} papers by --paper-ids", flush=True)
    elif args.limit > 0 and len(papers) > args.limit:
        papers = papers[:args.limit]

    print(f"Loaded {len(papers)} papers to audit ({len(existing_scores)} skipped)", flush=True)

    # Audit
    llm = RiskLLM(
        save_dir=str(input_dir.parent),
        llm_name=args.llm_name,
        llm_base_url=args.llm_base_url,
        max_new_tokens=args.max_new_tokens,
        request_timeout=args.llm_timeout,
        max_retries=args.llm_retries,
    )

    results: Dict[str, Any] = dict(existing_scores)
    verdicts: Dict[str, int] = {}
    total_score = 0.0

    for idx, paper in enumerate(papers, start=1):
        title = paper.get("title", "")[:60]
        print(f"[{idx}/{len(papers)}] auditing {title}", flush=True)
        audit = audit_paper(llm, paper)
        results[paper.get("paper_id", f"unknown_{idx}")] = audit
        v = audit.get("verdict", "reject")
        verdicts[v] = verdicts.get(v, 0) + 1
        total_score += audit.get("overall_score", 0)
        print(f"  score={audit.get('overall_score',0):.2f} → {v.upper()}", flush=True)

    avg_score = round(total_score / len(papers), 3) if papers else None

    report = {
        "audited": len(papers),
        "avg_score": avg_score,
        "verdicts": verdicts,
        "papers": results,
    }
    write_json(output_path, report)

    print(f"\nDone. keep={verdicts.get('keep',0)} reject={verdicts.get('reject',0)} avg_score={avg_score}", flush=True)
    print(f"Report: {output_path}", flush=True)


if __name__ == "__main__":
    main()
