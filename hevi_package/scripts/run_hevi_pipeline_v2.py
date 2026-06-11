#!/usr/bin/env python3
"""
HEVI v2 Pipeline — Slot-aware completion.

Unlike v1 (which generates chains from scratch, blind to impact statement),
v2 preserves existing ref_hevi items and fills only empty slots.

Flow (7 LLM calls per paper):
  Step 1: Nexus extraction — group ref_hevi items into 1-2 causal chains    (1 call)
  Step 2: CS completion — fill empty hazard/exposure via CS retrieval        (2 calls)
  Step 3: SS query build                                                     (1 call)
  Step 4: SS completion — fill empty vuln/impact/KCN via SS retrieval        (2 calls)
  Step 5: DR synthesis — splice CS+SS segments at each Nexus                 (1 call)

Output: outputs/hevi_workflow_v2/{paper_id}.json
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
from hipporag.hevi_workflow.hit_report import (
    compact_cs_output,
    compact_ss_output,
    judge_hits_with_llm,
)
from hipporag.hevi_workflow.pipeline import HEVI_KEYS, clean_text, judge_evidence, strip_internal_ids
from hipporag.hevi_workflow.retrievers import RiskRetriever
from hipporag.hevi_workflow.utils import load_project_api_key, write_json
from hipporag.hevi_workflow.hevi_framework import HEVI_FRAMEWORK, HEVI_SHORT

# ── helpers ──────────────────────────────────────────────────────────

def _compact_evidence(evidence, max_chars=900):
    compact = []
    for item in evidence:
        abstract = str(item.get("abstract") or item.get("doc_text") or "")
        compact.append({
            "evidence_id": item.get("evidence_id"),
            "paper_id": item.get("paper_id"),
            "title": item.get("title"),
            "score": item.get("score"),
            "abstract": abstract[:max_chars],
        })
    return compact

def count_items(hevi, key):
    items = hevi.get(key, [])
    return len([x for x in (items if isinstance(items, list) else []) if str(x).strip()])

def _require(result, path, what="field"):
    """Validate that a nested path exists in result. Raise if missing."""
    keys = path.split(".")
    val = result
    for k in keys:
        if isinstance(val, dict):
            if k not in val:
                raise ValueError(f"LLM output missing '{path}' ({what})")
            val = val[k]
        elif isinstance(val, list):
            # If we index into a list, check all items
            if not val:
                raise ValueError(f"LLM output has empty list at '{path}' ({what})")
            val = val[0]  # check first item for schema
        else:
            raise ValueError(f"LLM output '{path}' is not dict/list ({what})")
    if val is None or (isinstance(val, (list, str)) and len(val) == 0):
        raise ValueError(f"LLM output '{path}' is empty ({what})")
    return val

def _require_list(result, path, what="field"):
    val = _require(result, path, what)
    if not isinstance(val, list):
        raise ValueError(f"LLM output '{path}' must be a list, got {type(val).__name__} ({what})")
    return val

# ── Step 1: Nexus extraction ────────────────────────────────────────

def extract_nexus(llm: RiskLLM, paper: dict) -> dict:
    """Group ref_hevi flat items into 1-2 causal chains with Nexus."""
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    impact = paper.get("impact", "")
    ref_hevi = paper.get("ref_hevi", {})

    system = (
        "You are a HEVI risk chain reconstructor. "
        "Given a paper's flat ref_hevi items (unlinked across six slots), "
        "your job is to group them into 1-2 causal chains, each anchored by a PLACE-BASED Nexus (scenario, issue). "
        "A scenario is a specific deployment setting — vulnerability varies by place. "
        + HEVI_FRAMEWORK + "\n"
        "CRITICAL RULES:\n"
        "1. Output AT MOST 2 chains. Prefer 1. Only use 2 if items clearly belong to two unrelated risk directions.\n"
        "2. hazard and exposure are the SKELETON — at most ONE item per chain. They define the perturbation and who faces it.\n"
        "3. vulnerability, impact, key_control_nodes can have MULTIPLE items per chain. "
        "vulnerability = system sensitivity; KCN = coping capacity at individual/institutional/policy levels.\n"
        "4. Every ref_hevi item MUST be assigned to exactly one chain. Do not drop items.\n"
        "5. If hazard+exposure are ALL empty, reverse-engineer the Nexus from vuln/impact/KCN — "
        "infer what scenario and issue these downstream items imply.\n"
        "6. scenario: ≤80 chars, a place-based deployment/application setting label.\n"
        "   issue: ≤100 chars, the risk problem in this specific place.\n"
        "7. If a slot has items in ref_hevi, copy them VERBATIM (do not rewrite). If empty, leave the array empty.\n"
        "8. Keep scenario ≤80 chars, issue ≤100 chars.\n"
        "Return valid JSON ONLY, no other text."
    )
    user = {
        "task": "Reconstruct 1-2 causal risk chains from flat ref_hevi items.",
        "paper": {"title": title, "abstract": abstract[:600], "impact": impact},
        "ref_hevi": ref_hevi,
        "chain_schema": {
            "scenario": "≤80 chars — deployment/application setting label",
            "issue": "≤100 chars — the risk problem in this scenario",
            "hazard": ["≤1 item from ref_hevi, empty if none"],
            "exposure": ["≤1 item from ref_hevi, empty if none"],
            "vulnerability": ["items from ref_hevi"],
            "impact": ["items from ref_hevi"],
            "key_control_nodes": ["items from ref_hevi"],
        },
        "output_schema": {
            "chains": [
                {
                    "scenario": "string",
                    "issue": "string",
                    "hazard": ["string"],
                    "exposure": ["string"],
                    "vulnerability": ["string"],
                    "impact": ["string"],
                    "key_control_nodes": ["string"],
                }
            ]
        },
    }
    result = llm.json_call(system, json.dumps(user, ensure_ascii=False))
    chains = _require_list(result, "chains", "Nexus extraction output")
    for i, c in enumerate(chains):
        _require(c, "scenario", f"chain[{i}] scenario")
        _require(c, "issue", f"chain[{i}] issue")
        for slot in HEVI_KEYS:
            if slot not in c:
                c[slot] = []
    if not isinstance(chains, list) or len(chains) == 0:
        raise ValueError("Nexus extraction returned no chains")
    return chains[:2]

# ── Step 2: CS completion ───────────────────────────────────────────

def complete_cs(llm: RiskLLM, paper: dict, chains: list, cs_evidence_raw: list, cs_query: str) -> list:
    """Fill missing hazard/exposure in each chain using CS evidence. Lock ref_hevi items."""
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")

    # Only process chains that need CS-side completion
    need_cs = [c for c in chains if not c.get("hazard") or not c.get("exposure")]
    if not need_cs:
        return chains

    system = (
        "You are a CS risk completion agent. Fill ONLY empty hazard and exposure slots. "
        "Existing (non-empty) items are LOCKED — do not modify them. "
        "Hazard = perturbation/stress introduced or amplified by the method (≤30 words). "
        "Can originate INSIDE the system (method creates new capability) or OUTSIDE (method amplifies existing risk). "
        "Exposure = who/what system elements face the hazard (≤15 words). "
        "EXPOSURE ≠ SENSITIVITY: describe who is in harm's way, not how easily they're affected. "
        + HEVI_SHORT + " "
        "Return valid JSON ONLY, no other text."
    )
    user = {
        "task": "Fill empty hazard/exposure slots. Lock existing items.",
        "paper": {"title": title, "abstract": abstract[:600]},
        "cs_query": cs_query,
        "cs_evidence": _compact_evidence(cs_evidence_raw),
        "chains_need_completion": [
            {
                "chain_index": idx,
                "scenario": c.get("scenario"),
                "issue": c.get("issue"),
                "hazard": c.get("hazard", []),       # existing or empty
                "exposure": c.get("exposure", []),   # existing or empty
            }
            for idx, c in enumerate(need_cs)
        ],
        "rules": [
            "If hazard is non-empty → LOCKED, do not change.",
            "If exposure is non-empty → LOCKED, do not change.",
            "Only fill slots that are empty.",
            "hazard: one concise phrase (≤30 words). State the technical capability at the paper's level.",
            "exposure: who + brief context (≤15 words). No vulnerability/impact leakage.",
        ],
        "output_schema": {
            "completions": [
                {
                    "chain_index": 0,
                    "hazard": ["filled hazard or empty if still unfillable"],
                    "exposure": ["filled exposure or empty if still unfillable"],
                    "confidence": "low|medium|high",
                }
            ]
        },
    }
    result = llm.json_call(system, json.dumps(user, ensure_ascii=False))
    completions = _require_list(result, "completions", "CS completion output")
    for i, comp in enumerate(completions):
        idx = comp.get("chain_index", i)
        if idx < len(chains):
            if not chains[idx].get("hazard") and comp.get("hazard"):
                chains[idx]["hazard"] = _require_list(comp, "hazard", f"CS completion[{i}] hazard")
            if not chains[idx].get("exposure") and comp.get("exposure"):
                chains[idx]["exposure"] = _require_list(comp, "exposure", f"CS completion[{i}] exposure")
    return chains

# ── Step 3: SS query build ──────────────────────────────────────────

def build_ss_query_v2(llm: RiskLLM, chains: list) -> dict:
    """Build SS retrieval query from completed CS segments + Nexus."""
    nexus_sources = []
    for c in chains:
        nexus_sources.append({
            "scenario": c.get("scenario", ""),
            "issue": c.get("issue", ""),
            "hazard": c.get("hazard", []),
            "exposure": c.get("exposure", []),
        })

    system = "You are a social-science literature retrieval query planner. Return valid JSON ONLY, no other text."
    user = {
        "task": "Build one merged SS retrieval query from all chains.",
        "chains": nexus_sources,
        "rules": [
            "CRITICAL: 'query' MUST be a JSON array of strings.",
            "Use stakeholder, vulnerability, impact, governance, mitigation, trust, fairness, privacy, safety terms where relevant.",
            "Preserve concrete scenarios and issues from the CS segments.",
            "Use 10-30 concise terms or phrases.",
        ],
        "output_schema": {
            "query": ["array of strings"],
            "rationale": "one short sentence",
        },
    }
    plan = llm.json_call(system, json.dumps(user, ensure_ascii=False))
    ss_terms = _require_list(plan, "query", "SS query build")
    query_terms = [str(t).strip() for t in ss_terms if str(t).strip()]
    if not query_terms:
        raise ValueError("SS query build returned empty query terms")
    query = " ".join(query_terms)
    return {"query_terms": query_terms, "ss_query": query, "rationale": plan.get("rationale", "")}

# ── Step 4: SS completion ───────────────────────────────────────────

def complete_ss(llm: RiskLLM, chains: list, ss_evidence_raw: list) -> list:
    """Fill missing vuln/impact/KCN in each chain using SS evidence. Lock ref_hevi items."""
    need_ss = []
    for idx, c in enumerate(chains):
        has_v = bool(c.get("vulnerability"))
        has_i = bool(c.get("impact"))
        has_k = bool(c.get("key_control_nodes"))
        if not (has_v and has_i and has_k):
            need_ss.append((idx, c))

    if not need_ss:
        return chains

    system = (
        "You are a social-science risk completion agent. Fill ONLY empty vuln/impact/KCN slots. "
        "Existing items are LOCKED — do not modify. "
        "Vulnerability = system SENSITIVITY to the hazard (≤30 words). Operates at multiple scales: "
        "individual, institutional, infrastructural, societal. NOT generic 'lack of regulation'. "
        "Impact = negative CONSEQUENCES when hazard meets exposure under vulnerable conditions (≤30 words). "
        "Consequences ripple through coupled human-technical systems (e.g., biased decisions → eroded trust → reduced adoption). "
        "KCN = COPING CAPACITY at multiple levels (≤10 words each, name only): "
        "(a) Individual/autonomous: adversarial training, input validation, user vigilance. "
        "(b) Institutional/organizational: audit, human-in-the-loop, red-teaming, monitoring, anomaly detection, rate limiting. "
        "(c) Policy/societal: regulation, standards, ethical review, public code release. "
        "NOT the paper's method. NOT research proposals. "
        + HEVI_SHORT + " "
        "Return valid JSON ONLY, no other text."
    )
    user = {
        "task": "Fill empty vuln/impact/KCN slots. Lock existing items.",
        "ss_evidence": _compact_evidence(ss_evidence_raw),
        "chains_need_completion": [
            {
                "chain_index": idx,
                "scenario": c.get("scenario"),
                "issue": c.get("issue"),
                "hazard": c.get("hazard", []),
                "exposure": c.get("exposure", []),
                "vulnerability": c.get("vulnerability", []),
                "impact": c.get("impact", []),
                "key_control_nodes": c.get("key_control_nodes", []),
            }
            for idx, c in need_ss
        ],
        "rules": [
            "Non-empty slots are LOCKED — do not modify.",
            "Only fill slots that are empty.",
            "Merge overlapping items across chains into distinct concise entries.",
            "KCN: (a) technical controls or (b) author actions. NOT research proposals.",
        ],
        "output_schema": {
            "completions": [
                {
                    "chain_index": 0,
                    "vulnerability": ["filled items or empty"],
                    "impact": ["filled items or empty"],
                    "key_control_nodes": ["filled items or empty"],
                    "confidence": "low|medium|high",
                }
            ]
        },
    }
    result = llm.json_call(system, json.dumps(user, ensure_ascii=False))
    completions = _require_list(result, "completions", "SS completion output")
    for i, comp in enumerate(completions):
        idx = comp.get("chain_index", i)
        if idx < len(chains):
            if not chains[idx].get("vulnerability") and comp.get("vulnerability"):
                chains[idx]["vulnerability"] = _require_list(comp, "vulnerability", f"SS completion[{i}] vuln")
            if not chains[idx].get("impact") and comp.get("impact"):
                chains[idx]["impact"] = _require_list(comp, "impact", f"SS completion[{i}] impact")
            if not chains[idx].get("key_control_nodes") and comp.get("key_control_nodes"):
                chains[idx]["key_control_nodes"] = _require_list(comp, "key_control_nodes", f"SS completion[{i}] KCN")
    return chains

# ── Step 5: DR synthesis ────────────────────────────────────────────

def synthesize_dr_v2(llm: RiskLLM, chains: list) -> list:
    """Splice CS+SS segments at each Nexus into a dose_response sentence."""
    system = (
        "You are a Dose-Response synthesizer. Splice CS+SS segments at each place-based Nexus "
        "into ONE dose_response sentence (≤40 words). "
        "DR = how hazard scale/intensity translates into impact magnitude through the coupled system. "
        "It is the CAUSAL TRANSLATION: more/faster/wider deployment → more/worse harm. "
        "Acknowledge coupled system feedback where present (e.g., response in technical subsystem → "
        "affects human subsystem). Do NOT invent new content. "
        "Return valid JSON ONLY, no other text."
    )
    user = {
        "task": "Splice each chain into a dose_response sentence.",
        "chains": [
            {
                "chain_index": idx,
                "scenario": c.get("scenario"),
                "issue": c.get("issue"),
                "hazard": c.get("hazard", []),
                "exposure": c.get("exposure", []),
                "vulnerability": c.get("vulnerability", []),
                "impact": c.get("impact", []),
                "key_control_nodes": c.get("key_control_nodes", []),
            }
            for idx, c in enumerate(chains)
        ],
        "output_schema": {
            "completions": [
                {
                    "chain_index": 0,
                    "dose_response": ["ONE sentence ≤40 words"],
                }
            ]
        },
    }
    result = llm.json_call(system, json.dumps(user, ensure_ascii=False))
    completions = _require_list(result, "completions", "DR synthesis output")
    for i, comp in enumerate(completions):
        idx = comp.get("chain_index", i)
        if idx < len(chains):
            chains[idx]["dose_response"] = _require_list(comp, "dose_response", f"DR synthesis[{i}] dose_response")
    return chains

# ── Build CS query from paper + Nexus ───────────────────────────────

def build_cs_query_v2(llm: RiskLLM, paper: dict, chains: list) -> str:
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    system = "You are a CS literature retrieval query builder. Return valid JSON ONLY, no other text."
    user = {
        "task": "Build a CS retrieval query from paper metadata and risk chain context.",
        "title": title,
        "abstract": abstract[:500],
        "nexus": [
            {"scenario": c.get("scenario", ""), "issue": c.get("issue", "")}
            for c in chains
        ],
        "query_rules": [
            "CRITICAL: 'query' MUST be a JSON array of strings, e.g. [\"term one\", \"term two\"].",
            "Use method names, task names, mechanisms, models, datasets from the paper.",
            "Include domain/application terms from the Nexus (scenario, issue) — the risk context matters for retrieval.",
            "Include 2-3 synonyms for core technical concepts.",
            "Use 12-20 concise terms.",
        ],
        "output_schema": {
            "query": ["array of strings"],
        },
    }
    result = llm.json_call(system, json.dumps(user, ensure_ascii=False))
    query_terms = _require_list(result, "query", "CS query build")
    terms = [clean_text(t) for t in query_terms if clean_text(t)]
    if not terms:
        raise ValueError("CS query build returned empty query terms")
    return " ".join(terms)

# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="HEVI v2 — slot-aware completion pipeline")
    parser.add_argument("--icml-dir", default="outputs/hevi_workflow/hevi_icml_deepseek-v4-pro")
    parser.add_argument("--quality-report", default="")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--paper-ids", default="")
    parser.add_argument("--cs-index", default="indices/cs")
    parser.add_argument("--ss-index", default="indices/ss")
    parser.add_argument("--top-k-cs", type=int, default=5)
    parser.add_argument("--top-k-ss", type=int, default=5)
    parser.add_argument("--llm-name", default="deepseek-v4-pro")
    parser.add_argument("--llm-base-url", default="https://www.highland-api.top/v1")
    parser.add_argument("--max-new-tokens", type=int, default=384000)
    parser.add_argument("--llm-timeout", type=int, default=120)
    parser.add_argument("--llm-retries", type=int, default=10)
    parser.add_argument("--embedding-name", default="text-embedding-3-large")
    parser.add_argument("--embedding-base-url", default="https://www.highland-api.top/v1")
    parser.add_argument("--group", type=int, default=0, help="Group number (1-6). Auto-sets icml-dir and output-dir.")
    parser.add_argument("--output-dir", default="outputs/hevi_workflow_v2")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.group and 1 <= args.group <= 6:
        args.icml_dir = f"{args.icml_dir}/group_{args.group}"
        args.quality_report = f"{args.icml_dir}/quality_report.json"
        args.output_dir = f"{args.output_dir}/group_{args.group}"

    if args.max_new_tokens == 0:
        args.max_new_tokens = 384000 if "deepseek" in args.llm_name.lower() else 128000

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    for n in ("httpx", "httpcore", "openai", "openai._base_client"):
        logging.getLogger(n).setLevel(logging.WARNING)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    load_project_api_key()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    llm = RiskLLM(
        save_dir=str(out_dir),
        llm_name=args.llm_name,
        llm_base_url=args.llm_base_url,
        max_new_tokens=args.max_new_tokens,
        request_timeout=args.llm_timeout,
        max_retries=args.llm_retries,
    )
    cs_retriever = RiskRetriever(args.cs_index, "cs", args.llm_name, args.llm_base_url,
                                  args.embedding_name, args.embedding_base_url)
    ss_retriever = RiskRetriever(args.ss_index, "ss", args.llm_name, args.llm_base_url,
                                  args.embedding_name, args.embedding_base_url)

    # Load papers
    icml_dir = Path(args.icml_dir)
    papers = []
    for f in sorted(icml_dir.glob("group_*/*.json")):
        if not f.stem.startswith("icml_"):
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        papers.append({
            "paper_id": f.stem,
            "title": data.get("title", ""),
            "abstract": data.get("abstract", ""),
            "impact": data.get("impact", ""),
            "query": data.get("query", ""),
            "query_terms": data.get("query_terms", []),
            "ref_hevi": data.get("ref_hevi", {}),
        })

    # Filter by quality report if given
    if args.quality_report:
        qr_path = Path(args.quality_report)
        if qr_path.exists():
            qr = json.loads(qr_path.read_text(encoding="utf-8"))
            keep_ids = {pid for pid, a in qr.get("papers", {}).items() if a.get("verdict") == "keep"}
            papers = [p for p in papers if p["paper_id"] in keep_ids]
            print(f"Filtered to {len(papers)} papers with verdict=keep", flush=True)

    if args.paper_ids:
        ids = set(args.paper_ids.split(","))
        papers = [p for p in papers if p["paper_id"] in ids]
    elif args.limit > 0 and len(papers) > args.limit:
        papers = papers[:args.limit]

    # Resume
    completed = set()
    if args.resume:
        for d in out_dir.iterdir():
            if d.is_dir() and (d / "completed_hevi.json").exists():
                completed.add(d.name)

    total = len(papers)
    print(f"Loaded {total} papers", flush=True)
    written = 0
    skipped = 0

    for idx, paper in enumerate(papers, start=1):
        pid = paper["paper_id"]
        paper_dir = out_dir / pid

        if pid in completed:
            skipped += 1
            print(f"[{idx}/{total}] skip {pid}: already completed", flush=True)
            continue

        print(f"[{idx}/{total}] {pid}", flush=True)

        try:
            # ── Step 1: Nexus extraction ──────────────────────────
            print("  [1/5] Nexus extraction", flush=True)
            chains = extract_nexus(llm, paper)
            n_chains = len(chains)
            print(f"  [1/5] {n_chains} Nexus chain(s)", flush=True)

            # ── Step 2: CS completion ────────────────────────────
            print("  [2/5] CS completion", flush=True)
            cs_query = build_cs_query_v2(llm, paper, chains)
            cs_evidence_raw = cs_retriever.retrieve(cs_query, top_k=args.top_k_cs) if cs_query else []
            chains = complete_cs(llm, paper, chains, cs_evidence_raw, cs_query)
            cs_evidence_judged = strip_internal_ids(
                judge_evidence(llm, "cs", cs_query, compact_cs_output({"hazard": [], "nexus_candidates": chains}), cs_evidence_raw)
            ) if cs_evidence_raw else []

            # ── Step 3: SS query build ────────────────────────────
            print("  [3/5] SS query build", flush=True)
            ss_plan = build_ss_query_v2(llm, chains)
            ss_query = ss_plan.get("ss_query", "")

            # ── Step 4: SS completion ────────────────────────────
            print("  [4/5] SS completion", flush=True)
            ss_evidence_raw = ss_retriever.retrieve(ss_query, top_k=args.top_k_ss) if ss_query else []
            chains = complete_ss(llm, chains, ss_evidence_raw)
            ss_evidence_judged = strip_internal_ids(
                judge_evidence(llm, "ss", ss_query, compact_ss_output({"nexus_responses": chains}), ss_evidence_raw)
            ) if ss_evidence_raw else []

            # ── Step 5: DR synthesis ──────────────────────────────
            print("  [5/5] DR synthesis", flush=True)
            chains = synthesize_dr_v2(llm, chains)

            # ── Tag sources ───────────────────────────────────────
            for c in chains:
                c.setdefault("dose_response", [])
                c["_sources"] = {
                    "hazard": "ref_hevi" if paper["ref_hevi"].get("hazard") else "cs_agent",
                    "exposure": "ref_hevi" if paper["ref_hevi"].get("exposure") else "cs_agent",
                    "dose_response": "dr_synthesis",
                    "vulnerability": "ref_hevi" if paper["ref_hevi"].get("vulnerability") else "ss_agent",
                    "impact": "ref_hevi" if paper["ref_hevi"].get("impact") else "ss_agent",
                    "key_control_nodes": "ref_hevi" if paper["ref_hevi"].get("key_control_nodes") else "ss_agent",
                }

            # ── Save ──────────────────────────────────────────────
            output = {
                "paper_id": pid,
                "title": paper["title"],
                "abstract": paper["abstract"],
                "impact": paper["impact"],
                "ref_hevi": paper["ref_hevi"],
                "chains": chains,
                "cs_evidence": cs_evidence_judged,
                "ss_evidence": ss_evidence_judged,
                "stats": {
                    "n_chains": n_chains,
                    "cs_filled": sum(1 for c in chains if c["_sources"]["hazard"] == "cs_agent" or c["_sources"]["exposure"] == "cs_agent"),
                    "ss_filled": sum(1 for c in chains if c["_sources"]["vulnerability"] == "ss_agent" or c["_sources"]["impact"] == "ss_agent" or c["_sources"]["key_control_nodes"] == "ss_agent"),
                },
            }
            paper_dir.mkdir(parents=True, exist_ok=True)
            write_json(paper_dir / "completed_hevi.json", output)
            written += 1
            print(f"  [done] {pid}", flush=True)

        except ValueError as exc:
            # Print raw response snippet for debugging
            err_msg = str(exc)
            if "unparseable response" in err_msg:
                # The snippet is already embedded in the exception message
                print(f"  [{idx}/{total}] {pid} FAILED: {err_msg[:500]}", flush=True)
            else:
                print(f"  [{idx}/{total}] {pid} FAILED: {exc}", flush=True)
            continue
        except Exception as exc:
            print(f"  [{idx}/{total}] {pid} FAILED: {exc}", flush=True)
            continue

    print(f"\nDone. written={written} skipped={skipped} failed={total - written - skipped}", flush=True)
    print(f"Output: {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
