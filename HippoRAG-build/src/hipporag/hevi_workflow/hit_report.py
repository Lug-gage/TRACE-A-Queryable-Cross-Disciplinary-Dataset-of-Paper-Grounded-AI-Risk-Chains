from __future__ import annotations

import json
import re
from typing import Any, Dict, List


STOPWORDS = {
    "about", "above", "after", "again", "against", "also", "although", "among",
    "and", "are", "because", "been", "being", "between", "both", "can", "could",
    "abstract", "does", "during", "each", "from", "have", "having", "here", "into", "its",
    "itself", "lead", "leading", "more", "most", "much", "only", "other", "our",
    "over", "paper", "papers", "present", "propose", "proposed", "provide",
    "query", "scenario", "show", "shows", "such", "than", "that", "the", "their", "them", "then",
    "there", "these", "this", "those", "through", "title", "using", "were",
    "when", "where", "which", "while", "with", "within", "without", "would", "issue",
}


def keywords(text: str, max_terms: int = 80) -> List[str]:
    terms = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", text.lower())
    seen = set()
    result = []
    for term in terms:
        if term in STOPWORDS or term in seen:
            continue
        seen.add(term)
        result.append(term)
        if len(result) >= max_terms:
            break
    return result


def compact_cs_output(cs_proposal: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "hazard": cs_proposal.get("hazard"),
        "exposure": [
            {
                "scenario": item.get("scenario"),
                "issue": item.get("issue"),
                "text": item.get("exposure") or item.get("scenario"),
                "confidence": item.get("confidence"),
            }
            for item in cs_proposal.get("nexus_candidates", []) or []
        ],
        "nexus_candidates": [
            {
                "scenario": item.get("scenario"),
                "issue": item.get("issue"),
                "confidence": item.get("confidence"),
            }
            for item in cs_proposal.get("nexus_candidates", []) or []
        ],
    }


def compact_ss_output(ss_response: Dict[str, Any]) -> Dict[str, Any]:
    by_slot = {
        "vulnerability": [],
        "impact": [],
        "key_control_node": [],
    }
    for item in ss_response.get("nexus_responses", []) or []:
        scenario = item.get("scenario")
        issue = item.get("issue")
        prefix = {}
        if scenario:
            prefix["scenario"] = scenario
        if issue:
            prefix["issue"] = issue
        for value in item.get("vulnerability", []) or []:
            by_slot["vulnerability"].append({**prefix, "text": value})
        for value in item.get("impact", []) or []:
            by_slot["impact"].append({**prefix, "text": value})
        for value in item.get("key_control_nodes", []) or []:
            by_slot["key_control_node"].append({**prefix, "text": value})
    return {
        "by_slot": by_slot,
    }


def default_judgment(reason: str = "LLM judge did not return a usable judgment.") -> Dict[str, Any]:
    return {"label": "low", "reason": reason}


def compact_hit(evidence: Dict[str, Any], judgment: Dict[str, Any] | None = None) -> Dict[str, Any]:
    title = str(evidence.get("title") or "")
    abstract = str(evidence.get("abstract") or evidence.get("doc_text") or "")
    judgment = judgment or {}
    return {
        "title": title,
        "abstract": abstract,
        "retrieval_score": evidence.get("score"),
        "evidence_relevance": judgment.get("evidence_relevance") or default_judgment(),
        "evidence_support": judgment.get("evidence_support") or default_judgment(),
        "hit_points": judgment.get("hit_points", []),
    }


def judge_hits_with_llm(
    llm: Any,
    side: str,
    query: str,
    agent_output: Dict[str, Any],
    evidence: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not evidence:
        return []
    system = (
        "You are an evidence judge for a CS/social-science risk workflow. "
        "Do not use keyword overlap as the criterion. Judge semantic relevance and support. "
        "Return valid JSON only."
    )
    user = {
        "task": "Judge each retrieved evidence paper.",
        "side": side,
        "query": query,
        "agent_output": agent_output,
        "rubric": {
            "evidence_relevance": {
                "high": "The paper directly studies the core technology, scenario, mechanism, or risk requested by the query.",
                "medium": "The paper studies an adjacent mechanism, task, setting, or risk that can reasonably inform the query.",
                "low": "The paper is only broadly related or does not meaningfully answer the query.",
            },
            "evidence_support": {
                "high": "The paper directly supports a specific claim in the agent output.",
                "medium": "The paper indirectly supports the output through an analogous mechanism, setting, or background evidence.",
                "low": "The paper does not support the output beyond broad topical similarity.",
            },
        },
        "evidence": [
            {
                "index": idx,
                "title": item.get("title"),
                "abstract": item.get("abstract") or item.get("doc_text", ""),
            }
            for idx, item in enumerate(evidence)
        ],
        "output_schema": {
            "judgments": [
                {
                    "index": 0,
                    "evidence_relevance": {
                        "label": "high|medium|low",
                        "reason": "brief semantic reason",
                    },
                    "evidence_support": {
                        "label": "high|medium|low",
                        "reason": "brief semantic reason",
                    },
                    "hit_points": ["short supporting point from title/abstract, paraphrase or brief quote"],
                }
            ]
        },
    }
    judged = llm.json_call(system, json.dumps(user, ensure_ascii=False))
    judgments_by_index = {
        int(item.get("index")): item
        for item in judged.get("judgments", []) or []
        if str(item.get("index", "")).isdigit()
    }
    return [judgments_by_index.get(idx, {}) for idx in range(len(evidence))]
