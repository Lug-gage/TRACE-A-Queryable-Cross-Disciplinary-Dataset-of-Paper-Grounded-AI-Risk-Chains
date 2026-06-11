from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List
from urllib import error, request

from src.hipporag.hevi_workflow.utils import extract_json_object


def _compact_evidence(evidence: List[Dict[str, Any]], max_chars: int = 900) -> List[Dict[str, Any]]:
    compact = []
    for item in evidence:
        abstract = str(item.get("abstract") or item.get("doc_text") or "")
        compact.append(
            {
                "evidence_id": item.get("evidence_id"),
                "paper_id": item.get("paper_id"),
                "title": item.get("title"),
                "score": item.get("score"),
                "abstract": abstract[:max_chars],
            }
        )
    return compact


class RiskLLM:
    def __init__(
        self,
        save_dir: str,
        llm_name: str,
        llm_base_url: str | None,
        max_new_tokens: int,
        temperature: float = 0.0,
        request_timeout: int = 1200,
        max_retries: int = 3,
    ) -> None:
        cache_dir = Path(save_dir) / "llm_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache_dir / "hevi_workflow_cache.sqlite"
        self.llm_name = llm_name
        self.llm_base_url = (llm_base_url or "https://api.openai.com/v1").rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.max_retries = max_retries

    def json_call(self, system: str, user: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_error = None
        last_message = "<no response>"
        for attempt in range(1, self.max_retries + 1):
            message, metadata, cache_hit = self._chat(messages)
            last_message = message
            try:
                parsed = extract_json_object(message)
                parsed["_metadata"] = metadata
                parsed["_cache_hit"] = cache_hit
                return parsed
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                wait_seconds = min(5 * attempt, 15)
                print(
                    f"[warn] LLM JSON parse failed; retry {attempt}/{self.max_retries} after {wait_seconds}s: {exc}",
                    flush=True,
                )
                time.sleep(wait_seconds)
        snippet = last_message[:300]
        raise ValueError(f"LLM returned unparseable response after {self.max_retries} attempts: {snippet}") from last_error

    def _chat(self, messages: List[Dict[str, str]]) -> tuple[str, Dict[str, Any], bool]:
        key_data = {
            "messages": messages,
            "model": self.llm_name,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }
        key = hashlib.sha256(json.dumps(key_data, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, message TEXT, metadata TEXT)"
            )
            row = conn.execute("SELECT message, metadata FROM cache WHERE key = ?", (key,)).fetchone()
            if row:
                return row[0], json.loads(row[1]), True

        payload = {
            "model": self.llm_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
        }
        req = request.Request(
            f"{self.llm_base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with request.urlopen(req, timeout=self.request_timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                break
            except (TimeoutError, socket.timeout, error.URLError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
                wait_seconds = min(10 * attempt, 30)
                print(
                    f"[warn] LLM request failed or timed out; retry {attempt}/{self.max_retries} after {wait_seconds}s",
                    flush=True,
                )
                time.sleep(wait_seconds)
        else:
            raise RuntimeError(f"LLM request failed: {last_error}")

        message = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        metadata = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "finish_reason": data["choices"][0].get("finish_reason"),
        }
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                (key, message, json.dumps(metadata)),
            )
        return message, metadata, False


class CSAgent:
    """CS Agent: proposes Hazard -> Exposure from CS evidence, critiques SS claims, revises."""

    def __init__(self, llm: RiskLLM) -> None:
        self.llm = llm

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
            "Propose Nexus candidates as (scenario, issue), but keep application scenarios anchored in the retrieval query or CS evidence. "
            "Do not introduce new application domains beyond the query/evidence. "
            "Do not frame a Nexus as malicious/adversarial/attacker-driven unless the query or CS evidence explicitly "
            "mentions attack, adversary, malicious use, misuse, security, poisoning, or evasion. "
            "If no concrete application domain is stated, use a neutral generic scenario such as deployment of the retrieved capability "
            "in the stated task or system type. "
            "Your self_score must reflect: (a) how well your hazard is grounded in CS evidence, "
            "(b) how well your exposure scenarios are anchored in the query/evidence, "
            "(c) how complete your coverage of the technical risk surface is. "
            "Be honest — score below 0.5 if evidence is thin. "
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
                        "hazard": "technical hazard introduced or sharpened by the paper",
                        "confidence": "low|medium|high",
                    }
                ],
                "nexus_candidates": [
                    {
                        "scenario": "deployment or encounter setting anchored in cs_query or cs_evidence only",
                        "issue": "risk issue created by the technical mechanism",
                        "exposure": "who or what could be exposed in this scenario",
                        "confidence": "low|medium|high",
                    }
                ],
            },
        }
        return self.llm.json_call(system, json.dumps(user, ensure_ascii=False))

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

    def revise_proposal(
        self,
        cs_proposal: Dict[str, Any],
        ss_critique: Dict[str, Any],
        cs_evidence: List[Dict[str, Any]],
        cs_query: str,
    ) -> Dict[str, Any]:
        system = (
            "You are the CS Agent in a symmetric bilateral consensus protocol. "
            "Your role in this phase is to revise your CS proposal based on the SS Agent's critique. "
            "For each critique point: either accept the suggestion and revise your proposal accordingly, "
            "or explain why you maintain your original position (only if you have strong technical evidence). "
            "Do NOT simply dismiss valid social science concerns — if the SS Agent points out a missing stakeholder "
            "or an implausible exposure pathway, adjust your nexus_candidates to address it. "
            "Update your self_score to reflect the revised proposal quality. "
            "Return valid JSON only."
        )
        user = {
            "task": "Revise the CS proposal absorbing the SS Agent's critique.",
            "original_proposal": {
                "hazard": cs_proposal.get("hazard"),
                "nexus_candidates": cs_proposal.get("nexus_candidates", []),
                "self_score": cs_proposal.get("self_score"),
            },
            "ss_critique": ss_critique.get("critiques", []),
            "ss_overall": ss_critique.get("overall", ""),
            "cs_evidence": _compact_evidence(cs_evidence),
            "cs_query": cs_query,
            "output_schema": {
                "agent": "cs",
                "self_score": "0.0-1.0 updated confidence after revision",
                "revision_notes": "one sentence summarizing what was changed and why",
                "hazard": [
                    {
                        "hazard": "technical hazard description",
                        "confidence": "low|medium|high",
                    }
                ],
                "nexus_candidates": [
                    {
                        "scenario": "deployment or encounter setting",
                        "issue": "risk issue created by the technical mechanism",
                        "exposure": "who or what could be exposed",
                        "confidence": "low|medium|high",
                    }
                ],
            },
        }
        return self.llm.json_call(system, json.dumps(user, ensure_ascii=False))



class SSAgent:
    """SS Agent: responds with Vulnerability -> Impact -> KCN, critiques CS claims, revises."""

    def __init__(self, llm: RiskLLM) -> None:
        self.llm = llm

    def respond_bilateral(
        self,
        cs_proposal: Dict[str, Any],
        ss_evidence: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        system = (
            "You are the Social Science Agent in a symmetric bilateral consensus protocol. "
            "Respond to the CS Agent's Nexus candidates. You may accept, reject, or revise each Nexus. "
            "You are responsible only for Nexus -> Vulnerability -> Impact and control nodes. "
            "Do not produce Exposure; Exposure belongs to the CS Agent's Hazard -> Exposure segment. "
            "Use retrieved SS evidence for social mechanisms and impacts. "
            "Your self_score must reflect: (a) how well your vulnerability claims are grounded in SS evidence, "
            "(b) how plausible your social mechanisms are, (c) how complete your impact coverage is. "
            "Be honest — score below 0.5 if evidence is thin. "
            "key_control_nodes include TWO categories: "
            "(a) technical controls: adversarial training, anomaly detection, rate limiting, human-in-the-loop, auditing, differential privacy, etc. "
            "(b) author actions: releasing code or datasets, publishing benchmarks, inviting red-teaming, proposing evaluation protocols, building tools for defenders. "
            "If the CS proposal or its context mentions code release, tool building, or evaluation resources, include them as KCN. "
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
                        "scenario": "accepted or revised scenario",
                        "issue": "accepted or revised issue",
                        "revision_reason": "empty if accepted",
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

    def critique_cs(
        self,
        cs_proposal: Dict[str, Any],
        ss_evidence: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        system = (
            "You are the Social Science Agent in a symmetric bilateral consensus protocol. "
            "Your role in this phase is to critique the CS Agent's nexus_candidates from a social science standpoint. "
            "For each CS nexus_candidate, assess whether the exposure scenarios and risk issues are socially plausible. "
            "Point out: missing stakeholder groups, overlooked vulnerability dimensions, implausible social mechanisms, "
            "or impact pathways that social science evidence suggests would differ. "
            "Be specific and constructive — suggest concrete additions or revisions. "
            "If a candidate IS socially well-grounded, say so clearly. "
            "Return valid JSON only."
        )
        user = {
            "task": "Critique each CS nexus_candidate for social plausibility and completeness.",
            "cs_proposal": {
                "hazard": cs_proposal.get("hazard"),
                "nexus_candidates": cs_proposal.get("nexus_candidates", []),
            },
            "ss_evidence": _compact_evidence(ss_evidence),
            "output_schema": {
                "agent": "ss",
                "critiques": [
                    {
                        "nexus_index": 0,
                        "scenario": "scenario being critiqued",
                        "assessment": "supported|speculative|unsupported",
                        "social_critique": "specific social science reasoning — what is plausible or missing",
                        "suggested_change": "concrete revision suggestion, or null if fully supported",
                    }
                ],
                "overall": "one-sentence summary of SS-side social science assessment",
            },
        }
        return self.llm.json_call(system, json.dumps(user, ensure_ascii=False))

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
            "key_control_nodes include TWO categories: "
            "(a) technical controls: adversarial training, anomaly detection, rate limiting, human-in-the-loop, auditing, differential privacy, etc. "
            "(b) author actions: releasing code or datasets, publishing benchmarks, inviting red-teaming, proposing evaluation protocols, building tools for defenders. "
            "If the CS proposal or critique mentions code release, tool building, or evaluation resources, include them as KCN. "
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
