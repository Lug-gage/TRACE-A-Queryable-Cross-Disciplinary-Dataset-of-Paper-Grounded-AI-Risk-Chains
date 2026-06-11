from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.hipporag.hevi_workflow.utils import parse_paper_doc
from src.hipporag.utils.config_utils import BaseConfig


def _scores_to_list(scores) -> List[float]:
    if scores is None:
        return []
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    return [float(score) for score in scores]


@dataclass
class Evidence:
    evidence_id: str
    source: str
    paper_id: str
    title: str
    abstract: str
    score: float | None
    doc_text: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "source": self.source,
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "score": self.score,
            "doc_text": self.doc_text,
        }


class HippoRAGRetriever:
    def __init__(
        self,
        index_dir: str,
        source: str,
        llm_name: str,
        llm_base_url: str | None,
        embedding_name: str,
        embedding_base_url: str | None,
        openie_mode: str,
        rerank_dspy_file_path: str | None,
    ) -> None:
        from src.hipporag.HippoRAG import HippoRAG

        self.source = source
        config = BaseConfig(
            save_dir=index_dir,
            llm_name=llm_name,
            llm_base_url=llm_base_url,
            embedding_model_name=embedding_name,
            embedding_base_url=embedding_base_url,
            openie_mode=openie_mode,
            rerank_dspy_file_path=rerank_dspy_file_path,
        )
        self.rag = HippoRAG(global_config=config)

    def retrieve(self, query: str, top_k: int) -> List[Evidence]:
        solution = self.rag.retrieve([query], num_to_retrieve=top_k)[0]
        scores = _scores_to_list(solution.doc_scores)
        if scores:
            s_min = min(scores)
            s_max = max(scores)
            if s_max > s_min:
                scores = [(s - s_min) / (s_max - s_min) for s in scores]
            else:
                scores = [1.0 for _ in scores]
        evidence: List[Evidence] = []
        for idx, doc in enumerate(solution.docs[:top_k]):
            parsed = parse_paper_doc(doc)
            evidence.append(
                Evidence(
                    evidence_id=f"{self.source}_e{idx + 1}",
                    source=self.source,
                    paper_id=parsed["paper_id"],
                    title=parsed["title"],
                    abstract=parsed["abstract"],
                    score=scores[idx] if idx < len(scores) else None,
                    doc_text=parsed["doc_text"],
                )
            )
        return evidence


class RiskRetriever:
    def __init__(
        self,
        index_dir: str,
        source: str,
        llm_name: str,
        llm_base_url: str | None,
        embedding_name: str,
        embedding_base_url: str | None,
        openie_mode: str = "online",
        rerank_dspy_file_path: str | None = None,
    ) -> None:
        self.impl = HippoRAGRetriever(
            index_dir=index_dir,
            source=source,
            llm_name=llm_name,
            llm_base_url=llm_base_url,
            embedding_name=embedding_name,
            embedding_base_url=embedding_base_url,
            openie_mode=openie_mode,
            rerank_dspy_file_path=rerank_dspy_file_path,
        )

    def retrieve(self, query: str, top_k: int) -> List[Dict[str, object]]:
        last_error = None
        for attempt in range(1, 4):
            try:
                return [item.to_dict() for item in self.impl.retrieve(query, top_k)]
            except Exception as exc:
                last_error = exc
                if attempt == 3:
                    break
                wait_seconds = 5 * attempt
                print(
                    f"[warn] retrieval failed on attempt {attempt}/3: {exc}. "
                    f"Retrying in {wait_seconds}s..."
                )
                time.sleep(wait_seconds)
        raise last_error


def dedupe_evidence(evidence_groups: List[List[Dict[str, object]]]) -> List[Dict[str, object]]:
    seen = set()
    merged: List[Dict[str, object]] = []
    for group in evidence_groups:
        for item in group:
            key = item.get("paper_id") or item.get("doc_text")
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    for idx, item in enumerate(merged, start=1):
        item["evidence_id"] = f"{item.get('source', 'e')}_e{idx}"
    return merged
