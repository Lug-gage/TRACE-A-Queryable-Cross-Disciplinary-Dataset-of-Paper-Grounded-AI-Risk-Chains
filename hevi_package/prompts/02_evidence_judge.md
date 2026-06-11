# 证据裁决 (CS+SS 共用)

**Stage**: Stage 2 & 3
**Location**: `hit_report.py → judge_hits_with_llm`

```python
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

```
