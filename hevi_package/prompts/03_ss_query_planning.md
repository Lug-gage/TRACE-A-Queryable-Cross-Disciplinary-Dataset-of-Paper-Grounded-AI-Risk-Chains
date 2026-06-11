# SS 检索查询规划

**Stage**: Stage 3
**Location**: `pipeline.py → build_ss_queries`

```python
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

```
