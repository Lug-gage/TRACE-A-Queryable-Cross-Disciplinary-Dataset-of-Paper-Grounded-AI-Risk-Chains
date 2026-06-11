#!/usr/bin/env python3
"""
完整建库脚本：OpenIE JSON → Parquet + 向量库 + 社区报告

用法:
    python build_index.py cs           # CS 数据集（完整）
    python build_index.py ss           # SS 数据集（完整）
    python build_index.py all          # 两个都跑
    python build_index.py ss --fast    # SS 快速建库（跳过 LLM 社区报告，~2分钟）

耗时预估 (API 稳定情况下):
    完整: SS ~12min | CS ~15min（含 LLM 社区报告 + Embedding）
    快速: ~2min（仅 Embedding，社区报告用占位文本）

注意: --fast 模式下 global/drift 查询不可用，local/basic 正常。
"""

import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import lancedb as ldb
import numpy as np
import pandas as pd
import requests
import urllib3
from json_repair import repair_json

from graphrag.index.operations.cluster_graph import cluster_graph

# ===================== 配置 =====================

API_KEY = "sk-IMpKWiHTwOURKtKoRqKg7A7hgCmEIUIaukfbXin1OTxn7cE1"
API_BASE = "https://www.highland-api.top/v1"
LLM_MODEL = "deepseek-v4-pro"
EMBEDDING_MODEL = "text-embedding-3-large"

BASE_DIR = Path(__file__).parent / "indices"
MAX_CLUSTER_SIZE = 50
LLM_CONCURRENCY = 5       # LLM 并发
LLM_RETRY = 5
LLM_RETRY_DELAY = 120     # 重试间隔秒
LLM_CALL_INTERVAL = 0     # 调用间隔秒（设为0，API稳定时不需要）

urllib3.disable_warnings()

# ===================== 工具函数 =====================

def hash_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def api_request(endpoint: str, payload: dict, timeout: int = 180) -> dict:
    """同步 API 请求（不做重试，重试由上层处理）."""
    resp = requests.post(
        f"{API_BASE}{endpoint}",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
        verify=False,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json()

# ===================== 步骤 1-3: JSON → DataFrame =====================

def read_openie_json(name: str) -> dict:
    with open(BASE_DIR / name / "openie_results_ner_deepseek-v4-pro.json") as f:
        return json.load(f)


def build_entity_relationship_df(data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    docs = data["docs"]
    entity_info: dict[str, dict] = {}
    triples_set: dict[str, dict] = {}

    for doc in docs:
        cid = doc["idx"]
        src = doc["passage"]["source"]
        for e in doc["extracted_entities"]:
            entity_info.setdefault(e, {"title": e, "text_unit_ids": set(), "source": src})
            entity_info[e]["text_unit_ids"].add(cid)
        for t in doc["extracted_triples"]:
            if len(t) != 3:
                continue
            s, p, o = t
            for x in [s, o]:
                entity_info.setdefault(x, {"title": x, "text_unit_ids": set(), "source": src})
            key = f"{s}|||{o}|||{p}"
            if key not in triples_set:
                triples_set[key] = {"source": s, "target": o, "description": p, "text_unit_ids": set()}
            triples_set[key]["text_unit_ids"].add(cid)

    # entities
    rows = []
    deg = {}
    for t in triples_set.values():
        deg[t["source"]] = deg.get(t["source"], 0) + 1
        deg[t["target"]] = deg.get(t["target"], 0) + 1
    for idx, (title, info) in enumerate(entity_info.items()):
        rows.append({
            "id": hash_id(title), "human_readable_id": idx, "title": title,
            "type": info.get("source", ""), "description": title,
            "text_unit_ids": list(info["text_unit_ids"]),
            "frequency": len(info["text_unit_ids"]), "degree": deg.get(title, 0),
        })
    entities = pd.DataFrame(rows)

    # relationships
    rel_rows = []
    for idx, (key, info) in enumerate(triples_set.items()):
        rel_rows.append({
            "id": hash_id(key), "human_readable_id": idx,
            "source": info["source"], "target": info["target"],
            "description": info["description"], "weight": 1.0,
            "combined_degree": deg.get(info["source"], 0) + deg.get(info["target"], 0),
            "text_unit_ids": list(info["text_unit_ids"]),
        })
    relationships = pd.DataFrame(rel_rows)
    return entities, relationships


def build_documents_text_units_df(data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    docs = data["docs"]
    papers: dict[str, list] = {}
    for d in docs:
        papers.setdefault(d["passage"]["paper_id"], []).append(d)

    docs_out, tu_out = [], []
    for pid, chunks in papers.items():
        d_id = hash_id(pid)
        docs_out.append({
            "id": d_id, "human_readable_id": len(docs_out),
            "title": chunks[0]["passage"]["title"],
            "text": chunks[0]["passage"]["abstract"],
            "text_unit_ids": [c["idx"] for c in chunks],
            "creation_date": datetime.now(timezone.utc).date().isoformat(),
            "raw_data": None,
        })
        for c in chunks:
            tu_out.append({
                "id": c["idx"], "human_readable_id": len(tu_out),
                "text": c["passage"]["abstract"],
                "n_tokens": len(c["passage"]["abstract"].split()),
                "document_id": d_id, "entity_ids": [], "relationship_ids": [], "covariate_ids": [],
            })
    return pd.DataFrame(docs_out), pd.DataFrame(tu_out)

# ===================== 步骤 4: 社区检测 =====================

def run_community_detection(entities: pd.DataFrame, relationships: pd.DataFrame) -> pd.DataFrame:
    edges = relationships[["source", "target", "weight"]].copy()
    clusters = cluster_graph(edges, MAX_CLUSTER_SIZE, use_lcc=False, seed=42)

    t2id = dict(zip(entities["title"], entities["id"]))
    comm = pd.DataFrame(
        clusters, columns=pd.Index(["level", "community", "parent", "title"])
    ).explode("title")
    comm["community"] = comm["community"].astype(int)

    # entity_ids per community
    em = comm[["community", "title"]].copy()
    em["entity_id"] = em["title"].map(t2id)
    eids = em.dropna(subset=["entity_id"]).groupby("community").agg(entity_ids=("entity_id", list)).reset_index()

    # intra-community relationships
    parts = []
    for lv in comm["level"].unique():
        lc = comm[comm["level"] == lv]
        ws = relationships.merge(lc, left_on="source", right_on="title", how="inner")
        wb = ws.merge(lc, left_on="target", right_on="title", how="inner")
        intra = wb[wb["community_x"] == wb["community_y"]]
        if intra.empty:
            continue
        g = (intra.explode("text_unit_ids")
             .groupby(["community_x", "parent_x"])
             .agg(relationship_ids=("id", list), text_unit_ids=("text_unit_ids", list))
             .reset_index())
        g["level"] = lv
        parts.append(g)

    if parts:
        grouped = pd.concat(parts, ignore_index=True).rename(
            columns={"community_x": "community", "parent_x": "parent"})
        grouped["relationship_ids"] = grouped["relationship_ids"].apply(lambda x: sorted(set(x)))
        grouped["text_unit_ids"] = grouped["text_unit_ids"].apply(lambda x: sorted(set(x)))
    else:
        grouped = pd.DataFrame(columns=["community", "parent", "relationship_ids", "text_unit_ids", "level"])

    result = grouped.merge(eids, on="community", how="inner")
    result["id"] = [str(uuid4()) for _ in range(len(result))]
    result["human_readable_id"] = result["community"]
    result["title"] = "Community " + result["community"].astype(str)
    result["parent"] = result["parent"].astype(int)

    pg = result.groupby("parent").agg(children=("community", "unique"))
    result = result.merge(pg, left_on="community", right_on="parent", how="left")
    result["children"] = result["children"].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else [])
    result["period"] = datetime.now(timezone.utc).date().isoformat()
    result["size"] = result["entity_ids"].apply(len)

    return result[["id", "human_readable_id", "community", "level", "parent", "children",
                    "title", "entity_ids", "relationship_ids", "text_unit_ids", "period", "size"]]

# ===================== 步骤 5a: 占位社区报告（快速模式） =====================

def build_placeholder_reports(communities: pd.DataFrame) -> pd.DataFrame:
    """生成占位社区报告，不调用 LLM。仅限 local/basic 查询使用。"""
    rows = []
    for _, c in communities.iterrows():
        rows.append({
            "id": hashlib.sha512(str(c["community"]).encode()).hexdigest()[:16],
            "human_readable_id": c["community"],
            "community": c["community"], "level": c["level"],
            "parent": c["parent"], "children": c.get("children", []),
            "title": f"Community {c['community']}",
            "summary": "Placeholder — fast build, no LLM report generated.",
            "full_content": json.dumps({"title": f"Community {c['community']}", "summary": "Placeholder"}, ensure_ascii=False),
            "rank": 1.0, "rating_explanation": "",
            "findings": [],
            "full_content_json": json.dumps({"title": f"Community {c['community']}", "summary": "Placeholder"}, ensure_ascii=False),
            "period": c.get("period", ""), "size": len(c.get("entity_ids", [])),
        })
    return pd.DataFrame(rows)


# ===================== 步骤 5b: LLM 社区报告 =====================

COMMUNITY_REPORT_PROMPT = """You are an AI assistant helping analyze an academic research knowledge graph.

Given a list of entities and their relationships within a community, write a comprehensive report about this community.

# Report Structure
- TITLE: A short, specific name summarizing the community's key research themes.
- SUMMARY: An executive summary (2-3 sentences) of the community's overall research focus and key findings.
- RATING: A float 1-10 representing the richness/importance of research findings in this community.
- RATING_EXPLANATION: One sentence explaining the rating.
- FINDINGS: 3-5 key insights, each with a short summary and a paragraph of explanation.

# Output Format
Return ONLY a JSON object (no markdown, no extra text):
{
    "title": "...",
    "summary": "...",
    "rating": 5.0,
    "rating_explanation": "...",
    "findings": [
        {"summary": "...", "explanation": "..."},
        {"summary": "...", "explanation": "..."}
    ]
}

# Community Data
{context}

Output:"""


def _build_community_context(row: pd.Series, entities: pd.DataFrame, relationships: pd.DataFrame) -> str:
    """构建社区上下文文本."""
    eids = row.get("entity_ids", [])
    rids = row.get("relationship_ids", [])

    comm_entities = entities[entities["id"].isin(eids)]
    comm_rels = relationships[relationships["id"].isin(rids)]

    lines = ["## Entities"]
    for _, e in comm_entities.head(30).iterrows():
        lines.append(f"- {e['title']}")

    lines.append("\n## Relationships")
    for _, r in comm_rels.head(60).iterrows():
        lines.append(f"- [{r['source']}] --({r['description']})--> [{r['target']}]")

    return "\n".join(lines)


def _sync_llm_report(community_id: int, context: str) -> dict:
    """单个社区报告同步调用."""
    payload = {"model": LLM_MODEL, "messages": [
        {"role": "user", "content": COMMUNITY_REPORT_PROMPT.replace("{context}", context)}
    ], "temperature": 0.3}

    for attempt in range(LLM_RETRY):
        try:
            if attempt == 0:
                time.sleep(LLM_CALL_INTERVAL)  # 首次调用前间隔，减轻 API 压力
            data = api_request("/chat/completions", payload, timeout=180)
            raw = data["choices"][0]["message"]["content"].strip()
            if not raw:
                raise ValueError("Empty response")
            # 清理 markdown
            for prefix in ["```json", "```"]:
                if raw.startswith(prefix):
                    raw = raw[len(prefix):]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
            # json_repair 修复截断/格式错误
            repaired = repair_json(raw)
            return json.loads(repaired)
        except Exception:
            if attempt < LLM_RETRY - 1:
                time.sleep(LLM_RETRY_DELAY)
            else:
                # 最后一次失败，返回占位报告
                return {"title": f"Community {community_id}",
                        "summary": "A community of research concepts.",
                        "rating": 1.0, "rating_explanation": "", "findings": []}


def build_community_reports(
    communities: pd.DataFrame, entities: pd.DataFrame, relationships: pd.DataFrame, name: str
) -> pd.DataFrame:
    """并发生成所有社区报告."""
    total = len(communities)
    print(f"  Generating {total} community reports (concurrency={LLM_CONCURRENCY})...")
    print(f"  Building contexts...", flush=True)

    # 预构建所有上下文
    tasks = []
    for i, (_, row) in enumerate(communities.iterrows()):
        ctx = _build_community_context(row, entities, relationships)
        tasks.append((row["community"], ctx))
        if (i + 1) % 100 == 0:
            print(f"  Contexts: {i + 1}/{total}", flush=True)
    print(f"  Calling LLM...", flush=True)

    results = {}
    done = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as pool:
        futures = {pool.submit(_sync_llm_report, cid, ctx): cid for cid, ctx in tasks}
        for future in as_completed(futures):
            cid = futures[future]
            try:
                results[cid] = future.result()
            except Exception:
                results[cid] = {"title": f"Community {cid}", "summary": "A community of research concepts.",
                                "rating": 1.0, "rating_explanation": "", "findings": []}
                failed += 1
            done += 1
            if done % 20 == 0 or done == total:
                print(f"  Community reports: {done}/{total}", flush=True)

    if failed > 0:
        print(f"  {failed} reports generated as placeholder (API failure)")

    # 转为 DataFrame
    rows = []
    for _, c in communities.iterrows():
        cid = c["community"]
        report = results.get(cid, {})
        rows.append({
            "id": hashlib.sha512(json.dumps(report).encode()).hexdigest()[:16],
            "human_readable_id": c["community"],
            "community": c["community"], "level": c["level"],
            "parent": c["parent"], "children": c.get("children", []),
            "title": report.get("title", c["title"]),
            "summary": report.get("summary", ""),
            "full_content": json.dumps(report, ensure_ascii=False),
            "rank": float(report.get("rating", 1.0)),
            "rating_explanation": report.get("rating_explanation", ""),
            "findings": report.get("findings", []),
            "full_content_json": json.dumps(report, ensure_ascii=False),
            "period": c.get("period", ""), "size": len(c.get("entity_ids", [])),
        })
    return pd.DataFrame(rows)

# ===================== 步骤 6: Embedding =====================

def generate_embeddings_sync(texts: list[str]) -> list[list[float]]:
    """批量生成 embedding."""
    all_embs = []
    total = len(texts)
    batch = 50

    for i in range(0, total, batch):
        chunk = texts[i:i + batch]
        data = api_request("/embeddings", {"model": EMBEDDING_MODEL, "input": chunk}, timeout=120)
        all_embs.extend([item["embedding"] for item in data["data"]])
        done = min(i + batch, total)
        print(f"  Embedding: {done}/{total}", flush=True)
        if done < total:
            time.sleep(LLM_CALL_INTERVAL)

    return all_embs


def build_vector_store(entities: pd.DataFrame, output_dir: Path, name: str):
    """生成 embedding 并写入 LanceDB."""
    titles = entities["title"].tolist()
    print(f"  Generating embeddings for {len(titles)} entities...")
    embeddings = generate_embeddings_sync(titles)

    db_path = output_dir / "lancedb"
    db_path.mkdir(parents=True, exist_ok=True)
    db = ldb.connect(str(db_path))

    table_name = "entity_description"  # GraphRAG 固定表名
    try:
        db.drop_table(table_name)
    except Exception:
        pass

    records = []
    for i, (_, row) in enumerate(entities.iterrows()):
        records.append({
            "id": row["id"], "title": row["title"],
            "vector": embeddings[i],
            "text_unit_ids": json.dumps(row.get("text_unit_ids", [])),
            "degree": int(row.get("degree", 0)), "type": row.get("type", ""),
        })
    db.create_table(table_name, records)
    print(f"  LanceDB → {db_path} ({len(records)} vectors)")

# ===================== 步骤 7: 写入 =====================

def write_parquets(output_dir: Path, entities, relationships, documents, text_units,
                   communities, community_reports):
    for df, fname in [
        (entities, "entities"), (relationships, "relationships"),
        (documents, "documents"), (text_units, "text_units"),
        (communities, "communities"), (community_reports, "community_reports"),
    ]:
        df.to_parquet(output_dir / f"{fname}.parquet", index=False)

    # 空 covariates
    pd.DataFrame(columns=[
        "id", "human_readable_id", "covariate_type", "type", "description",
        "subject_id", "object_id", "status", "start_date", "end_date",
        "source_text", "text_unit_id",
    ]).to_parquet(output_dir / "covariates.parquet", index=False)

# ===================== Settings =====================

def create_settings(name: str, fast: bool = False):
    output_dir = (BASE_DIR / name / "output").resolve()
    lancedb_path = (output_dir / "lancedb").resolve()

    # fast 模式下跳过 LLM 社区报告，community_prop=0 避免占位文本浪费 token
    community_prop_line = "  community_prop: 0.0" if fast else ""

    content = f"""\
completion_models:
  default:
    model_provider: openai
    model: {LLM_MODEL}
    api_base: {API_BASE}
    api_key: {API_KEY}
    auth_method: api_key
    concurrent_requests: 5
    async_mode: threaded

embedding_models:
  default:
    model_provider: openai
    model: {EMBEDDING_MODEL}
    api_base: {API_BASE}
    api_key: {API_KEY}
    auth_method: api_key

input:
  type: file
  base_dir: "input"

output_storage:
  type: file
  base_dir: "{output_dir}"

cache:
  type: memory

vector_store:
  type: lancedb
  db_uri: "{lancedb_path}"

embed_text:
  embedding_model_id: default
  vector_store_id: default

extract_claims:
  enabled: false

cluster_graph:
  max_cluster_size: {MAX_CLUSTER_SIZE}

community_reports:
  completion_model_id: default
  max_length: 2000
  max_input_length: 8000

local_search:
  completion_model_id: default
  embedding_model_id: default
{community_prop_line}
global_search:
  completion_model_id: default

drift_search:
  completion_model_id: default
  embedding_model_id: default

basic_search:
  completion_model_id: default
  embedding_model_id: default
"""
    (BASE_DIR / name / "settings.yaml").write_text(content)
    print(f"  Settings → {BASE_DIR / name / 'settings.yaml'}")

# ===================== 主流程 =====================

def build_dataset(name: str, fast: bool = False):
    print(f"\n{'='*60}")
    print(f"  Building: {name}  {'(fast mode — no LLM reports)' if fast else ''}")
    print(f"{'='*60}")

    out = BASE_DIR / name / "output"
    out.mkdir(parents=True, exist_ok=True)

    print("[1/7] Reading JSON...")
    data = read_openie_json(name)
    print(f"  {len(data['docs'])} documents")

    print("[2/7] Building entities & relationships...")
    entities, relationships = build_entity_relationship_df(data)
    print(f"  {len(entities)} entities, {len(relationships)} relationships")

    print("[3/7] Building documents & text units...")
    documents, text_units = build_documents_text_units_df(data)
    tu_e = {}; tu_r = {}
    for _, r in entities.iterrows():
        for tid in r.get("text_unit_ids", []): tu_e.setdefault(tid, []).append(r["id"])
    for _, r in relationships.iterrows():
        for tid in r.get("text_unit_ids", []): tu_r.setdefault(tid, []).append(r["id"])
    text_units["entity_ids"] = text_units["id"].map(lambda t: tu_e.get(t, []))
    text_units["relationship_ids"] = text_units["id"].map(lambda t: tu_r.get(t, []))
    print(f"  {len(documents)} documents, {len(text_units)} text units")

    print("[4/7] Community detection (Leiden)...")
    communities = run_community_detection(entities, relationships)
    print(f"  {len(communities)} communities")

    if fast:
        print("[5/7] Generating placeholder community reports (no LLM)...")
        community_reports = build_placeholder_reports(communities)
    else:
        print("[5/7] Generating community reports (LLM)...")
        community_reports = build_community_reports(communities, entities, relationships, name)
    print(f"  {len(community_reports)} reports")

    print("[6/7] Building vector store (Embedding)...")
    build_vector_store(entities, out, name)

    print("[7/7] Writing Parquet files...")
    write_parquets(out, entities, relationships, documents, text_units, communities, community_reports)

    print(f"\n  ✓ Done → {out}")
    print(f"    {len(entities)} entities | {len(relationships)} relationships")
    print(f"    {len(documents)} documents | {len(communities)} communities | {len(community_reports)} reports")

if __name__ == "__main__":
    args = sys.argv[1:]
    fast = "--fast" in args
    targets = [a for a in args if not a.startswith("--")]
    target = targets[0].lower() if targets else "all"

    if target in ("cs", "all"):
        build_dataset("cs", fast=fast); create_settings("cs", fast=fast)
    if target in ("ss", "all"):
        build_dataset("ss", fast=fast); create_settings("ss", fast=fast)
    print(f"\n✅ All done. Target: {target}")
    print(f"   Query: .venv/bin/python hevi_query/scripts/graphrag_hevi_query.py --dataset {target}")
