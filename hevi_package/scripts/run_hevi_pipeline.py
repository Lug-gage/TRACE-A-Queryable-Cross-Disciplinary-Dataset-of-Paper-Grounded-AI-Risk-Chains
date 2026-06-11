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

from scripts.summarize_hevi_compare import summarize
from hipporag.hevi_workflow.agents import CSAgent, RiskLLM, SSAgent
from hipporag.hevi_workflow.hevi_compare import HEVIComparator
from hipporag.hevi_workflow.hit_report import compact_cs_output, compact_ss_output
from hipporag.hevi_workflow.pipeline import (
    build_ss_queries,
    judge_evidence,
    load_completed,
    normalize_hazard_list,
    run_bilateral_consensus,
    strip_internal_ids,
    write_summary_csv,
)
from hipporag.hevi_workflow.retrievers import RiskRetriever
from hipporag.hevi_workflow.utils import load_project_api_key, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="HEVI risk assessment pipeline with bilateral consensus.")
    parser.add_argument("--icml-dir", default="outputs/hevi_workflow/hevi_icml_{model}",
                        help="Directory with pre-computed ref_hevi JSON files from extract_reference_hevi.py")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--paper-ids", default="", help="Comma-separated paper IDs to process (overrides --limit)")
    parser.add_argument("--quality-report", default="outputs/hevi_workflow/hevi_icml_{model}/quality_report.json",
                        help="Path to quality_report.json; only process papers with verdict=keep")
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
    parser.add_argument("--hevi-output", default="outputs/hevi_workflow/hevi_{model}")
    parser.add_argument("--theta", type=float, default=0.8, help="Convergence threshold for bilateral consensus")
    parser.add_argument("--max-consensus-rounds", type=int, default=0, help="Max bilateral consensus rounds (0 = skip critique, DR synthesis only)")
    parser.add_argument("--group", type=int, default=0, help="Group number (1-6) for parallel runs. Auto-sets icml-dir/quality-report/hevi-output.")
    parser.add_argument("--min-slots", type=int, default=0, help="Only process papers with >= N non-empty HEVI slots (0=all)")
    parser.add_argument("--resume", action="store_true")
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

    if args.group and 1 <= args.group <= 6:
        base = Path(f"outputs/hevi_workflow/hevi_icml_{model_tag}")
        args.icml_dir = str(base / f"group_{args.group}")
        args.quality_report = str(base / f"group_{args.group}" / "quality_report.json")
        args.hevi_output = str(Path(f"outputs/hevi_workflow/hevi_{model_tag}") / f"group_{args.group}")

    hevi_base_dir = Path(str(args.hevi_output).replace("{model}", model_tag))
    hevi_base_dir.mkdir(parents=True, exist_ok=True)
    icml_dir = Path(str(args.icml_dir).replace("{model}", model_tag))

    llm = RiskLLM(
        save_dir=str(hevi_base_dir),
        llm_name=args.llm_name,
        llm_base_url=args.llm_base_url,
        max_new_tokens=args.max_new_tokens,
        request_timeout=args.llm_timeout,
        max_retries=args.llm_retries,
    )
    cs_retriever = RiskRetriever(
        index_dir=args.cs_index,
        source="cs",
        llm_name=args.llm_name,
        llm_base_url=args.llm_base_url,
        embedding_name=args.embedding_name,
        embedding_base_url=args.embedding_base_url,
    )
    ss_retriever = RiskRetriever(
        index_dir=args.ss_index,
        source="ss",
        llm_name=args.llm_name,
        llm_base_url=args.llm_base_url,
        embedding_name=args.embedding_name,
        embedding_base_url=args.embedding_base_url,
    )
    cs_agent = CSAgent(llm)
    ss_agent = SSAgent(llm)
    comparator = HEVIComparator(llm)

    # Load papers from pre-computed icml JSON files
    papers = []
    if icml_dir.exists():
        for f in sorted(icml_dir.glob("icml_*.json")):
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
    if args.quality_report:
        qr_path = Path(str(args.quality_report).replace("{model}", model_tag))
        if qr_path.exists():
            qr = json.loads(qr_path.read_text(encoding="utf-8"))
            keep_ids = {pid for pid, a in qr.get("papers", {}).items() if a.get("verdict") == "keep"}
            papers = [p for p in papers if p["paper_id"] in keep_ids]
            print(f"Filtered to {len(papers)} papers with verdict=keep from {qr_path}", flush=True)
        else:
            print(f"Warning: quality report not found at {qr_path}", flush=True)
    if args.paper_ids:
        ids = set(args.paper_ids.split(","))
        papers = [p for p in papers if p["paper_id"] in ids]
        print(f"Filtered to {len(papers)} papers by --paper-ids: {sorted(ids)}", flush=True)
    elif args.limit > 0 and len(papers) > args.limit:
        papers = papers[:args.limit]

    if args.min_slots > 0:
        before = len(papers)
        papers = [p for p in papers if sum(1 for v in p.get("ref_hevi", {}).values()
                  if isinstance(v, list) and len(v) > 0) >= args.min_slots]
        print(f"Filtered to {len(papers)} papers with >= {args.min_slots} slots (was {before})", flush=True)

    total = len(papers)
    completed = load_completed(hevi_base_dir) if args.resume else set()
    compare_rows: List[Dict[str, Any]] = []
    written = 0
    skipped = 0

    print(f"Loaded {total} papers from {icml_dir}", flush=True)

    for idx, paper in enumerate(papers, start=1):
        paper_id = paper["paper_id"]
        paper_dir = hevi_base_dir / paper_id

        if paper_id in completed:
            skipped += 1
            print(f"[{idx}/{total}] skip {paper_id}: already completed", flush=True)
            compare_file = paper_dir / "5_compare.json"
            if compare_file.exists():
                compare = json.loads(compare_file.read_text(encoding="utf-8"))
                if compare:
                    compare_rows.append(compare)
            continue

        print(f"[{idx}/{total}] {paper_id}", flush=True)

        try:
            # ── Stage 1: Reference ──────────────────────────────────────────
            print("[stage 1/5] reference: load ref_hevi", flush=True)
            query_terms = paper["query_terms"]
            ref_data = {
                "title": paper["title"],
                "abstract": paper["abstract"],
                "impact": paper.get("impact", ""),
                "query": paper["query"],
                "query_terms": query_terms,
                "ref_hevi": paper["ref_hevi"],
            }
            write_json(paper_dir / "1_reference.json", ref_data)

            # ── Stage 2: CS Proposal ────────────────────────────────────────
            print("[stage 2/5] CS: retrieve + propose + evidence judge", flush=True)
            cs_query = ref_data["query"]
            cs_evidence_raw = cs_retriever.retrieve(cs_query, top_k=args.top_k_cs)
            cs_proposal = cs_agent.propose_bilateral(
                paper_id=paper_id,
                cs_query=cs_query,
                cs_evidence=cs_evidence_raw,
            )
            cs_data = {
                "cs_query": cs_query,
                "cs_query_terms": ref_data.get("query_terms", []),
                "cs_evidence": strip_internal_ids(judge_evidence(
                    llm, "cs", cs_query,
                    compact_cs_output(cs_proposal),
                    cs_evidence_raw,
                )),
                "hazard": normalize_hazard_list(cs_proposal.get("hazard")),
                "nexus_candidates": [
                    {
                        "scenario": item.get("scenario"),
                        "issue": item.get("issue"),
                        "exposure": item.get("exposure"),
                        "confidence": item.get("confidence"),
                    }
                    for item in cs_proposal.get("nexus_candidates", []) or []
                ],
                "self_score": cs_proposal.get("self_score"),
            }
            write_json(paper_dir / "2_cs_proposal.json", strip_internal_ids(cs_data))

            # ── Stage 3: SS Response ────────────────────────────────────────
            print("[stage 3/5] SS: query + retrieve + respond + evidence judge", flush=True)
            ss_query_plan = build_ss_queries(cs_proposal, llm)
            ss_queries = ss_query_plan.get("ss_queries", [])
            ss_query = str(ss_queries[0].get("query", "")).strip() if ss_queries else ""

            ss_evidence_raw: List[Dict[str, Any]] = []
            if ss_query:
                ss_evidence_raw = ss_retriever.retrieve(ss_query, top_k=args.top_k_ss)
            ss_response = ss_agent.respond_bilateral(cs_proposal, ss_evidence_raw)

            ss_data = {
                "ss_query": ss_query,
                "ss_query_terms": ss_query_plan.get("query_terms", []),
                "ss_evidence": strip_internal_ids(judge_evidence(
                    llm, "ss", ss_query,
                    compact_ss_output(ss_response),
                    ss_evidence_raw,
                )),
                "nexus_responses": [
                    {
                        "decision": item.get("decision"),
                        "scenario": item.get("scenario"),
                        "issue": item.get("issue"),
                        "revision_reason": item.get("revision_reason"),
                        "vulnerability": item.get("vulnerability", []),
                        "impact": item.get("impact", []),
                        "key_control_nodes": item.get("key_control_nodes", []),
                        "social_mechanism": item.get("social_mechanism"),
                        "confidence": item.get("confidence"),
                    }
                    for item in ss_response.get("nexus_responses", []) or []
                ],
                "self_score": ss_response.get("self_score"),
            }
            write_json(paper_dir / "3_ss_response.json", strip_internal_ids(ss_data))

            # ── Stage 4: Bilateral Consensus ────────────────────────────────
            print("[stage 4/5] bilateral consensus: critique -> revise -> converge -> DR synthesis", flush=True)
            consensus = run_bilateral_consensus(
                cs_agent=cs_agent,
                ss_agent=ss_agent,
                llm=llm,
                cs_proposal=cs_proposal,
                ss_response=ss_response,
                cs_evidence=cs_evidence_raw,
                ss_evidence=ss_evidence_raw,
                cs_query=cs_query,
                theta=args.theta,
                max_rounds=args.max_consensus_rounds,
            )
            write_json(paper_dir / "4_consensus.json", strip_internal_ids(consensus))

            # ── Stage 5: Compare ────────────────────────────────────────────
            print("[stage 5/5] compare: workflow HEVI vs ref_hevi", flush=True)
            reference = {
                "title": ref_data["title"],
                "impact": paper.get("impact", ""),
                "hevi": ref_data["ref_hevi"],
            }
            hevi_compare = comparator.compare(consensus, reference)
            hevi_compare = strip_internal_ids(hevi_compare)
            write_json(paper_dir / "5_compare.json", hevi_compare)
            compare_rows.append(hevi_compare)

            print(f"[done] wrote {paper_id}/", flush=True)
            written += 1

        except Exception as exc:
            print(f"[{idx}/{total}] {paper_id} FAILED: {exc}", flush=True)
            continue

    summary_data = summarize(compare_rows)
    write_json(hevi_base_dir / "summary.json", summary_data)
    write_summary_csv(hevi_base_dir / "summary.csv", summary_data["papers"])

    print(
        json.dumps(
            {
                "hevi_output_dir": str(hevi_base_dir),
                "written": written,
                "skipped": skipped,
                "overall_item_recall": summary_data["overall"]["item_recall"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
