#!/usr/bin/env python3
"""Entry point for HEVI pipeline. Sets up sys.path and provides CLI."""
import os
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent

# Ensure the package's hipporag/ is found first, even when an editable install
# of another hipporag exists elsewhere on sys.path.
# 1. Remove any existing hipporag from sys.modules (avoid stale cache)
for _mod in list(sys.modules):
    if _mod == "hipporag" or _mod.startswith("hipporag."):
        del sys.modules[_mod]
# 2. Insert package root at absolute front of sys.path
if str(PKG_ROOT) in sys.path:
    sys.path.remove(str(PKG_ROOT))
sys.path.insert(0, str(PKG_ROOT))

# Auto-load API key from api_key.txt if present
_key_file = PKG_ROOT / "api_key.txt"
if _key_file.exists() and "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = _key_file.read_text(encoding="utf-8").strip()

if __name__ == "__main__":
    # Ensure all relative paths resolve against the package root
    os.chdir(str(PKG_ROOT))

    import argparse
    parser = argparse.ArgumentParser(description="HEVI Risk Assessment Pipeline")
    sub = parser.add_subparsers(dest="command")

    p1 = sub.add_parser("extract", help="Stage 1: Extract reference HEVI from papers")
    p2 = sub.add_parser("audit", help="Quality audit of extracted HEVI")
    p3 = sub.add_parser("run", help="Run full HEVI pipeline (stages 2-5)")
    p4 = sub.add_parser("export", help="Export comparison CSV")
    p5 = sub.add_parser("all", help="Serial: extract -> audit -> run -> visualize. Stops when target keep papers reached.")
    p6 = sub.add_parser("v2", help="HEVI v2 completion pipeline: fill empty ref_hevi slots")
    p5.add_argument("--target", type=int, default=10, help="Number of keep papers to collect before running pipeline (default: 10)")
    p5.add_argument("--top-k-cs", type=int, default=5)
    p5.add_argument("--top-k-ss", type=int, default=5)
    p5.add_argument("--llm-name", type=str, default="deepseek-v4-pro")

    args, unknown = parser.parse_known_args()

    if args.command == "extract":
        from scripts.extract_reference_hevi import main
    elif args.command == "audit":
        from scripts.audit_hevi_quality import main
    elif args.command == "run":
        from scripts.run_hevi_pipeline import main
    elif args.command == "export":
        from scripts.export_hevi_csv import main
    elif args.command == "v2":
        from scripts.run_hevi_pipeline_v2 import main
    elif args.command == "all":
        import csv
        import io
        import json
        import shutil

        from hipporag.hevi_workflow.agents import RiskLLM
        from hipporag.hevi_workflow.pipeline import build_workflow_input
        from scripts.audit_hevi_quality import audit_paper

        model = args.llm_name
        icml_dir = Path(f"outputs/hevi_workflow/hevi_icml_{model}")
        hevi_dir = Path(f"outputs/hevi_workflow/hevi_{model}")
        icml_dir.mkdir(parents=True, exist_ok=True)

        target = args.target
        min_impact = 500
        min_slots = 3

        # Existing papers (already extracted)
        already = set()
        for f in icml_dir.glob("icml_*.json"):
            if f.name not in ("quality_report.json", "summary.json"):
                already.add(f.stem)

        # Existing audit results
        qr_path = icml_dir / "quality_report.json"
        audit_results = {}
        if qr_path.exists():
            with open(qr_path) as f:
                audit_results = json.load(f).get("papers", {})

        # Count already-keep papers
        keep_ids = sorted([pid for pid, a in audit_results.items() if a.get("verdict") == "keep"])
        print(f"[all] Target: {target} keep papers | Already keep: {len(keep_ids)} | Already extracted: {len(already)}")

        # ── Serial extract + audit until target reached ──
        csv_path = PKG_ROOT / "data" / "icml_corpus_with_len.csv"
        if not csv_path.exists():
            print(f"[all] CSV not found: {csv_path}")
            sys.exit(1)

        with open(csv_path, encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r.get("paper_id", "").startswith("icml_")]

        llm = RiskLLM(
            save_dir=str(icml_dir),
            llm_name=model,
            llm_base_url="https://www.highland-api.top/v1",
            max_new_tokens=384000,
            temperature=0.0,
            request_timeout=120,
            max_retries=10,
        )

        extracted, kept = 0, len(keep_ids)
        for idx, paper in enumerate(rows, 1):
            if kept >= target:
                break

            pid = paper.get("paper_id", "")
            if not pid:
                continue

            # Skip if impact too short
            impact_len = int(paper.get("impact_chars", 0) or len(paper.get("impact", "").strip()))
            if impact_len < min_impact:
                continue

            # Extract if not already done
            out_path = icml_dir / f"{pid}.json"
            if pid not in already:
                print(f"  [{idx}] {pid} extracting...", flush=True)
                try:
                    ref_data = build_workflow_input(paper, llm)
                    n_slots = sum(1 for v in ref_data.get("ref_hevi", {}).values() if isinstance(v, list) and len(v) > 0)
                    if n_slots < min_slots:
                        # Still save, but skip audit
                        print(f"  [{idx}] {pid} slots={n_slots} < {min_slots}, skip", flush=True)
                        paper_out = {
                            "title": paper.get("title", ""),
                            "abstract": paper.get("abstract", ""),
                            "query": ref_data.get("query", ""),
                            "query_terms": ref_data.get("query_terms", []),
                            "impact": paper.get("impact", ""),
                            "hevi_node": n_slots,
                            "ref_hevi": ref_data.get("ref_hevi", {}),
                        }
                        with open(out_path, "w", encoding="utf-8") as fout:
                            json.dump(paper_out, fout, ensure_ascii=False, indent=2)
                        already.add(pid)
                        extracted += 1
                        continue

                    print(f"  [{idx}] {pid} slots={n_slots}, auditing...", flush=True)
                    # Save extracted file
                    paper_out = {
                        "title": paper.get("title", ""),
                        "abstract": paper.get("abstract", ""),
                        "query": ref_data.get("query", ""),
                        "query_terms": ref_data.get("query_terms", []),
                        "impact": paper.get("impact", ""),
                        "hevi_node": n_slots,
                        "ref_hevi": ref_data.get("ref_hevi", {}),
                    }
                    with open(out_path, "w", encoding="utf-8") as fout:
                        json.dump(paper_out, fout, ensure_ascii=False, indent=2)
                    already.add(pid)
                    extracted += 1
                except Exception as exc:
                    print(f"  [{idx}] {pid} extract FAILED: {exc}", flush=True)
                    continue
            else:
                # Already extracted, load slot count
                with open(out_path) as f:
                    slot_data = json.load(f)
                n_slots = sum(1 for v in slot_data.get("ref_hevi", {}).values() if isinstance(v, list) and len(v) > 0)
                if n_slots < min_slots:
                    continue

            # Audit if not already done
            if pid not in audit_results:
                print(f"  [{idx}] {pid} slots={n_slots}, auditing...", flush=True)
                try:
                    with open(out_path) as f:
                        paper_data = json.load(f)
                    audit = audit_paper(llm, paper_data)
                    audit_results[pid] = {
                        "overall_score": audit.get("overall_score", 0),
                        "verdict": audit.get("verdict", "reject"),
                        "scores": audit.get("scores", {}),
                        "issues": audit.get("issues", []),
                        "strengths": audit.get("strengths", []),
                        "summary": audit.get("summary", ""),
                        "diagnostics": audit.get("diagnostics", {}),
                    }
                    v = audit.get("verdict", "reject")
                    if v == "keep":
                        keep_ids.append(pid)
                        kept += 1
                        print(f"  [{idx}] {pid} score={audit.get('overall_score',0):.2f} → KEEP ({kept}/{target})", flush=True)
                    else:
                        print(f"  [{idx}] {pid} score={audit.get('overall_score',0):.2f} → REJECT", flush=True)
                except Exception as exc:
                    print(f"  [{idx}] {pid} audit FAILED: {exc}", flush=True)
                    continue

        # Save updated audit results
        verdicts = {"keep": 0, "reject": 0}
        for v in audit_results.values():
            verdicts[v.get("verdict", "reject")] = verdicts.get(v.get("verdict", "reject"), 0) + 1
        avg_score = sum(a.get("overall_score", 0) for a in audit_results.values()) / max(len(audit_results), 1)
        with open(qr_path, "w", encoding="utf-8") as f:
            json.dump({
                "audited": len(audit_results),
                "avg_score": round(avg_score, 3),
                "verdicts": verdicts,
                "papers": audit_results,
            }, f, ensure_ascii=False, indent=2)

        print(f"\n[all] Serial done: {len(keep_ids)} keep, {verdicts['reject']} reject (of {len(audit_results)} audited, {extracted} new)")

        # Take the first `target` keep papers
        target_ids = keep_ids[:target]
        if not target_ids:
            print("[all] No keep papers — cannot run pipeline")
            sys.exit(1)

        print(f"[all] Running {len(target_ids)} papers: {', '.join(target_ids[:5])}{'...' if len(target_ids)>5 else ''}")

        # ── Clean old pipeline outputs ──
        for pid in target_ids:
            d = hevi_dir / pid
            if d.exists():
                shutil.rmtree(d)

        # ── Run pipeline ──
        import subprocess as sp
        print("\n" + "="*60)
        print(f"STEP: run pipeline ({len(target_ids)} papers)")
        print("="*60)
        run_args = [
            sys.executable, "scripts/run_hevi_pipeline.py",
            "--limit", "0",
            "--paper-ids", ",".join(target_ids),
            "--top-k-cs", str(args.top_k_cs),
            "--top-k-ss", str(args.top_k_ss),
        ]
        ret = sp.run(run_args, cwd=str(PKG_ROOT))
        if ret.returncode != 0:
            print("[all] pipeline run failed")
            sys.exit(1)

        # ── Visualize ──
        print("\n" + "="*60)
        print("STEP: visualize")
        print("="*60)
        ret = sp.run(
            [sys.executable, "scripts/generate_visual.py"] + target_ids,
            cwd=str(PKG_ROOT)
        )

        print("\n" + "="*60)
        print(f"[all] DONE — {len(target_ids)} papers")
        print(f"  results: {hevi_dir}/")
        print(f"  visual:  visual/")
        print("="*60)
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)

    # Pass remaining args to the script's argparse
    sys.argv = [sys.argv[0]] + unknown
    main()
