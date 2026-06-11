import argparse
import csv
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SOURCES = ("icml", "cs", "ss")


def load_project_api_key() -> None:
    if os.getenv("OPENAI_API_KEY"):
        return

    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return

    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("\"'")
            if key == "OPENAI_API_KEY" and value:
                os.environ["OPENAI_API_KEY"] = value
                return


def patch_limited_openie(max_workers: int, delay_seconds: float = 0.0) -> None:
    """Limit online OpenIE concurrency. Set max_workers=1 for strict serial execution."""
    from tqdm import tqdm

    from src.hipporag.information_extraction.openie_openai import ChunkInfo, OpenIE
    from src.hipporag.utils.misc_utils import NerRawOutput, TripleRawOutput

    if max_workers < 1:
        raise ValueError("--openie-workers must be >= 1")

    def limited_batch_openie(
        self: OpenIE,
        chunks: Dict[str, ChunkInfo],
    ) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
        chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}

        ner_results_list: List[NerRawOutput] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        num_cache_hit = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            ner_futures = {
                executor.submit(self.ner, chunk_key, passage): chunk_key
                for chunk_key, passage in chunk_passages.items()
            }
            pbar = tqdm(as_completed(ner_futures), total=len(ner_futures), desc=f"NER workers={max_workers}")
            for future in pbar:
                result = future.result()
                ner_results_list.append(result)
                metadata = result.metadata
                total_prompt_tokens += metadata.get("prompt_tokens", 0)
                total_completion_tokens += metadata.get("completion_tokens", 0)
                if metadata.get("cache_hit"):
                    num_cache_hit += 1
                pbar.set_postfix(
                    {
                        "total_prompt_tokens": total_prompt_tokens,
                        "total_completion_tokens": total_completion_tokens,
                        "num_cache_hit": num_cache_hit,
                    }
                )
                if delay_seconds:
                    time.sleep(delay_seconds)

        triple_results_list: List[TripleRawOutput] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        num_cache_hit = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            triple_futures = {
                executor.submit(
                    self.triple_extraction,
                    ner_result.chunk_id,
                    chunk_passages[ner_result.chunk_id],
                    ner_result.unique_entities,
                ): ner_result.chunk_id
                for ner_result in ner_results_list
            }
            pbar = tqdm(
                as_completed(triple_futures),
                total=len(triple_futures),
                desc=f"Extracting triples workers={max_workers}",
            )
            for future in pbar:
                result = future.result()
                triple_results_list.append(result)
                metadata = result.metadata
                total_prompt_tokens += metadata.get("prompt_tokens", 0)
                total_completion_tokens += metadata.get("completion_tokens", 0)
                if metadata.get("cache_hit"):
                    num_cache_hit += 1
                pbar.set_postfix(
                    {
                        "total_prompt_tokens": total_prompt_tokens,
                        "total_completion_tokens": total_completion_tokens,
                        "num_cache_hit": num_cache_hit,
                    }
                )
                if delay_seconds:
                    time.sleep(delay_seconds)

        ner_results_dict = {res.chunk_id: res for res in ner_results_list}
        triple_results_dict = {res.chunk_id: res for res in triple_results_list}
        return ner_results_dict, triple_results_dict

    OpenIE.batch_openie = limited_batch_openie


def selected_sources(source: str) -> List[str]:
    return list(SOURCES) if source == "all" else [source]


def load_docs(corpus_path: Path, sample_size: int | None) -> List[str]:
    docs = []
    with open(corpus_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            docs.append(row["doc_text"])
            if sample_size is not None and len(docs) >= sample_size:
                break
    return docs


def is_local_url(url: str | None) -> bool:
    return bool(url and ("localhost" in url or "127.0.0.1" in url or "::1" in url))


def validate_credentials(args: argparse.Namespace) -> None:
    if args.dry_run:
        return

    load_project_api_key()

    needs_openai_api_key = False
    needs_openai_api_key = needs_openai_api_key or (
        args.openie_mode == "online"
        and not args.llm_name.startswith(("bedrock", "Transformers/"))
        and args.azure_endpoint is None
        and not is_local_url(args.llm_base_url)
    )
    needs_openai_api_key = needs_openai_api_key or (
        "text-embedding" in args.embedding_name
        and args.azure_embedding_endpoint is None
        and not is_local_url(args.embedding_base_url)
    )

    if (is_local_url(args.llm_base_url) or is_local_url(args.embedding_base_url)) and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "sk-local-placeholder"

    if needs_openai_api_key and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "Missing OPENAI_API_KEY.\n\n"
            "Set it before building indexes, for example:\n"
            "  export OPENAI_API_KEY='your_api_key'\n\n"
            "For an OpenAI-compatible provider, set the provider key in OPENAI_API_KEY and pass its base URLs, for example:\n"
            "  python scripts/build_paper_indexes.py --source icml --sample-size 100 "
            "--llm-name your-llm --llm-base-url https://your-provider/v1 "
            "--embedding-name text-embedding-3-small --embedding-base-url https://your-provider/v1\n"
        )


def build_one_source(args: argparse.Namespace, source: str) -> Dict[str, object]:
    corpus_path = Path(args.input_dir) / f"{source}_corpus.csv"
    save_dir = Path(args.save_root) / source
    docs = load_docs(corpus_path, args.sample_size)

    if args.dry_run:
        return {
            "source": source,
            "corpus_path": str(corpus_path),
            "save_dir": str(save_dir),
            "docs_loaded": len(docs),
            "dry_run": True,
        }

    from src.hipporag.HippoRAG import HippoRAG
    from src.hipporag.utils.config_utils import BaseConfig

    config = BaseConfig(
        save_dir=str(save_dir),
        llm_name=args.llm_name,
        llm_base_url=args.llm_base_url,
        max_new_tokens=args.max_new_tokens,
        embedding_model_name=args.embedding_name,
        embedding_base_url=args.embedding_base_url,
        azure_endpoint=args.azure_endpoint,
        azure_embedding_endpoint=args.azure_embedding_endpoint,
        force_index_from_scratch=args.force_index_from_scratch,
        force_openie_from_scratch=args.force_openie_from_scratch,
        openie_mode=args.openie_mode,
        ner_template_name=args.ner_template_name,
        triple_extraction_template_name=args.triple_template_name,
        triple_extraction_max_new_tokens=args.triple_max_new_tokens,
        rerank_dspy_file_path=args.rerank_dspy_file_path,
        save_openie=True,
    )

    hipporag = HippoRAG(global_config=config)
    hipporag.index(docs=docs)

    manifest = {
        "source": source,
        "corpus_path": str(corpus_path),
        "save_dir": str(save_dir),
        "docs_indexed": len(docs),
        "sample_size": args.sample_size,
        "llm_name": args.llm_name,
        "llm_base_url": args.llm_base_url,
        "max_new_tokens": args.max_new_tokens,
        "embedding_name": args.embedding_name,
        "embedding_base_url": args.embedding_base_url,
        "openie_mode": args.openie_mode,
        "openie_workers": args.openie_workers,
        "ner_template_name": args.ner_template_name,
        "triple_template_name": args.triple_template_name,
        "triple_max_new_tokens": args.triple_max_new_tokens,
        "rerank_dspy_file_path": args.rerank_dspy_file_path,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "index_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build separate HippoRAG indexes for normalized paper corpora.")
    parser.add_argument("--source", choices=("all", *SOURCES), default="all")
    parser.add_argument("--input-dir", default="papers/processed")
    parser.add_argument("--save-root", default="outputs/corpora_sample")
    parser.add_argument("--sample-size", type=int, default=0, help="Use all rows if set to 0.")
    parser.add_argument("--llm-name", default="deepseek-v4-flash")
    parser.add_argument("--llm-base-url", default="https://www.highland-api.top/v1")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=384000,
        help="Max output tokens for general LLM calls.",
    )
    parser.add_argument("--embedding-name", default="text-embedding-3-large")
    parser.add_argument("--embedding-base-url", default="https://www.highland-api.top/v1")
    parser.add_argument("--azure-endpoint", default=None)
    parser.add_argument("--azure-embedding-endpoint", default=None)
    parser.add_argument("--openie-mode", choices=("online", "offline", "Transformers-offline"), default="online")
    parser.add_argument("--openie-workers", type=int, default=4)
    parser.add_argument("--openie-delay", type=float, default=0.0)
    parser.add_argument("--ner-template-name", default="ner")
    parser.add_argument("--triple-template-name", default="triple_extraction")
    parser.add_argument(
        "--triple-max-new-tokens",
        type=int,
        default=384000,
        help="Max output tokens for triple extraction. Increase this if triple extraction finishes with reason=length.",
    )
    parser.add_argument("--rerank-dspy-file-path", default=None)
    parser.add_argument("--force-index-from-scratch", action="store_true")
    parser.add_argument("--force-openie-from-scratch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Show INFO logs from HippoRAG and HTTP clients.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    for logger_name in ("httpx", "httpcore", "openai", "openai._base_client"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.sample_size == 0:
        args.sample_size = None

    validate_credentials(args)

    if args.openie_mode == "online" and not args.dry_run:
        patch_limited_openie(args.openie_workers, args.openie_delay)

    results = [build_one_source(args, source) for source in selected_sources(args.source)]
    Path(args.save_root).mkdir(parents=True, exist_ok=True)
    with open(Path(args.save_root) / "build_manifest.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
