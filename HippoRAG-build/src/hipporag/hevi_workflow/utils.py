import ast
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[3]


def load_project_api_key() -> None:
    if os.getenv("OPENAI_API_KEY"):
        return

    run_highland_path = REPO_ROOT / "run_highland.py"
    if not run_highland_path.exists():
        return

    content = run_highland_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"os\.environ\[[\"']OPENAI_API_KEY[\"']\]\s*=\s*([\"'].*?[\"'])", content)
    if not match:
        return

    try:
        api_key = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key


def parse_paper_doc(doc: str) -> Dict[str, str]:
    parsed = {"source": "", "paper_id": "", "title": "", "abstract": "", "doc_text": doc}
    for key, field in [("Source", "source"), ("Paper ID", "paper_id"), ("Title", "title"), ("Abstract", "abstract")]:
        match = re.search(rf"^{re.escape(key)}:\s*(.*)$", doc, flags=re.MULTILINE)
        if match:
            parsed[field] = match.group(1).strip()
    if not parsed["abstract"]:
        parsed["abstract"] = doc
    return parsed


def load_icml_inputs(path: str, limit: int = 0) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "source": row.get("source", "icml"),
                    "paper_id": row.get("paper_id", ""),
                    "original_id": row.get("original_id", ""),
                    "title": row.get("title", ""),
                    "abstract": row.get("abstract", ""),
                    "impact": row.get("impact", ""),
                    "impact_chars": row.get("impact_chars", ""),
                }
            )
            if limit and len(rows) >= limit:
                break
    return rows


def dump_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _repair_json(text: str) -> str:
    # Fix invalid escape sequences: only \", \\, \/, \b, \f, \n, \r, \t, \uXXXX are valid.
    text = re.sub(r"\\(?![\"/\\bfnrtu])", r"\\\\", text)
    return text


def extract_json_object(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            return json.loads(_repair_json(fenced.group(1)))

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        json_str = text[start : end + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return json.loads(_repair_json(json_str))

    raise ValueError("LLM response did not contain a JSON object.")

