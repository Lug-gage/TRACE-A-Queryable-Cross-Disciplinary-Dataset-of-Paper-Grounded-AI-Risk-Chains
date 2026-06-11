"""
HEVI 查询脚本 — LightRAG 版本

从 dataset.json 读取论文和 workflow_chains，用 LightRAG 检索 + LLM 生成
dose_response / vulnerability / impact。

用法:
    python hevi_query/scripts/hevi_query.py                          # 全部串行，ss 库
    python hevi_query/scripts/hevi_query.py --workers 5              # 5 协程并行，自动分片
    python hevi_query/scripts/hevi_query.py --limit 3                # 前 3 条
    python hevi_query/scripts/hevi_query.py --paper icml_2024_0001   # 指定论文
    python hevi_query/scripts/hevi_query.py --no-kg                  # 纯 LLM
    python hevi_query/scripts/hevi_query.py --dpr-only               # naive 向量检索
    python hevi_query/scripts/hevi_query.py --start 0 --count 100    # 手动分片
"""
import json, argparse, sys, logging, os, time, re, asyncio
from pathlib import Path
from functools import partial

logging.basicConfig(level=logging.WARNING, force=True)
_QUIET_LOGGERS = (
    "lightrag", "nano-vectordb", "openai", "httpx", "openai._base_client",
    "asyncio", "urllib3", "requests", "httpcore",
)
for _name in _QUIET_LOGGERS:
    logging.getLogger(_name).setLevel(logging.WARNING)


def _reapply_log_silence():
    """每次 initialize_storages 后重新压制日志"""
    for _name in _QUIET_LOGGERS:
        logging.getLogger(_name).setLevel(logging.WARNING)
    lightrag_logger.setLevel(logging.WARNING)


sys.path.insert(0, str(Path(__file__).parent.parent.parent))
os.environ["TQDM_DISABLE"] = "1"

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger as lightrag_logger

lightrag_logger.handlers.clear()
lightrag_logger.setLevel(logging.ERROR)
lightrag_logger.propagate = False

# ---- 路径 & 配置 ----
BASE_DIR       = Path(__file__).resolve().parent.parent   # hevi_query/
SCRIPTS_DIR    = Path(__file__).resolve().parent           # hevi_query/scripts/
DATASET_PATH   = BASE_DIR / "dataset.json"
OUTPUT_DIR     = BASE_DIR / "lightrag_results"; OUTPUT_DIR.mkdir(exist_ok=True)
VI_PROMPT      = (SCRIPTS_DIR / "hevi_vuln_impact.txt").read_text(encoding="utf-8")
DR_PROMPT      = (SCRIPTS_DIR / "hevi_dr.txt").read_text(encoding="utf-8")

# api_key.txt: 按 hevi_query/ → 上级目录 → CWD 顺序查找
_key_path = None
for _candidate in (BASE_DIR / "api_key.txt",
                   BASE_DIR.parent / "api_key.txt",
                   Path("api_key.txt")):
    if _candidate.exists():
        _key_path = _candidate
        break
if _key_path is None:
    raise FileNotFoundError("api_key.txt 未找到，请放在 hevi_query/ 或项目根目录或当前工作目录")
_api_key = _key_path.read_text().strip()
os.environ["OPENAI_API_KEY"] = _api_key

API_KEY        = _api_key
BASE_URL       = "https://www.highland-api.top/v1"
LLM_MODEL      = "deepseek-v4-pro"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM  = 3072
RETRY_MAX      = 5

# rag_storage 路径: 优先在 hevi_query/ 同级的项目根目录找，其次 CWD
_PROJECT_ROOT = BASE_DIR.parent


# ============================================================
#  helpers
# ============================================================

def _parse_json(text: str) -> dict:
    """多层兜底 --- 直接解析 -> 代码块 -> 最外层 {}"""
    for source in [text,
                   re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL),
                   re.search(r'\{.*\}', text, re.DOTALL)]:
        try:
            src = source.group(1) if isinstance(source, re.Match) else source
            return json.loads(src)
        except: pass
    return {}


# ============================================================
#  LLM 调用
# ============================================================

async def _llm_call(prompt: str, system_prompt: str = None, **kwargs) -> str:
    """5 次重试 + 指数退避（过滤 LightRAG 内部注入的 hashing_kv 等参数）"""
    forward_kwargs = {k: v for k, v in kwargs.items()
                      if k not in ("hashing_kv",)}
    last_error = None
    for attempt in range(1, RETRY_MAX + 1):
        try:
            return await openai_complete_if_cache(
                LLM_MODEL, prompt,
                system_prompt=system_prompt,
                api_key=API_KEY, base_url=BASE_URL,
                max_tokens=384000, timeout=120,
                **forward_kwargs,
            )
        except Exception as e:
            last_error = e
            if attempt < RETRY_MAX:
                wait = min(2 ** attempt, 30)
                print(f"    [LLM重试 {attempt}/{RETRY_MAX}] {type(e).__name__}, {wait}s后重试...")
                await asyncio.sleep(wait)
    raise last_error


# ---- 格式约束 system prompt（Turner (2003) 核心定义前置，不受检索上下文干扰）----
_VI_SYSTEM_PROMPT = """You are a risk assessment analyst following the Turner et al. (2003) HEVI framework.

=== TURNER (2003) DEFINITIONS ===
Hazard: A perturbation (spike, originates outside the system) or stress (continuous pressure, originates inside).
Exposure: Who / what / where the system encounters the hazard. EXPOSURE IS NOT SENSITIVITY — it describes the exposed unit and its context, NOT why harm happens.
Sensitivity (Vulnerability): The human-system conditions that determine how the system responds to exposure. It is the pre-existing coping capacity — why the exposed target cannot resist.
Impact: The negative consequences when hazard meets exposure under vulnerable conditions. What happens — who is affected, at what scale, what type of damage.

KEY BOUNDARY: Vulnerability ≠ Impact — vulnerability is sensitivity (WHY harm is possible). Impact is consequence (WHAT happens). Do NOT conflate them.
Exposure ≠ Vulnerability — exposure is who/what/where. Vulnerability is why that exposed unit cannot resist.

=== WRITING RULES ===
vulnerability: ONE concise phrase (≤30 words). The pre-existing condition making the exposed target susceptible. WHY harm is enabled. Not what happens after.
impact: ONE concise phrase (≤30 words). The concrete negative outcome when hazard exploits vulnerability. WHO is affected and WHAT type of damage. Not how it happens.

ASSERTIONS only — "X enables Y", never "X could potentially enable Y".
SPECIFIC — anchor every claim in the provided inputs.
Do NOT invent entities, mechanisms, or consequences absent from the input.

=== OUTPUT ===
ALWAYS output ONLY a single JSON object with no markdown fences and no extra text.
The JSON keys are exactly: vulnerability, impact."""

_DR_SYSTEM_PROMPT = """You are a dose-response synthesizer following Turner et al. (2003).

=== TURNER (2003) DEFINITION ===
Dose-Response is the CAUSAL TRANSLATION FUNCTION from hazard to impact through the coupled human-technical system.
It answers: how does the hazard translate into impact, given the specific exposure scenario and vulnerability conditions?
It is NOT a restatement of vulnerability. It is NOT a restatement of impact. It is the CAUSAL ARC that SPLICES them together.
dose-response = f(hazard + exposure context) → mechanism (via vulnerability) → output (impact magnitude).

=== WRITING RULES ===
EXACTLY ONE sentence, ≤40 words.
Structure: [hazard] → (through [exposure context]) → triggers [vulnerability condition] → leading to [impact].
The hazard and impact endpoints MUST stay intact from the input — only the arc between them is yours to write.
Do NOT restate vulnerability or impact verbatim as standalone claims. SPLICE them into one causal chain.
Do NOT inject vulnerability conditions into the arc as standalone descriptions — use them as the TRIGGER MECHANISM.
ASSERTIONS only. No "could", no "potentially", no hedging.

=== OUTPUT ===
Output ONLY a single JSON object with no markdown fences and no extra text.
The JSON key is exactly: dose_response."""


# ============================================================
#  检索：两条聚焦查询 + 合并去重
# ============================================================

async def _retrieve_and_format(
    rag: LightRAG, chain: dict, paper: dict, mode: str
) -> str:
    """用 hazard+exposure 和 scenario+issue 两条短查询分别检索，合并去重

    相比用完整 prompt 模板做检索，聚焦查询让嵌入模型拿到干净的语义信号，
    避免指令和定义文本淹没检索信号导致关键词提取失败。
    """
    hazard = _first(chain, "hazard")
    exposure = _first(chain, "exposure")
    scenario = chain.get("scenario", "")
    issue = chain.get("issue", "")

    he_query = f"{hazard}. {exposure}."
    si_query = f"{scenario}. {issue}."

    param = QueryParam(mode=mode, enable_rerank=False)

    # 并行检索两条查询
    ctx_he, ctx_si = await asyncio.gather(
        rag.aquery_data(he_query, param=param),
        rag.aquery_data(si_query, param=param),
    )

    # 合并去重
    merged = _merge_retrieval_contexts(ctx_he, ctx_si)
    return _context_to_text(merged)


def _merge_retrieval_contexts(*ctxs: dict) -> dict:
    """合并多个 aquery_data 结果，按 name/content 去重"""
    seen_entities = set()
    seen_relations = set()
    seen_chunks = set()

    merged = {"entities": [], "relationships": [], "chunks": []}

    for ctx in ctxs:
        data = ctx.get("data", ctx)
        for e in data.get("entities", []):
            key = (e.get("entity_name", ""), e.get("description", ""))
            if key not in seen_entities:
                seen_entities.add(key)
                merged["entities"].append(e)

        for r in data.get("relationships", []):
            key = (r.get("src_id", ""), r.get("tgt_id", ""), r.get("description", ""))
            if key not in seen_relations:
                seen_relations.add(key)
                merged["relationships"].append(r)

        for c in data.get("chunks", []):
            content = c.get("content", "")
            if content and content not in seen_chunks:
                seen_chunks.add(content)
                merged["chunks"].append(c)

    return merged


def _context_to_text(ctx: dict) -> str:
    """把检索上下文转成 LLM 可读文本，保持 compact"""
    lines = []

    entities = ctx.get("entities", [])
    for e in entities:
        name = e.get("entity_name", "")
        desc = e.get("description", "")
        if name and desc:
            lines.append(f"[Entity: {name}] {desc}")

    relations = ctx.get("relationships", [])
    for r in relations:
        src = r.get("src_id", "")
        tgt = r.get("tgt_id", "")
        desc = r.get("description", "")
        if src and tgt:
            lines.append(f"[Relation: {src} → {tgt}] {desc}")

    chunks = ctx.get("chunks", [])
    for c in chunks:
        content = c.get("content", "")
        if content:
            lines.append(f"[Chunk]: {content}")

    # 按条目边界截断：每条完整保留，超出总量才停
    max_chars = 10000
    result = []
    used = 0
    for line in lines:
        if used + len(line) + 1 > max_chars:
            break
        result.append(line)
        used += len(line) + 1

    return "\n".join(result) if result else "(No relevant knowledge found)"


# ============================================================
#  prompt 构造
# ============================================================

def _first(chain, key):
    v = chain.get(key, [])
    return v[0] if isinstance(v, list) and v else str(v)


def _vi_prompt(paper, chain):
    return VI_PROMPT.format(
        title=paper.get("title", ""), abstract=paper.get("abstract", ""),
        scenario=chain.get("scenario", ""), issue=chain.get("issue", ""),
        hazard=_first(chain, "hazard"), exposure=_first(chain, "exposure"),
    )


def _dr_prompt(chain, vulnerability, impact, paper=None):
    return DR_PROMPT.format(
        scenario=chain.get("scenario", ""), issue=chain.get("issue", ""),
        hazard=_first(chain, "hazard"), exposure=_first(chain, "exposure"),
        vulnerability=vulnerability, impact=impact,
        title=paper.get("title", "") if paper else "",
        abstract=paper.get("abstract", "") if paper else "",
    )


# ============================================================
#  LightRAG 实例
# ============================================================

def _make_rag(working_dir: str) -> LightRAG:
    return LightRAG(
        working_dir=working_dir,
        llm_model_func=_llm_call,
        default_llm_timeout=120,
        default_embedding_timeout=120,
        embedding_func_max_async=2,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=8192,
            func=partial(
                openai_embed.func,
                model=EMBEDDING_MODEL,
                api_key=API_KEY,
                base_url=BASE_URL,
            ),
        ),
    )


# ============================================================
#  主流程
# ============================================================

async def _process_chain(paper, ci, chain, rag, use_kg, mode, wid=0):
    """处理单条 chain 的完整 VI + DR 流水线，返回 (saved: bool, failed: bool)"""
    pid = paper["paper_id"]

    # Step 1: vulnerability + impact
    vi_prompt_text = _vi_prompt(paper, chain)
    if use_kg and rag:
        try:
            rag_ctx = await _retrieve_and_format(rag, chain, paper, mode)
        except Exception as e:
            # 检索超时/失败 → 退回到纯 LLM 模式，不丢数据
            print(f"  [W{wid}] {pid}_chain{ci} ⚠ 检索失败({type(e).__name__})，退回纯LLM", flush=True)
            rag_ctx = "(检索超时，以下为纯LLM推理)"
        user_prompt = (
            f"=== RETRIEVED KNOWLEDGE ===\n{rag_ctx}\n\n"
            f"=== TASK ===\n{vi_prompt_text}"
        )
        raw = await _llm_call(user_prompt, system_prompt=_VI_SYSTEM_PROMPT)
    else:
        raw = await _llm_call(vi_prompt_text, system_prompt=_VI_SYSTEM_PROMPT)
    vi = _parse_json(raw if isinstance(raw, str) else str(raw))

    # 门控
    vuln = vi.get("vulnerability", "")
    imp = vi.get("impact", "")
    if not vuln or not imp:
        print(f"  [W{wid}] {pid}_chain{ci} ✗ VI失败，跳过", flush=True)
        return False, True  # saved=False, failed=True

    # Step 2: dose_response
    dr_prompt_text = _dr_prompt(chain, vuln, imp, paper)
    raw = await _llm_call(dr_prompt_text, system_prompt=_DR_SYSTEM_PROMPT)
    dr = _parse_json(raw)
    dr_text = dr.get("dose_response", "")
    if not dr_text:
        dr_text = raw[:300] if isinstance(raw, str) else str(raw)[:300]

    out = {
        "paper_id": pid, "chain_index": ci,
        "title": paper.get("title", ""), "abstract": paper.get("abstract", ""),
        "impact": paper.get("impact", ""),
        "query_input": {
            "scenario": chain.get("scenario", ""), "issue": chain.get("issue", ""),
            "hazard": chain.get("hazard", []), "exposure": chain.get("exposure", []),
        },
        "lightrag_result": {"vulnerability": vuln, "impact": imp, "dose_response": dr_text},
        "reference": {
            "dose_response": chain.get("dose_response", []),
            "vulnerability": chain.get("vulnerability", []),
            "impact": chain.get("impact", []),
        },
    }

    dest = OUTPUT_DIR / f"{pid}_chain{ci}.json"
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(dest)

    print(f"  [W{wid}] {pid}_chain{ci} ✓", flush=True)
    return True, False


async def _run_worker(rag, tasks, use_kg, mode, wid):
    """单个 worker：顺序处理分配给它的 chains"""
    n = 0
    n_failed = 0
    for paper, ci, chain in tasks:
        saved, failed = await _process_chain(paper, ci, chain, rag, use_kg, mode, wid)
        if saved:
            n += 1
        if failed:
            n_failed += 1
    return n, n_failed


async def run(index="ss", limit=0, paper_id=None, no_kg=False, dpr_only=False,
              start=-1, count=0, workers=1):

    papers = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    use_kg = not no_kg
    working_dir = str(_PROJECT_ROOT / f"rag_storage_{index}")
    rag = _make_rag(working_dir) if use_kg else None
    if rag:
        await rag.initialize_storages()
        _reapply_log_silence()

    parallel = start >= 0 and count > 0
    mode = "naive" if dpr_only else "mix"

    # ---- 收集 chains ----
    tasks = []
    for p in papers:
        if paper_id and p["paper_id"] != paper_id:
            continue
        for ci, chain in enumerate(p.get("workflow_chains", [])):
            if not chain.get("hazard") or not chain.get("exposure"):
                continue
            tasks.append((p, ci, chain))

    total = len(tasks)

    if parallel:
        tasks = tasks[start:start + count]
    else:
        # 过滤已有结果
        tasks = [t for t in tasks
                 if not (OUTPUT_DIR / f"{t[0]['paper_id']}_chain{t[1]}.json").exists()]
        skipped_by_cache = total - len(tasks)
        skipped_by_limit = 0
        if limit and len(tasks) > limit:
            skipped_by_limit = len(tasks) - limit
            tasks = tasks[:limit]

    # ---- 打印任务概况 ----
    print(f"知识库: {index}  |  模式: {mode}")
    if parallel:
        print(f"手动分片: start={start} count={count}  →  {len(tasks)} 条")
    elif workers > 1:
        print(f"并行 worker: {workers}")
        print(f"跳过 {skipped_by_cache} 条（已有结果），待查询 {len(tasks)} 条")
    else:
        parts = []
        if skipped_by_cache:
            parts.append(f"跳过 {skipped_by_cache} 条（已有结果）")
        if skipped_by_limit:
            parts.append(f"因 limit 忽略 {skipped_by_limit} 条")
        if parts:
            print(", ".join(parts) + f"，待查询 {len(tasks)} 条")
        else:
            print(f"共 {len(tasks)} 条")
    print(flush=True)

    if not tasks:
        if rag:
            await rag.finalize_storages()
        return

    # ---- 执行 ----
    if workers > 1 and not parallel:
        # 多协程并行：把 tasks 分成 workers 组
        chunks = [[] for _ in range(workers)]
        for i, t in enumerate(tasks):
            chunks[i % workers].append(t)
        print(f"分组: {' | '.join(f'W{w}={len(c)}条' for w, c in enumerate(chunks))}")
        print(flush=True)

        results = await asyncio.gather(*[
            _run_worker(rag, chunk, use_kg, mode, wid)
            for wid, chunk in enumerate(chunks)
        ])
        n = sum(r[0] for r in results)
        n_failed = sum(r[1] for r in results)
    else:
        # 串行
        n, n_failed = await _run_worker(rag, tasks, use_kg, mode, 0)

    if rag:
        await rag.finalize_storages()

    summary_parts = [f"{n} 个文件  →  {OUTPUT_DIR}"]
    if n_failed:
        summary_parts.append(f"（{n_failed} 条 VI 失败跳过）")
    print("".join(summary_parts), flush=True)


# ============================================================
def main():
    p = argparse.ArgumentParser(description="HEVI 查询 (LightRAG)")
    p.add_argument("--index", "-i", default="ss", choices=["ss", "cs"])
    p.add_argument("--limit", "-l", type=int, default=0)
    p.add_argument("--paper", "-p")
    p.add_argument("--no-kg", action="store_true")
    p.add_argument("--dpr-only", action="store_true")
    p.add_argument("--start", "-s", type=int, default=-1)
    p.add_argument("--count", "-c", type=int, default=0)
    p.add_argument("--workers", "-w", type=int, default=1,
                   help="并行 worker 数（自动分片，与 --start/--count 互斥）")
    args = p.parse_args()
    asyncio.run(run(index=args.index, limit=args.limit, paper_id=args.paper,
                    no_kg=args.no_kg, dpr_only=args.dpr_only,
                    start=args.start, count=args.count,
                    workers=args.workers))


if __name__ == "__main__":
    main()
