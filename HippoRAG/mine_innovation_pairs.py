"""
创新节点对挖掘 + 轻量 V3 语义覆盖评估脚本

管线:
  1. 加载 HippoRAG 索引 (indices/<name>)
  2. 自动识别 target 论文（文件名含 "【target】" 或在 openie 中标记）
  3. 挖掘 ref → target 跨论文创新节点对
  4. 计算轻量 V3 语义覆盖指标 (soft precision / recall / coverage)
  5. 输出 top 节点对 + 覆盖报告

用法:
    python mine_innovation_pairs.py                          # 默认 hlg_9336
    python mine_innovation_pairs.py --index-name hlg         # 其他索引
    python mine_innovation_pairs.py --top-n 20 --explain     # top20 + LLM解释
    python mine_innovation_pairs.py -o outputs/result.json   # 保存JSON
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

API_KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_key.txt")
with open(API_KEY_FILE) as f:
    os.environ["OPENAI_API_KEY"] = f.read().strip()

BASE_URL = "https://www.highland-api.top/v1"
LLM_MODEL = "deepseek-v4-pro"
EMBEDDING_MODEL = "text-embedding-3-large"
PROJECT_ROOT = Path(__file__).parent

from src.hipporag import HippoRAG
from src.hipporag.utils.config_utils import BaseConfig


# ============================================================
#  1. 加载索引
# ============================================================

def load_hipporag(index_name: str) -> HippoRAG:
    index_dir = PROJECT_ROOT / "indices" / index_name
    config = BaseConfig(
        save_dir=str(index_dir),
        llm_base_url=BASE_URL, llm_name=LLM_MODEL,
        embedding_model_name=EMBEDDING_MODEL, embedding_base_url=BASE_URL,
        force_index_from_scratch=False, force_openie_from_scratch=False,
        retrieval_top_k=20, linking_top_k=5, qa_top_k=5,
        max_new_tokens=384000, openie_mode="online",
    )
    hipporag = HippoRAG(global_config=config)
    hipporag.prepare_retrieval_objects()
    return hipporag


def load_openie_data(index_name: str) -> List[dict]:
    openie_path = PROJECT_ROOT / "indices" / index_name / "openie_results_ner_deepseek-v4-pro.json"
    if not openie_path.exists():
        return []
    return json.loads(openie_path.read_text(encoding="utf-8")).get("docs", [])


def load_raw_hlgs(source_dir: str) -> Dict[str, dict]:
    """直接从原始 HLG JSON 加载 Level1/2/3 + Relations"""
    hlg_dir = PROJECT_ROOT / source_dir
    papers = {}
    for f in sorted(hlg_dir.glob("*_hlg.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            name = f.stem.replace("_hlg", "")
            papers[name] = data
        except Exception:
            pass
    return papers


# ============================================================
#  2. 实体 → 论文映射 + Target 识别
# ============================================================

def get_entity_to_paper_mapping(hipporag: HippoRAG) -> Dict[str, List[str]]:
    mapping = {}
    for ent_id, chunk_ids in hipporag.ent_node_to_chunk_ids.items():
        papers = []
        for cid in chunk_ids:
            try:
                row = hipporag.chunk_embedding_store.get_row(cid)
                content = row.get("content", "")
                if content.startswith("Paper: "):
                    papers.append(content.split("\n")[0].replace("Paper: ", ""))
                else:
                    papers.append(content[:80])
            except Exception:
                papers.append(cid[:16])
        mapping[ent_id] = papers
    return mapping


def identify_target_paper(entity_to_papers: Dict[str, List[str]]) -> str:
    """自动识别 target 论文（文件名含 【target】）"""
    all_papers = set()
    for papers in entity_to_papers.values():
        all_papers.update(papers)
    for p in all_papers:
        if "【target】" in p:
            return p
    # fallback: 取第一篇作为 target
    return sorted(all_papers)[0] if all_papers else None


def get_entity_text(hipporag: HippoRAG, entity_id: str) -> str:
    try:
        return hipporag.entity_embedding_store.get_row(entity_id).get("content", entity_id)
    except Exception:
        return entity_id


# ============================================================
#  3. 跨论文节点对挖掘（ref → target 聚焦）
# ============================================================

def get_target_concepts(raw_hlgs: Dict[str, dict]) -> Dict[str, Set[str]]:
    """从原始 HLG 提取 target 论文的三层概念"""
    for name, hlg in raw_hlgs.items():
        if "【target】" in name:
            return {
                "Level1": set(hlg.get("Level1", [])),
                "Level2": set(hlg.get("Level2", [])),
                "Level3": set(hlg.get("Level3", [])),
                "all": set(hlg.get("Level1", []) + hlg.get("Level2", []) + hlg.get("Level3", [])),
            }
    return {"Level1": set(), "Level2": set(), "Level3": set(), "all": set()}


def compute_innovation_pairs(
    hipporag: HippoRAG,
    entity_to_papers: Dict[str, List[str]],
    target_paper: str,
    min_similarity: float = 0.3,
    top_k_per_entity: int = 5,
) -> List[dict]:
    """
    挖掘 ref paper 实体 → target paper 实体的创新节点对。
    只输出跨论文（至少一个是 ref，一个属于 target 或另一篇 ref）。
    """
    n = len(hipporag.entity_node_keys)
    embeddings = hipporag.entity_embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized_embs = embeddings / norms

    name_to_idx = hipporag.node_name_to_vertex_idx
    graph = hipporag.graph

    # 分类实体：target / ref
    target_entity_ids = set()
    ref_entity_ids = set()
    for ent_id, papers in entity_to_papers.items():
        if any("【target】" in p or p == target_paper for p in papers):
            target_entity_ids.add(ent_id)
        else:
            ref_entity_ids.add(ent_id)

    target_indices = [i for i, eid in enumerate(hipporag.entity_node_keys) if eid in target_entity_ids]
    ref_indices = [i for i, eid in enumerate(hipporag.entity_node_keys) if eid in ref_entity_ids]

    print(f"  Target 实体: {len(target_entity_ids)} 个")
    print(f"  Reference 实体: {len(ref_entity_ids)} 个")

    pairs = []
    seen_pairs = set()

    # 对每个 ref 实体，找最近的 target 实体和 cross-ref 实体
    batch_size = 32
    for batch_start in range(0, len(ref_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(ref_indices))
        batch_global_indices = [ref_indices[bi] for bi in range(batch_start, batch_end)]
        batch_embs = normalized_embs[batch_global_indices]

        # 与所有实体的相似度
        sim_all = np.dot(batch_embs, normalized_embs.T)

        for b in range(len(batch_global_indices)):
            gi = batch_global_indices[b]
            entity_i = hipporag.entity_node_keys[gi]
            papers_i = entity_to_papers.get(entity_i, [])
            sims = sim_all[b]

            # 排除自身
            sims[gi] = -1.0

            # 分别找最近的 target 实体和其他 ref 实体
            candidates = []

            # (a) 最近的 target 实体
            if target_indices:
                target_sims = [(j, sims[j]) for j in target_indices if sims[j] >= min_similarity]
                target_sims.sort(key=lambda x: x[1], reverse=True)
                for j, score in target_sims[:3]:
                    candidates.append(("target", j, score))

            # (b) 最近的其他 ref 实体（跨论文）
            for j in ref_indices:
                if j == gi:
                    continue
                score = float(sims[j])
                if score < min_similarity:
                    continue
                entity_j = hipporag.entity_node_keys[j]
                papers_j = entity_to_papers.get(entity_j, [])
                if set(papers_i) & set(papers_j):
                    continue  # 同论文跳过
                candidates.append(("ref", j, score))

            # 去重 + 排序
            candidates.sort(key=lambda x: x[2], reverse=True)
            added = 0
            for ctype, j, sim_score in candidates:
                if added >= top_k_per_entity:
                    break
                entity_j = hipporag.entity_node_keys[j]
                pair_key = tuple(sorted([entity_i, entity_j]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                papers_j = entity_to_papers.get(entity_j, [])

                # 图邻近度
                graph_prox = 0.1
                has_edge = False
                try:
                    idx_i = name_to_idx.get(entity_i)
                    idx_j = name_to_idx.get(entity_j)
                    if idx_i is not None and idx_j is not None:
                        eid = graph.get_eid(idx_i, idx_j, directed=False, error=False)
                        if eid != -1:
                            has_edge = True
                            graph_prox = 1.0
                        else:
                            path = graph.get_shortest_paths(idx_i, to=idx_j, output="vpath")
                            if path and path[0]:
                                dist = len(path[0]) - 1
                                graph_prox = {1: 1.0, 2: 0.8, 3: 0.6}.get(dist, 0.3)
                except Exception:
                    pass

                # 创新分数: 语义相似度 + 图结构 + 跨类型奖励
                cross_type_bonus = 0.15 if ctype == "target" else 0.0
                innovation_score = (
                    sim_score * 0.45
                    + graph_prox * 0.25
                    + (1.0 - abs(sim_score - 0.7)) * 0.15  # 最佳区间 0.6-0.8
                    + cross_type_bonus
                )

                pairs.append({
                    "entity_a": get_entity_text(hipporag, entity_i),
                    "entity_b": get_entity_text(hipporag, entity_j),
                    "paper_a": list(set(papers_i)),
                    "paper_b": list(set(papers_j)),
                    "pair_type": "ref→target" if ctype == "target" else "ref→ref",
                    "semantic_similarity": round(float(sim_score), 4),
                    "graph_proximity": round(graph_prox, 4),
                    "has_direct_edge": has_edge,
                    "innovation_score": round(innovation_score, 4),
                })
                added += 1

    pairs.sort(key=lambda x: x["innovation_score"], reverse=True)
    return pairs


# ============================================================
#  4. 轻量 V3 语义覆盖评估
# ============================================================

def evaluate_semantic_coverage(
    hipporag: HippoRAG,
    entity_to_papers: Dict[str, List[str]],
    target_paper: str,
    target_concepts: Dict[str, Set[str]],
    top_pairs: List[dict],
) -> dict:
    """
    轻量 V3 语义覆盖:
    - 计算所有 ref 实体对 target Level1+2+3 概念的 soft coverage
    - 模仿 V3 的 soft precision / recall / coverage 指标
    """
    if not target_concepts["all"]:
        return {"error": "未找到 target 概念"}

    ref_entity_ids = [
        eid for eid, papers in entity_to_papers.items()
        if not any("【target】" in p or p == target_paper for p in papers)
    ]

    if not ref_entity_ids:
        return {"error": "无 ref 实体"}

    ref_embeddings_list = []
    ref_texts = []
    for eid in ref_entity_ids:
        idx = hipporag.entity_node_keys.index(eid) if eid in hipporag.entity_node_keys else None
        if idx is not None:
            ref_embeddings_list.append(hipporag.entity_embeddings[idx])
            ref_texts.append(get_entity_text(hipporag, eid))

    if not ref_embeddings_list:
        return {"error": "ref 实体无 embedding"}

    ref_embs = np.array(ref_embeddings_list)
    ref_norms = np.linalg.norm(ref_embs, axis=1, keepdims=True)
    ref_norms = np.where(ref_norms == 0, 1.0, ref_norms)
    ref_embs = ref_embs / ref_norms

    # 对 target 概念编码（用 query embedding pipeline）
    target_texts = list(target_concepts["all"])
    target_embs = hipporag.embedding_model.batch_encode(target_texts, norm=True)
    target_embs = np.array(target_embs)

    # Soft Precision: mean over ref entities of max similarity to any target concept
    sim_ref_to_target = np.dot(ref_embs, target_embs.T)  # (n_ref, n_target)
    soft_precision = float(np.mean(np.max(sim_ref_to_target, axis=1)))

    # Soft Recall (Coverage): mean over target concepts of max similarity to any ref entity
    soft_recall = float(np.mean(np.max(sim_ref_to_target, axis=0)))

    soft_f1 = 2 * soft_precision * soft_recall / (soft_precision + soft_recall) if (soft_precision + soft_recall) > 0 else 0

    # 分层覆盖
    level_coverage = {}
    level_weights = {"Level1": 1.0, "Level2": 0.8, "Level3": 0.4}
    weighted_sum = 0.0
    weight_total = 0.0

    for level_name in ["Level1", "Level2", "Level3"]:
        concepts = list(target_concepts[level_name])
        if not concepts:
            level_coverage[level_name] = 0.0
            continue
        lvl_embs = hipporag.embedding_model.batch_encode(concepts, norm=True)
        lvl_embs = np.array(lvl_embs)
        lvl_sim = np.dot(ref_embs, lvl_embs.T)
        lvl_recall = float(np.mean(np.max(lvl_sim, axis=0)))
        level_coverage[level_name] = round(lvl_recall, 4)
        w = level_weights.get(level_name, 1.0)
        weighted_sum += w * lvl_recall * len(concepts)
        weight_total += w * len(concepts)

    weighted_coverage = weighted_sum / weight_total if weight_total > 0 else 0.0

    # Best matches per target concept
    per_concept_matches = []
    for ti, target_text in enumerate(target_texts):
        scores = sim_ref_to_target[:, ti]
        best_idx = int(np.argmax(scores))
        per_concept_matches.append({
            "target_concept": target_text,
            "best_ref_entity": ref_texts[best_idx],
            "similarity": round(float(scores[best_idx]), 4),
        })
    per_concept_matches.sort(key=lambda x: x["similarity"], reverse=True)

    # Coverage from top pairs
    top_pair_entities = set()
    for p in top_pairs[:20]:
        top_pair_entities.add(p["entity_a"])
        top_pair_entities.add(p["entity_b"])

    return {
        "soft_precision": round(soft_precision, 4),
        "soft_recall": round(soft_recall, 4),
        "soft_f1": round(soft_f1, 4),
        "level_coverage": level_coverage,
        "weighted_coverage": round(weighted_coverage, 4),
        "per_concept_matches": per_concept_matches[:15],
        "n_ref_entities": len(ref_entity_ids),
        "n_target_concepts": len(target_texts),
        "top_pair_coverage": len([c for c in per_concept_matches if c["best_ref_entity"] in top_pair_entities]),
    }


# ============================================================
#  5. 输出
# ============================================================

def explain_with_llm(pair: dict) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=BASE_URL)
    prompt = f"""You are a research innovation analyst. Given two concepts from different papers,
explain in 1 sentence what novel research idea combining them could produce.

Concept A: "{pair['entity_a']}" (from: {pair['paper_a']})
Concept B: "{pair['entity_b']}" (from: {pair['paper_b']})
Type: {pair['pair_type']} | Similarity: {pair['semantic_similarity']}

Novel research idea:"""
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120, temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(unavailable: {e})"


def print_report(
    pairs: List[dict],
    coverage: dict,
    target_concepts: Dict[str, Set[str]],
    top_n: int,
    explain: bool,
):
    """格式化输出完整报告"""

    # ── 语义覆盖报告 ──
    print(f"\n{'='*70}")
    print(f" 📊 轻量 V3 语义覆盖评估")
    print(f"{'='*70}")
    print(f"  Target 概念数: Level1={len(target_concepts['Level1'])}  "
          f"Level2={len(target_concepts['Level2'])}  Level3={len(target_concepts['Level3'])}")
    print(f"  Reference 实体数: {coverage.get('n_ref_entities', 'N/A')}")
    print()
    if "error" not in coverage:
        print(f"  Soft Precision (ref→target): {coverage['soft_precision']:.4f}")
        print(f"  Soft Recall    (target←ref): {coverage['soft_recall']:.4f}")
        print(f"  Soft F1:                      {coverage['soft_f1']:.4f}")
        print(f"  Weighted Coverage:            {coverage['weighted_coverage']:.4f}")
        print()
        print(f"  分层覆盖率:")
        for lvl in ["Level1", "Level2", "Level3"]:
            bar = "█" * int(coverage['level_coverage'][lvl] * 40)
            print(f"    {lvl}: {coverage['level_coverage'][lvl]:.4f} {bar}")

    # ── 创新节点对 ──
    print(f"\n{'='*70}")
    print(f" 🔗 跨论文创新节点对 (Top {top_n})")
    print(f"{'='*70}")

    ref_target = [p for p in pairs if p["pair_type"] == "ref→target"]
    ref_ref = [p for p in pairs if p["pair_type"] == "ref→ref"]

    print(f"\n  ── ref → target ({len(ref_target)} 对) ──")
    for rank, pair in enumerate(ref_target[:top_n]):
        _print_pair(rank + 1, pair, explain)

    if ref_ref:
        print(f"\n  ── ref → ref ({len(ref_ref)} 对) ──")
        for rank, pair in enumerate(ref_ref[:top_n // 2]):
            _print_pair(rank + 1, pair, explain)

    # ── 统计 ──
    print(f"\n{'='*70}")
    print(f" 统计: 共 {len(pairs)} 对 ({len(ref_target)} ref→target + {len(ref_ref)} ref→ref)")
    if pairs:
        scores = [p["innovation_score"] for p in pairs]
        sims = [p["semantic_similarity"] for p in pairs]
        print(f" 创新分数: {min(scores):.4f} ~ {max(scores):.4f} (avg {np.mean(scores):.4f})")
        print(f" 语义相似度: {min(sims):.4f} ~ {max(sims):.4f} (avg {np.mean(sims):.4f})")


def _print_pair(rank: int, pair: dict, explain: bool):
    tag = "🎯" if pair["pair_type"] == "ref→target" else "🔄"
    print(f"\n {tag} #{rank} | 创新: {pair['innovation_score']:.4f}  "
          f"| 相似: {pair['semantic_similarity']:.4f}  | 图: {pair['graph_proximity']:.4f}")
    if pair["has_direct_edge"]:
        print(f"     [直连边]")
    # 截断过长文本
    a = pair['entity_a'][:80]
    b = pair['entity_b'][:80]
    pa = str(pair['paper_a'])[:100]
    pb = str(pair['paper_b'])[:100]
    print(f"     A: {a}")
    print(f"        ← {pa}")
    print(f"     B: {b}")
    print(f"        ← {pb}")
    if explain:
        exp = explain_with_llm(pair)
        print(f"     💡 {exp}")


# ============================================================
#  6. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="创新节点对挖掘 + 语义覆盖评估")
    parser.add_argument("--index-name", "-i", default="hlg_9336", help="索引名")
    parser.add_argument("--source-dir", "-s", default=None, help="原始HLG目录（默认同索引名）")
    parser.add_argument("--top-n", "-n", type=int, default=15, help="输出top N")
    parser.add_argument("--explain", "-e", action="store_true", help="LLM解释")
    parser.add_argument("--min-sim", type=float, default=0.3, help="最低相似度")
    parser.add_argument("--output", "-o", default=None, help="保存JSON")
    args = parser.parse_args()

    source_dir = args.source_dir or args.index_name

    # 1. 加载
    print(f"\n{'='*60}")
    print(f" 加载索引: indices/{args.index_name}")
    hipporag = load_hipporag(args.index_name)
    info = hipporag.get_graph_info()
    print(f"  节点: {info['num_total_nodes']}, 边: {info['num_total_triples']}")

    # 2. 原始 HLG（用于获取 target 概念层级）
    raw_hlgs = load_raw_hlgs(source_dir)
    print(f"  加载 {len(raw_hlgs)} 个原始 HLG")

    # 3. 实体映射 + Target 识别
    entity_to_papers = get_entity_to_paper_mapping(hipporag)
    target_paper = identify_target_paper(entity_to_papers)
    target_concepts = get_target_concepts(raw_hlgs)

    print(f"\n{'='*60}")
    print(f" 🎯 Target 论文: {target_paper}")
    print(f"    Level1: {len(target_concepts['Level1'])} | "
          f"Level2: {len(target_concepts['Level2'])} | "
          f"Level3: {len(target_concepts['Level3'])}")
    if target_concepts["Level1"]:
        print(f"    Level1 示例: {list(target_concepts['Level1'])[:5]}")
    if target_concepts["Level2"]:
        print(f"    Level2 示例: {list(target_concepts['Level2'])[:5]}")
    if target_concepts["Level3"]:
        print(f"    Level3 示例: {list(target_concepts['Level3'])[:5]}")

    # 4. 挖掘
    print(f"\n{'='*60}")
    print(f" 挖掘跨论文创新节点对...")
    pairs = compute_innovation_pairs(
        hipporag, entity_to_papers, target_paper, min_similarity=args.min_sim,
    )

    # 5. 语义覆盖评估
    print(f"\n 计算语义覆盖...")
    coverage = evaluate_semantic_coverage(
        hipporag, entity_to_papers, target_paper, target_concepts, pairs,
    )

    # 6. 输出
    print_report(pairs, coverage, target_concepts, args.top_n, args.explain)

    # 7. 保存
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "target_paper": target_paper,
            "target_concepts": {k: list(v) for k, v in target_concepts.items()},
            "coverage": coverage,
            "pairs": pairs[:args.top_n],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {output_path}")


if __name__ == "__main__":
    main()
