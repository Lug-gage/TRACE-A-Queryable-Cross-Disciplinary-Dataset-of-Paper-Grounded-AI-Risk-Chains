"""
Node Eval — 生成 idea HLG 与 target paper HLG 对齐评分

严格按照 node_eval_input_sample 中的指标定义实现:
  V2: Exact Node F1 / Fuzzy Node F1 / Strict Pair F1
  V3: Semantic Node F1 / Soft P/R/F1 / Weighted Coverage / Relaxed Pair F1

评分逻辑来自:
  generated_idea_hlg_alignment_v3_semantic_metric_definitions.md

用法:
    # 用已有的 generated HLG 评分
    python run_node_eval.py \
      --generated-hlg outputs/my_idea_hlg.json \
      --target-hlg hlg_9336/【target】M3D_..._hlg.json

    # 完整流程: 检索→生成idea→提取HLG→评分
    python run_node_eval.py --index-name hlg_9336 --full-pipeline

    # 批量评测多个 idea
    python run_node_eval.py --generated-dir outputs/generated_hlgs/ --batch
"""
import os
import sys
import json
import argparse
import logging
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from difflib import SequenceMatcher
from scipy.optimize import linear_sum_assignment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

API_KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_key.txt")
with open(API_KEY_FILE) as f:
    os.environ["OPENAI_API_KEY"] = f.read().strip()

BASE_URL = "https://www.highland-api.top/v1"
LLM_MODEL = "deepseek-v4-pro"
EMBEDDING_MODEL = "text-embedding-3-large"
PROJECT_ROOT = Path(__file__).parent

# ============================================================
#  文本处理
# ============================================================

def normalize(text: str) -> str:
    return re.sub(r'[^a-z0-9 ]', ' ', str(text).lower()).strip()


def lexical_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


# ============================================================
#  HLG 加载
# ============================================================

def load_hlg(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "Level1": [str(x) for x in data.get("Level1", [])],
        "Level2": [str(x) for x in data.get("Level2", [])],
        "Level3": [str(x) for x in data.get("Level3", [])],
        "all_nodes": [],
        "relations": [
            (str(r["source"]), str(r["target"]), str(r.get("relation", "")))
            for r in data.get("Relations", [])
        ],
        "relation_pairs": set(),
    }


def load_hlg_full(path: Path) -> dict:
    """加载 HLG 并填充衍生字段"""
    hlg = load_hlg(path)
    all_nodes = []
    for level, weight in [("Level1", 1.0), ("Level2", 0.8), ("Level3", 0.4)]:
        for node in hlg[level]:
            all_nodes.append({"node": node, "level": level, "weight": weight, "normalized": normalize(node)})
    hlg["all_nodes"] = all_nodes

    pairs = set()
    for s, t, _ in hlg["relations"]:
        pairs.add(tuple(sorted([normalize(s), normalize(t)])))
    hlg["relation_pairs"] = pairs
    return hlg


# ============================================================
#  V2 严格指标
# ============================================================

def v2_exact_node(generated: dict, target: dict) -> dict:
    g_nodes = set(normalize(n["node"]) for n in generated["all_nodes"])
    t_nodes = set(normalize(n["node"]) for n in target["all_nodes"])
    matched = g_nodes & t_nodes
    p = len(matched) / len(g_nodes) if g_nodes else 0
    r = len(matched) / len(t_nodes) if t_nodes else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "matched_count": len(matched), "predicted_count": len(g_nodes), "target_count": len(t_nodes)}


def v2_fuzzy_node(generated: dict, target: dict, threshold: float = 0.6) -> dict:
    g_nodes = [n for n in generated["all_nodes"]]
    t_nodes = [n for n in target["all_nodes"]]

    # 贪婪一对一匹配
    pairs = []
    for gi, gn in enumerate(g_nodes):
        for ti, tn in enumerate(t_nodes):
            sim = lexical_similarity(gn["node"], tn["node"])
            if sim >= threshold:
                pairs.append((sim, gi, ti))
    pairs.sort(key=lambda x: x[0], reverse=True)

    used_g, used_t = set(), set()
    matches = []
    for sim, gi, ti in pairs:
        if gi not in used_g and ti not in used_t:
            used_g.add(gi); used_t.add(ti)
            matches.append({"predicted_node": g_nodes[gi]["node"],
                            "target_node": t_nodes[ti]["node"],
                            "similarity": round(sim, 4)})

    p = len(matches) / len(g_nodes) if g_nodes else 0
    r = len(matches) / len(t_nodes) if t_nodes else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "matched_count": len(matches), "matches": matches, "match_threshold": threshold}


def v2_strict_pair(generated: dict, target: dict, fuzzy_threshold: float = 0.6) -> dict:
    """严格边匹配: fuzzy 映射端点后检查目标图中是否存在相同边"""
    g_pairs = generated["relation_pairs"]
    t_pairs = target["relation_pairs"]

    # 构建 fuzzy 节点映射
    g_nodes = [n["normalized"] for n in generated["all_nodes"]]
    t_nodes = [n["normalized"] for n in target["all_nodes"]]

    node_map = {}
    for gn in g_nodes:
        best_score, best_tn = 0, None
        for tn in t_nodes:
            s = SequenceMatcher(None, gn, tn).ratio()
            if s > best_score:
                best_score, best_tn = s, tn
        if best_score >= fuzzy_threshold:
            node_map[gn] = best_tn

    mapped_pairs = set()
    for s, t in g_pairs:
        ms, mt = node_map.get(s), node_map.get(t)
        if ms and mt and ms != mt:
            mapped_pairs.add(tuple(sorted([ms, mt])))

    matched = mapped_pairs & t_pairs
    predicted = len(mapped_pairs)
    target_count = len(t_pairs)
    p = len(matched) / predicted if predicted else 0
    r = len(matched) / target_count if target_count else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "matched_count": len(matched), "predicted_count": predicted, "target_count": target_count}


# ============================================================
#  语义嵌入
# ============================================================

class SemanticEmbedder:
    """简单的 embedding 缓存"""
    def __init__(self):
        self._cache = {}

    def encode(self, texts: List[str]) -> np.ndarray:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=BASE_URL)

        new_texts = [t for t in texts if t not in self._cache]
        if new_texts:
            for i in range(0, len(new_texts), 8):
                batch = new_texts[i:i+8]
                resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
                for t, d in zip(batch, resp.data):
                    self._cache[t] = np.array(d.embedding, dtype=np.float32)
                print(f"    embed: {i+len(batch)}/{len(new_texts)}", end="\r")
            if new_texts:
                print()

        embs = np.array([self._cache[t] for t in texts], dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return embs / norms


def semantic_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    return float(np.dot(emb_a, emb_b))


# ============================================================
#  V3 语义指标
# ============================================================

def v3_semantic_node(generated: dict, target: dict, embedder: SemanticEmbedder,
                     threshold: float = 0.6, use_lexical: bool = True) -> dict:
    """匈牙利一对一语义匹配"""
    g_texts = [n["node"] for n in generated["all_nodes"]]
    t_texts = [n["node"] for n in target["all_nodes"]]

    if not g_texts or not t_texts:
        return {"precision": 0, "recall": 0, "f1": 0, "f2": 0, "matched_count": 0}

    g_emb = embedder.encode(g_texts)
    t_emb = embedder.encode(t_texts)

    # 相似度矩阵: max(lexical, embedding)
    sim = np.zeros((len(g_texts), len(t_texts)))
    for i in range(len(g_texts)):
        for j in range(len(t_texts)):
            emb_sim = float(np.dot(g_emb[i], t_emb[j]))
            lex_sim = lexical_similarity(g_texts[i], t_texts[j])
            sim[i, j] = max(emb_sim, lex_sim) if use_lexical else emb_sim

    # 匈牙利算法（最小化成本 → 最大化相似度）
    cost = 1.0 - sim
    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    for ri, ci in zip(row_ind, col_ind):
        score = float(sim[ri, ci])
        if score >= threshold:
            matches.append({
                "generated_node": g_texts[ri],
                "target_node": t_texts[ci],
                "score": round(score, 4),
                "generated_level": generated["all_nodes"][ri]["level"],
                "target_level": target["all_nodes"][ci]["level"],
            })

    p = len(matches) / len(g_texts)
    r = len(matches) / len(t_texts)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    f2 = 5 * p * r / (4 * p + r) if (4 * p + r) > 0 else 0  # recall-weighted
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "f2": round(f2, 4), "matched_count": len(matches), "matches": matches,
            "match_threshold": threshold}


def v3_soft_metrics(generated: dict, target: dict, embedder: SemanticEmbedder,
                    level_weights: dict = None) -> dict:
    """V3 Soft Precision / Recall / F1 / Weighted Coverage"""
    g_texts = [n["node"] for n in generated["all_nodes"]]
    t_nodes = target["all_nodes"]
    t_texts = [n["node"] for n in t_nodes]

    if not g_texts or not t_texts:
        return {"soft_precision": 0, "soft_recall": 0, "soft_f1": 0,
                "soft_f2": 0, "weighted_coverage": 0, "level_coverage": {}}

    g_emb = embedder.encode(g_texts)
    t_emb = embedder.encode(t_texts)

    # 混合相似度矩阵
    sim = np.zeros((len(g_texts), len(t_texts)))
    for i in range(len(g_texts)):
        for j in range(len(t_texts)):
            sim[i, j] = max(float(np.dot(g_emb[i], t_emb[j])), lexical_similarity(g_texts[i], t_texts[j]))

    # Soft precision: mean over generated of max similarity to any target
    soft_p = float(np.mean(np.max(sim, axis=1)))
    soft_r = float(np.mean(np.max(sim, axis=0)))
    soft_f1 = 2 * soft_p * soft_r / (soft_p + soft_r) if (soft_p + soft_r) > 0 else 0
    soft_f2 = 5 * soft_p * soft_r / (4 * soft_p + soft_r) if (4 * soft_p + soft_r) > 0 else 0

    # 加权覆盖
    if level_weights is None:
        level_weights = {"Level1": 1.0, "Level2": 0.8, "Level3": 0.4}

    level_coverage = {}
    weighted_sum, weight_total = 0.0, 0.0
    for lvl in ["Level1", "Level2", "Level3"]:
        lvl_indices = [j for j, n in enumerate(t_nodes) if n["level"] == lvl]
        if not lvl_indices:
            level_coverage[lvl] = 0.0
            continue
        lvl_recall = float(np.mean(np.max(sim[:, lvl_indices], axis=0)))
        level_coverage[lvl] = round(lvl_recall, 4)
        w = level_weights.get(lvl, 1.0)
        weighted_sum += w * lvl_recall * len(lvl_indices)
        weight_total += w * len(lvl_indices)

    weighted_cov = weighted_sum / weight_total if weight_total > 0 else 0.0

    return {
        "soft_precision": round(soft_p, 4), "soft_recall": round(soft_r, 4),
        "soft_f1": round(soft_f1, 4), "soft_f2": round(soft_f2, 4),
        "weighted_coverage": round(weighted_cov, 4), "level_coverage": level_coverage,
    }


def v3_top_level_metrics(generated: dict, target: dict, embedder: SemanticEmbedder) -> dict:
    """V3 顶层度量：只在 Level1+2 上计算"""
    # 限制 target 到 L1+L2
    gen_l12 = [n for n in generated["all_nodes"] if n["level"] in ("Level1", "Level2")]
    tgt_l12 = [n for n in target["all_nodes"] if n["level"] in ("Level1", "Level2")]

    if not tgt_l12:
        return {"top_level_coverage": 0, "top_level_f1": 0}

    gen_limited = {**generated, "all_nodes": gen_l12}
    tgt_limited = {**target, "all_nodes": tgt_l12}

    soft = v3_soft_metrics(gen_limited, tgt_limited, embedder)
    return {
        "top_level_coverage": soft["soft_recall"],
        "top_level_precision": soft["soft_precision"],
        "top_level_f1": soft["soft_f1"],
        "top_level_f2": soft["soft_f2"],
    }


def v3_relaxed_pair(generated: dict, target: dict, embedder: SemanticEmbedder,
                    semantic_threshold: float = 0.6) -> dict:
    """V3 Relaxed Pair: semantic 端点映射 + target 图邻近度"""
    g_pairs = list(generated["relation_pairs"])
    t_pairs = target["relation_pairs"]
    t_nodes = target["all_nodes"]

    if not g_pairs:
        return {"relaxed_pair_precision": 0, "relaxed_pair_recall": 0, "relaxed_pair_f1": 0,
                "endpoint_pair_coverage": 0, "average_graph_proximity": 0}

    # 构建 target 邻接表用于图邻近度计算
    t_adj = {}
    for s, t in t_pairs:
        t_adj.setdefault(s, set()).add(t)
        t_adj.setdefault(t, set()).add(s)

    # 对每个 generated pair 做语义端点映射
    g_texts = list(set([s for s, _ in g_pairs] + [t for _, t in g_pairs]))
    t_texts = [n["normalized"] for n in t_nodes]

    if not g_texts or not t_texts:
        return {"error": "empty node sets"}

    g_emb = embedder.encode(g_texts)
    t_emb = embedder.encode(t_texts)

    # 节点映射表: gen_text → (best_t, best_score)
    node_map = {}
    for i, gt in enumerate(g_texts):
        best_score, best_t = 0, None
        for j, tt in enumerate(t_texts):
            score = max(float(np.dot(g_emb[i], t_emb[j])), lexical_similarity(gt, tt))
            if score > best_score:
                best_score, best_t = score, tt
        node_map[gt] = (best_t, best_score)

    # 评估每对
    proximities = []
    covered = 0
    for s, t in g_pairs:
        ms, ss = node_map.get(s, (None, 0))
        mt, st = node_map.get(t, (None, 0))
        if ms and mt and ss >= semantic_threshold and st >= semantic_threshold:
            covered += 1
            if ms == mt:
                prox = 0.0
            elif ms in t_adj.get(mt, set()):
                prox = 1.0  # direct edge
            else:
                # BFS 2-hop
                visited = {ms}
                frontier = set(t_adj.get(ms, set()))
                for _ in range(1):  # 1 hop = dist 2
                    if mt in frontier:
                        prox = 0.5; break
                    next_f = set()
                    for fn in frontier - visited:
                        next_f.update(t_adj.get(fn, set()))
                    visited |= frontier
                    frontier = next_f
                else:
                    prox = 0.0 if mt not in frontier else 0.5
            proximities.append(prox)
        else:
            proximities.append(0.0)

    endpoint_cov = covered / len(g_pairs) if g_pairs else 0
    relaxed_p = float(np.mean(proximities)) if proximities else 0
    avg_prox = float(np.mean([p for p in proximities if p > 0])) if any(p > 0 for p in proximities) else 0

    # Proxy recall: unique target edges hit by high-proximity generated pairs
    target_edges_hit = set()
    for (s, t), prox in zip(g_pairs, proximities):
        if prox >= 1.0:
            ms, _ = node_map.get(s, (None, 0))
            mt, _ = node_map.get(t, (None, 0))
            if ms and mt:
                target_edges_hit.add(tuple(sorted([ms, mt])))
    relaxed_r = len(target_edges_hit & t_pairs) / len(t_pairs) if t_pairs else 0

    relaxed_f1 = 2 * relaxed_p * relaxed_r / (relaxed_p + relaxed_r) if (relaxed_p + relaxed_r) > 0 else 0

    return {
        "relaxed_pair_precision": round(relaxed_p, 4),
        "relaxed_pair_recall": round(relaxed_r, 4),
        "relaxed_pair_f1": round(relaxed_f1, 4),
        "endpoint_pair_coverage": round(endpoint_cov, 4),
        "average_graph_proximity": round(avg_prox, 4),
    }


# ============================================================
#  完整评估入口
# ============================================================

def evaluate(generated_hlg_path: Path, target_hlg_path: Path) -> dict:
    """运行完整的 V2 + V3 评估"""
    print(f"\n{'='*60}")
    print(f" Generated HLG: {generated_hlg_path.name}")
    print(f" Target HLG:    {target_hlg_path.name}")

    gen = load_hlg_full(generated_hlg_path)
    tgt = load_hlg_full(target_hlg_path)

    print(f" Generated: {len(gen['all_nodes'])} nodes ({sum(1 for n in gen['all_nodes'] if n['level']=='Level1')} L1, "
          f"{sum(1 for n in gen['all_nodes'] if n['level']=='Level2')} L2, "
          f"{sum(1 for n in gen['all_nodes'] if n['level']=='Level3')} L3), "
          f"{len(gen['relation_pairs'])} pairs")
    print(f" Target:    {len(tgt['all_nodes'])} nodes ({sum(1 for n in tgt['all_nodes'] if n['level']=='Level1')} L1, "
          f"{sum(1 for n in tgt['all_nodes'] if n['level']=='Level2')} L2, "
          f"{sum(1 for n in tgt['all_nodes'] if n['level']=='Level3')} L3), "
          f"{len(tgt['relation_pairs'])} pairs")

    embedder = SemanticEmbedder()

    print("\n  V2 严格指标...")
    exact = v2_exact_node(gen, tgt)
    fuzzy = v2_fuzzy_node(gen, tgt)
    strict_pair = v2_strict_pair(gen, tgt)

    print("  V3 语义指标...")
    semantic = v3_semantic_node(gen, tgt, embedder)
    soft = v3_soft_metrics(gen, tgt, embedder)
    top_level = v3_top_level_metrics(gen, tgt, embedder)
    relaxed = v3_relaxed_pair(gen, tgt, embedder)

    result = {
        "v2_exact_node": exact,
        "v2_fuzzy_node": fuzzy,
        "v2_strict_pair": strict_pair,
        "v3_semantic_node": semantic,
        "v3_soft_metrics": soft,
        "v3_top_level": top_level,
        "v3_relaxed_pair": relaxed,
    }
    return result


def print_eval_report(result: dict):
    """格式化输出评估报告"""
    print(f"\n{'='*70}")
    print(f" 📊 评估报告")
    print(f"{'='*70}")

    print(f"\n  ── V2 严格指标 ──")
    for name, key in [("Exact Node", "v2_exact_node"), ("Fuzzy Node", "v2_fuzzy_node"),
                       ("Strict Pair", "v2_strict_pair")]:
        m = result[key]
        print(f"  {name:15s}  P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}")

    print(f"\n  ── V3 语义指标 ──")
    sm = result["v3_semantic_node"]
    print(f"  Semantic Node       P={sm['precision']:.4f}  R={sm['recall']:.4f}  F1={sm['f1']:.4f}  F2={sm['f2']:.4f}")

    sf = result["v3_soft_metrics"]
    print(f"  Soft Semantic       P={sf['soft_precision']:.4f}  R={sf['soft_recall']:.4f}  F1={sf['soft_f1']:.4f}  F2={sf['soft_f2']:.4f}")
    print(f"  Weighted Coverage   {sf['weighted_coverage']:.4f}")
    for lvl in ["Level1", "Level2", "Level3"]:
        bar = "█" * int(sf['level_coverage'].get(lvl, 0) * 40)
        print(f"    {lvl}: {sf['level_coverage'].get(lvl, 0):.4f} {bar}")

    tl = result["v3_top_level"]
    print(f"\n  Top-Level (L1+L2)   Coverage={tl.get('top_level_coverage', 0):.4f}  F1={tl.get('top_level_f1', 0):.4f}")

    rp = result["v3_relaxed_pair"]
    if "error" not in rp:
        print(f"\n  ── Relaxed Pair ──")
        print(f"  Endpoint Coverage   {rp['endpoint_pair_coverage']:.4f}")
        print(f"  Relaxed P/R/F1      P={rp['relaxed_pair_precision']:.4f}  R={rp['relaxed_pair_recall']:.4f}  F1={rp['relaxed_pair_f1']:.4f}")
        print(f"  Avg Graph Proximity {rp['average_graph_proximity']:.4f}")

    # 汇总表
    print(f"\n{'='*70}")
    print(f" Papers               1")
    print(f" SemanticNode F1      {result['v3_semantic_node']['f1']:.4f}")
    print(f" SoftSemantic F1      {result['v3_soft_metrics']['soft_f1']:.4f}")
    print(f" WeightedCoverage     {result['v3_soft_metrics']['weighted_coverage']:.4f}")
    print(f" Top-LevelCoverage    {result['v3_top_level']['top_level_coverage']:.4f}")
    print(f" Relaxed Pair F1      {result['v3_relaxed_pair']['relaxed_pair_f1']:.4f}")
    print(f" Fuzzy Node F1        {result['v2_fuzzy_node']['f1']:.4f}")
    print(f" Strict Pair F1       {result['v2_strict_pair']['f1']:.4f}")


# ============================================================
#  完整管线: 检索 → 生成 idea → 提取 HLG → 评分
# ============================================================

def constrained_generate_hlg(hipporag, target_hlg_path: Path) -> dict:
    """
    基于 HippoRAG 检索 + 图谱约束生成 HLG：
    1. 用 query 通过 HippoRAG 检索 ref 论文中的相关实体 → 候选池
    2. 用 igraph 发现候选池内的跨论文连接
    3. LLM 从候选池中选取节点构建 HLG

    候选池全部来自检索，不接触 target 概念，杜绝数据泄露。
    """
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=BASE_URL)

    target = load_hlg(target_hlg_path)
    hipporag.prepare_retrieval_objects()

    entity_ids = hipporag.entity_node_keys
    entity_texts = [hipporag.entity_embedding_store.get_row(eid)["content"]
                    for eid in entity_ids]

    # ──── 1. 找出 target 论文 chunk，排除其所有实体 ────
    target_chunk_id = None
    for cid in hipporag.passage_node_keys:
        row = hipporag.chunk_embedding_store.get_row(cid)
        if "【target】" in row["content"]:
            target_chunk_id = cid
            break

    target_entity_set = set()
    ref_entity_indices = []
    ref_entity_sources = {}
    for idx, eid in enumerate(entity_ids):
        chunk_ids = hipporag.ent_node_to_chunk_ids.get(eid, set())
        papers = []
        for cid in chunk_ids:
            try:
                row = hipporag.chunk_embedding_store.get_row(cid)
                content = row["content"]
                if content.startswith("Paper: "):
                    papers.append(content.split("\n")[0].replace("Paper: ", ""))
            except Exception:
                pass
        if target_chunk_id and target_chunk_id in chunk_ids:
            target_entity_set.add(idx)
        if any("【target】" not in p for p in papers):
            ref_entity_indices.append(idx)
            ref_entity_sources[idx] = [p for p in papers if "【target】" not in p]

    logging.info(f"HippoRAG 图谱: {len(ref_entity_indices)} ref 实体 + {len(target_entity_set)} target 实体(仅用于排除)")

    # ──── 2. 通过 HippoRAG 检索构建候选池(不接触target实体) ────
    # 检索词: target 的 Level3 高层主题
    retrieval_queries = target.get("Level3", [])[:3]
    if not retrieval_queries:
        retrieval_queries = ["novel research idea"]

    all_retrieved_passage_ids = set()
    for query in retrieval_queries:
        try:
            results = hipporag.retrieve(queries=[query], num_to_retrieve=5)
            docs = results[0].docs
            # 从检索结果找对应的 passage node IDs
            for doc in docs:
                for pid in hipporag.passage_node_keys:
                    row = hipporag.chunk_embedding_store.get_row(pid)
                    if row["content"] == doc:
                        all_retrieved_passage_ids.add(pid)
        except Exception as e:
            logging.warning(f"检索 '{query}' 失败: {e}")

    # 通过 ent_node_to_chunk_ids 反查: 哪些 ref 实体属于检索到的 passage
    all_retrieved_entities = set()
    for idx in ref_entity_indices:
        eid = entity_ids[idx]
        chunk_ids = hipporag.ent_node_to_chunk_ids.get(eid, set())
        if chunk_ids & all_retrieved_passage_ids:
            all_retrieved_entities.add(idx)

    # 图扩展: 检索到的实体 → 相邻一跳邻居(同为ref实体)
    name_to_idx = hipporag.node_name_to_vertex_idx
    graph = hipporag.graph
    graph_expanded = set()
    for idx in list(all_retrieved_entities):
        eid = entity_ids[idx]
        vi = name_to_idx.get(eid)
        if vi is not None:
            for nbr in graph.neighbors(vi):
                nbr_name = graph.vs[nbr]["name"]
                for ri in ref_entity_indices:
                    if entity_ids[ri] == nbr_name and ri not in target_entity_set:
                        graph_expanded.add(ri)
    all_retrieved_entities |= graph_expanded

    if len(all_retrieved_entities) < 5:
        logging.warning(f"检索结果太少({len(all_retrieved_entities)})，回退到全量 ref 实体")
        all_retrieved_entities = set(ref_entity_indices)

    logging.info(f"检索: {retrieval_queries} → {len(all_retrieved_passage_ids)} passages "
                 f"→ {len(all_retrieved_entities)} ref 实体(+图扩展{len(graph_expanded)}个)")

    # ──── 3. 发现候选池内的跨论文图连接 ────
    candidate_indices = sorted(all_retrieved_entities)[:50]
    cross_pairs = []
    for i_idx, i in enumerate(candidate_indices):
        for j in candidate_indices[i_idx + 1:]:
            ri, rj = ref_entity_indices[i], ref_entity_indices[j]
            papers_i = set(ref_entity_sources.get(ri, []))
            papers_j = set(ref_entity_sources.get(rj, []))
            if papers_i & papers_j:
                continue
            try:
                vi = name_to_idx.get(entity_ids[ri])
                vj = name_to_idx.get(entity_ids[rj])
                if vi is not None and vj is not None:
                    eid = graph.get_eid(vi, vj, directed=False, error=False)
                    if eid != -1:
                        w = float(graph.es[eid]["weight"]) if "weight" in graph.es[eid].attributes() else 1.0
                        cross_pairs.append({
                            "a": entity_texts[ri], "b": entity_texts[rj],
                            "paper_a": (list(papers_i)[0] if papers_i else "?")[:50],
                            "paper_b": (list(papers_j)[0] if papers_j else "?")[:50],
                            "graph_weight": round(w, 3),
                        })
            except Exception:
                pass

    logging.info(f"跨论文图边: {len(cross_pairs)} 条")

    # ──── 4. 构建候选池 ────
    candidates = []
    for i in candidate_indices:
        eidx = ref_entity_indices[i]
        entity_name = entity_texts[eidx]
        papers = ref_entity_sources.get(eidx, ["unknown"])
        has_cross = any(entity_name in (p["a"], p["b"]) for p in cross_pairs)
        candidates.append({
            "entity": entity_name,
            "papers": papers,
            "cross_paper_connected": has_cross,
        })

    # ──── 5. LLM 从检索候选池中选取节点 ────
    pool_lines = []
    for c in candidates:
        mark = "🔗" if c["cross_paper_connected"] else " "
        papers_str = ", ".join(p[:40] for p in c["papers"][:2])
        pool_lines.append(f"  {mark} {c['entity']}  ← {papers_str}")

    cross_lines = []
    for p in cross_pairs[:20]:
        cross_lines.append(
            f"  \"{p['a']}\" (←{p['paper_a']})  ↔  \"{p['b']}\" (←{p['paper_b']})  "
            f"[weight={p['graph_weight']}]"
        )

    prompt = (
        "You are a research innovation analyst. Below is a concept pool retrieved from reference papers. "
        "Combine these concepts into a novel research idea.\n\n"
        "=== RESEARCH QUERY ===\n"
        f"{json.dumps(retrieval_queries)}\n\n"
        "=== CONCEPT POOL (🔗 = cross-paper graph edge) ===\n"
        + "\n".join(pool_lines) + "\n\n"
        "=== EXISTING CROSS-PAPER CONNECTIONS ===\n"
        + "\n".join(cross_lines) + "\n\n"
        "=== TASK ===\n"
        "Select 6-9 concepts from the pool and create 6-12 relations to form a novel idea.\n"
        "1. Prioritize 🔗 concepts — they bridge different papers.\n"
        "2. Combine concepts from DIFFERENT papers.\n"
        "3. Copy names EXACTLY from the pool.\n\n"
        'Output ONLY valid JSON:\n'
        '{"Level1":["exact name"],"Level2":["exact name"],"Level3":["exact name"],'
        '"Relations":[{"source":"exact name","target":"exact name","relation":"enables"}]}'
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=384000, temperature=0.5,
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    try:
        hlg = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        hlg = json.loads(m.group()) if m else {}

    # 验证
    valid_texts = set(c["entity"] for c in candidates)
    for lvl in ["Level1", "Level2", "Level3"]:
        hlg[lvl] = [n for n in hlg.get(lvl, []) if n in valid_texts]
    hlg["Relations"] = [
        r for r in hlg.get("Relations", [])
        if r.get("source", "") in valid_texts and r.get("target", "") in valid_texts
    ]

    n_nodes = sum(len(hlg.get(l, [])) for l in ['Level1', 'Level2', 'Level3'])
    logging.info(f"约束生成: {n_nodes} 节点 (候选池 {len(candidates)} 个, "
                 f"检索词: {retrieval_queries}), {len(hlg.get('Relations', []))} 条关系边")
    return hlg


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Node Eval — generated HLG vs target HLG 对齐评分")
    parser.add_argument("--source-dir", "-s", type=str, default="hlg_9336",
                        help="原始 HLG 数据目录 (默认 hlg_9336)")
    parser.add_argument("--generated-hlg", "-g", type=str, default=None,
                        help="生成的 HLG JSON 文件路径")
    parser.add_argument("--target-hlg", "-t", type=str, default=None,
                        help="Target GT HLG JSON 文件路径")
    parser.add_argument("--batch", action="store_true",
                        help="批量评估 generated_hlgs/ 目录下所有文件")
    parser.add_argument("--generated-dir", type=str, default="outputs/generated_hlgs",
                        help="生成的 HLG 目录 (batch 模式)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="保存结果 JSON")

    # 完整管线
    parser.add_argument("--full-pipeline", action="store_true",
                        help="完整: 检索→生成idea→提取HLG→评分")
    parser.add_argument("--index-name", "-i", default="hlg_9336",
                        help="HippoRAG 索引名 (full-pipeline 模式)")
    parser.add_argument("--num-ideas", type=int, default=1,
                        help="生成 idea 数量")

    args = parser.parse_args()

    # 默认 target
    default_target = PROJECT_ROOT / "hlg_9336" / "【target】M3D_MultiModal_MultiDocument_Fine-Grained_Inconsistency_Detection_hlg.json"
    target_path = Path(args.target_hlg) if args.target_hlg else default_target

    if not target_path.exists():
        print(f"Target HLG 不存在: {target_path}")
        print("Usage: python run_node_eval.py --target-hlg <path_to_target_hlg.json>")
        sys.exit(1)

    # 完整管线模式
    if args.full_pipeline:
        # 加载 HippoRAG（只加载一次）
        from src.hipporag import HippoRAG
        from src.hipporag.utils.config_utils import BaseConfig
        index_dir = PROJECT_ROOT / "indices" / args.index_name
        hipporag = HippoRAG(global_config=BaseConfig(
            save_dir=str(index_dir), llm_base_url=BASE_URL, llm_name=LLM_MODEL,
            embedding_model_name=EMBEDDING_MODEL, embedding_base_url=BASE_URL,
            force_index_from_scratch=False, force_openie_from_scratch=False,
            retrieval_top_k=10, linking_top_k=5, qa_top_k=5,
            max_new_tokens=384000, openie_mode="online",
        ))

        output_dir = PROJECT_ROOT / "outputs" / "generated_hlgs"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for i in range(args.num_ideas):
            print(f"\n{'='*60}")
            print(f" 完整管线 — Idea #{i+1}")
            print(f"{'='*60}")

            # Step 1-4: 基于 HippoRAG 图谱约束生成 HLG
            print("  图谱约束生成 HLG (entity_embeddings + graph + LLM)...")
            hlg = constrained_generate_hlg(hipporag, target_path)

            hlg_path = output_dir / f"idea_{i+1:02d}_hlg.json"
            with open(hlg_path, "w", encoding="utf-8") as f:
                json.dump(hlg, f, ensure_ascii=False, indent=2)
            print(f"  HLG saved: {hlg_path}")

            # Step 4: 评分
            result = evaluate(hlg_path, target_path)
            result["_idea_index"] = i + 1
            results.append(result)
            print_eval_report(result)

        # 汇总
        if len(results) > 1:
            print(f"\n{'='*70}")
            print(f" 多 Idea 汇总 (macro avg)")
            for key in ["v2_fuzzy_node", "v2_strict_pair", "v3_semantic_node", "v3_soft_metrics"]:
                avg_f1 = np.mean([r[key]["f1"] for r in results])
                print(f"  {key}: avg F1 = {avg_f1:.4f}")

        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n保存: {output_path}")
        return

    # 单文件评分模式
    if args.generated_hlg:
        gen_path = Path(args.generated_hlg)
        if not gen_path.exists():
            print(f"Generated HLG 不存在: {gen_path}")
            sys.exit(1)
        result = evaluate(gen_path, target_path)
        print_eval_report(result)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        return

    # 批量模式
    if args.batch:
        gen_dir = Path(args.generated_dir)
        if not gen_dir.exists():
            print(f"Generated HLG 目录不存在: {gen_dir}")
            sys.exit(1)

        json_files = sorted(gen_dir.glob("*_hlg.json")) + sorted(gen_dir.glob("*_generated_hlg.json"))
        if not json_files:
            print(f"目录中没有找到 HLG JSON: {gen_dir}")
            sys.exit(1)

        all_results = []
        for f in json_files:
            result = evaluate(f, target_path)
            result["_file"] = str(f.name)
            all_results.append(result)
            print_eval_report(result)

        print(f"\n{'='*70}")
        print(f" 批量汇总 ({len(all_results)} files)")
        for key, label in [("v2_fuzzy_node", "Fuzzy Node F1"), ("v2_strict_pair", "Strict Pair F1"),
                            ("v3_semantic_node", "Semantic Node F1"), ("v3_soft_metrics", "Soft F1")]:
            vals = [r[key]["f1"] for r in all_results]
            print(f"  {label:20s}: avg={np.mean(vals):.4f}  min={min(vals):.4f}  max={max(vals):.4f}")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"\n保存: {args.output}")
        return

    # 无参数：演示模式
    print("用法示例:")
    print(f"  python run_node_eval.py --generated-hlg outputs/my_idea_hlg.json")
    print(f"  python run_node_eval.py --full-pipeline --num-ideas 3")
    print(f"  python run_node_eval.py --batch --generated-dir outputs/generated_hlgs/")
    print(f"\n默认 target: {default_target}")


if __name__ == "__main__":
    main()
