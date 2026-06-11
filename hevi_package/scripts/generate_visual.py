#!/usr/bin/env python3
"""Generate per-paper interactive HTML visualization for HEVI pipeline results.

Usage:
  python3 scripts/generate_visual.py              # all papers
  python3 scripts/generate_visual.py icml_2024_0185  # single paper
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "hevi_workflow" / "hevi_deepseek-v4-pro"
VIZ_DIR = Path(__file__).resolve().parent.parent / "visual"

SLOT_ORDER = ["hazard", "exposure", "dose_response", "vulnerability", "impact", "key_control_nodes"]
SLOT_LABELS = {
    "hazard": "Hazard", "exposure": "Exposure", "dose_response": "Dose‑Response",
    "vulnerability": "Vulnerability", "impact": "Impact", "key_control_nodes": "Key control nodes",
}
SLOT_COLORS = {
    "hazard": "#0071e3", "exposure": "#5856d6", "dose_response": "#af52de",
    "vulnerability": "#ff9f0a", "impact": "#ff3b30", "key_control_nodes": "#34c759",
}


def load_paper(paper_dir: Path) -> dict | None:
    files = {
        "ref": paper_dir / "1_reference.json",
        "cs": paper_dir / "2_cs_proposal.json",
        "ss": paper_dir / "3_ss_response.json",
        "consensus": paper_dir / "4_consensus.json",
        "compare": paper_dir / "5_compare.json",
    }
    data = {"id": paper_dir.name}
    for key, path in files.items():
        if not path.exists():
            return None
        with open(path) as f:
            data[key] = json.load(f)
    return data


def e(s: str) -> str:
    if not s:
        return ""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def badge(label: str, cls: str) -> str:
    return f'<span class="badge badge-{cls}">{e(label)}</span>'


def query_chips(terms: list) -> str:
    if not terms:
        return '<span style="color:#ccc;">—</span>'
    tokens = [str(t).strip() for t in terms if str(t).strip()]
    if not tokens:
        return '<span style="color:#ccc;">—</span>'
    chips = ''.join(f'<span class="query-chip">{e(t)}</span>' for t in tokens)
    return f'<div class="query-chips">{chips}</div>'


# ---------------------------------------------------------------------------
# HTML snippet builders
# ---------------------------------------------------------------------------

def build_hero(title, pid, item_recall, matched_count, ref_item_count, rounds, converged):
    cls = "high" if item_recall >= 0.8 else "mid" if item_recall >= 0.4 else "low"
    return f'''<div class="hero">
  <div class="recall-ring {cls}">{int(item_recall * 100)}%</div>
  <h1>{e(title)}</h1>
  <div class="pid">{pid} &nbsp;|&nbsp; Item Recall: {matched_count}/{ref_item_count} &nbsp;|&nbsp; Consensus: {rounds} round(s) &nbsp;|&nbsp; Converged: {'✓' if converged else '✗'}</div>
</div>'''


def build_meta_bar(cs_final, ss_final, rounds, n_chains, n_cs_ev, n_ss_ev):
    return f'''<div class="meta-bar">
  <div class="meta-chip"><strong>{cs_final:.2f}</strong>CS final</div>
  <div class="meta-chip"><strong>{ss_final:.2f}</strong>SS final</div>
  <div class="meta-chip"><strong>{rounds}</strong>Rounds</div>
  <div class="meta-chip"><strong>{n_chains}</strong>Chains</div>
  <div class="meta-chip"><strong>{n_cs_ev}</strong>CS ev</div>
  <div class="meta-chip"><strong>{n_ss_ev}</strong>SS ev</div>
</div>'''


# ---- Context: Abstract + Impact ----

def build_context(abstract, impact):
    return f'''<div class="card" id="context">
  <h2>📄 Context</h2>
  <div class="cols">
    <div>
      <h3>Abstract</h3>
      <div class="text-box">{e(abstract)}</div>
    </div>
    <div>
      <h3>Impact statement</h3>
      <div class="text-box">{e(impact)}</div>
    </div>
  </div>
</div>'''


# ---- Ref HEVI ----

def build_ref_hevi(ref_hevi):
    slots_html = ""
    for s in SLOT_ORDER:
        items = ref_hevi.get(s, [])
        color = SLOT_COLORS[s]
        if items:
            items_html = "".join(f'<div class="ref-item">• {e(it)}</div>' for it in items)
        else:
            items_html = '<div class="ref-item empty-slot">—</div>'
        slots_html += f'''<div class="ref-slot" style="border-left-color:{color};">
      <div class="ref-slot-label" style="color:{color};">{SLOT_LABELS[s]}</div>
      {items_html}
    </div>'''

    return f'''<div class="card" id="ref-hevi">
  <h2>📋 Ref Hevi <span style="font-weight:400;font-size:11px;color:#86868b;">(extracted from impact statement)</span></h2>
  <div class="ref-hevi-grid">{slots_html}</div>
</div>'''


# ---- Stage 2: CS Proposal ----

def build_cs_proposal(cs_query, cs_hazard, cs_nexus, cs_self_score):
    score_cls = "high" if cs_self_score >= 0.8 else "medium" if cs_self_score >= 0.5 else "low"
    parts = [f'<div class="card" id="cs-proposal">']
    parts.append(f'<h2>⚙️ CS agent proposal {badge(f"self_score: {cs_self_score:.2f}", score_cls)}</h2>')
    parts.append(f'<h3>CS query <span style="font-weight:400;font-size:11px;color:#86868b;">→ CS Knowledge Graph</span></h3>')
    parts.append(query_chips(cs_query))

    parts.append(f'<h3 style="color:{SLOT_COLORS["hazard"]};">Hazard</h3>')
    for h in cs_hazard:
        parts.append(f'<div class="claim-item">• {e(h.get("hazard",""))} {badge(h.get("confidence","?"), h.get("confidence","low"))}</div>')

    parts.append('<h3>Nexus candidates</h3>')
    for i, n in enumerate(cs_nexus):
        parts.append(f'''<div class="nexus-card">
    <div class="nexus-label">Nexus {i+1} {badge(n.get("confidence","?"), n.get("confidence","low"))}</div>
    <div class="field-label">Scenario</div>
    <div>• {e(n.get("scenario",""))}</div>
    <div class="field-label">Issue</div>
    <div>• {e(n.get("issue",""))}</div>
    <div class="field-label" style="color:{SLOT_COLORS["exposure"]};">Exposure</div>
    <div>• {e(n.get("exposure",""))}</div>
  </div>''')

    parts.append('</div>')
    return '\n'.join(parts)


# ---- Stage 3: SS Response ----

def build_ss_response(ss_nexus_resp, ss_self_score, ss_query):
    score_cls = "high" if ss_self_score >= 0.8 else "medium" if ss_self_score >= 0.5 else "low"
    parts = [f'<div class="card" id="ss-response">']
    parts.append(f'<h2>🧠 SS agent response {badge(f"self_score: {ss_self_score:.2f}", score_cls)}</h2>')
    parts.append(f'<h3>SS query <span style="font-weight:400;font-size:11px;color:#86868b;">→ SS Knowledge Graph</span></h3>')
    parts.append(query_chips(ss_query))

    parts.append('<h3>Nexus responses</h3>')
    for i, nr in enumerate(ss_nexus_resp):
        dec = nr.get("decision", "reject")
        rev_html = ""
        if dec == "revise":
            rev_html = f'<div style="font-size:11px;color:#86868b;margin-top:4px;">Revision: {e(nr.get("revision_reason","")[:300])}</div>'
        vuln_items = nr.get("vulnerability", [])
        imp_items = nr.get("impact", [])
        kcn_items = nr.get("key_control_nodes", [])
        vuln = "<br>".join("• " + e(v) for v in vuln_items) if vuln_items else "—"
        impact = "<br>".join("• " + e(v) for v in imp_items) if imp_items else "—"
        kcn = "<br>".join("• " + e(k) for k in kcn_items) if kcn_items else "—"
        sm = e(nr.get("social_mechanism", ""))

        parts.append(f'''<div class="nexus-response-card">
    <div class="nr-header">
      <span class="nexus-label">Nexus {i+1}</span>
      {badge(dec.upper(), dec)}
      <strong>{e(nr.get("scenario",""))}</strong>
      {badge(nr.get("confidence","?"), nr.get("confidence","low"))}
    </div>
    {rev_html}
    <div class="field-label" style="color:{SLOT_COLORS["vulnerability"]};">Vulnerability</div>
    <div style="font-size:12px;">{vuln}</div>
    <div class="field-label" style="color:{SLOT_COLORS["impact"]};">Impact</div>
    <div style="font-size:12px;">{impact}</div>
    <div class="field-label" style="color:{SLOT_COLORS["key_control_nodes"]};">Key control nodes</div>
    <div style="font-size:12px;">{kcn}</div>
    <div class="field-label">Social mechanism</div>
    <div style="font-size:12px;">{sm}</div>
  </div>''')

    parts.append('</div>')
    return '\n'.join(parts)


# ---- Stage 4: Consensus ----

def build_consensus(trace, rounds, converged, cs_final, ss_final):
    parts = [f'<div class="card" id="consensus">']
    parts.append(f'<h2>🔄 Bilateral consensus protocol</h2>')

    if rounds == 0:
        parts.append(f'''
    <div style="text-align:center;padding:20px;">
      <div style="font-size:48px;">⚡</div>
      <div style="font-size:16px;font-weight:700;margin-top:8px;">Converged Immediately · 即刻收敛</div>
      <div style="font-size:13px;color:#86868b;margin-top:4px;">
        CS {cs_final:.2f} ≥ 0.8 and SS {ss_final:.2f} ≥ 0.8 —<br>
        both agents reached the consensus threshold on their first proposals.<br>
        双方 Agent 在首次提案即达到共识阈值，无需互评修订。
      </div>
    </div>''')
    else:
        parts.append('<div class="chart-container" id="chart-trajectory"></div>')
        parts.append('<div class="round-timeline">')
        for i, t in enumerate(trace):
            cs_s = t.get("cs_self_score") or t.get("r_A") or 0
            ss_s = t.get("ss_self_score") or t.get("r_B") or 0
            conv_cls = " converged" if cs_s >= 0.8 and ss_s >= 0.8 else ""
            parts.append(f'<div class="round-node{conv_cls}"><div class="rn">Round {i+1}</div><div style="display:flex;justify-content:center;gap:12px;margin-top:4px;"><div><span style="font-size:10px;color:#86868b;">CS</span><br><span class="score">{cs_s:.2f}</span></div><div><span style="font-size:10px;color:#86868b;">SS</span><br><span class="score">{ss_s:.2f}</span></div></div></div>')
        parts.append('</div>')

        # Round details
        for i, t in enumerate(trace):
            cs_crit = t.get("cs_critique", {})
            ss_crit = t.get("ss_critique", {})
            crit_parts1 = ''.join(f'<div style="margin-top:4px;font-size:11px;">{badge(c.get("assessment","?").upper(), c.get("assessment","reject"))} {e(c.get("technical_critique",""))}</div>' for c in cs_crit.get("critiques", []))
            crit_parts2 = ''.join(f'<div style="margin-top:4px;font-size:11px;">{e(c.get("social_critique", c.get("overall","")))}</div>' for c in ss_crit.get("critiques", []))
            parts.append(f'''<details style="margin:4px 0;">
    <summary style="font-weight:600;font-size:13px;padding:8px;background:#f9f9fb;border-radius:6px;">Round {i+1} Details</summary>
    <div class="cols" style="margin-top:8px;">
      <div class="critique-block critique-cs">
        <strong>CS → SS Critique</strong>
        <div style="margin-top:4px;">{e(cs_crit.get("overall",""))}</div>
        {crit_parts1}
      </div>
      <div class="critique-block critique-ss">
        <strong>SS → CS Critique</strong>
        <div style="margin-top:4px;">{e(ss_crit.get("overall",""))}</div>
        {crit_parts2}
      </div>
    </div>
  </details>''')

    parts.append('</div>')
    return '\n'.join(parts)


# ---- Chains ----

def build_chains(chains):
    parts = [f'<div class="card" id="chains">']
    parts.append(f'<h2>⛓️ Dose-response chains <span style="font-weight:400;font-size:12px;color:#86868b;">({len(chains)} chains)</span></h2>')

    for i, c in enumerate(chains):
        conf = c.get("confidence", "low")
        slot_html = ""
        for s in SLOT_ORDER:
            val = c.get(s)
            if isinstance(val, list) and val:
                txt = "<br>".join("• " + e(v) for v in val)
            elif val:
                txt = e(str(val))
            else:
                txt = '<span style="color:#ccc;">—</span>'
            slot_html += f'<div class="chain-slot"><strong style="color:{SLOT_COLORS[s]};">{SLOT_LABELS[s]}:</strong> {txt}</div>'

        cs_ev = "".join(f'<div class="evidence-item cs">📄 <strong>{e(ev.get("title",""))}</strong><br><span style="color:#86868b;">{e(ev.get("point",""))}</span></div>' for ev in c.get("evidence_trace", {}).get("cs_evidence", []))
        ss_ev = "".join(f'<div class="evidence-item ss">📄 <strong>{e(ev.get("title",""))}</strong><br><span style="color:#86868b;">{e(ev.get("point",""))}</span></div>' for ev in c.get("evidence_trace", {}).get("ss_evidence", []))

        parts.append(f'''<details class="chain-card" open>
    <summary>
      <span class="chain-idx">Chain {i+1}</span>
      {e(c.get("scenario",""))}
      {badge(conf.upper(), conf)}
    </summary>
    <div class="chain-body">
      <div style="font-size:12px;color:#86868b;margin-bottom:8px;"><strong>Issue:</strong> {e(c.get("issue",""))}</div>
      {slot_html}
      <div style="margin-top:12px;padding-top:12px;border-top:1px solid #e5e5e5;">
        <strong style="font-size:11px;color:#86868b;">📚 Evidence Trace</strong>
        {cs_ev}{ss_ev}
      </div>
    </div>
  </details>''')

    parts.append('</div>')
    return '\n'.join(parts)


# ---- Comparison ----

def build_comparison(slot_matches):
    parts = [f'<div class="card" id="comparison">']
    parts.append(f'<h2>📏 Comparison: Ref Hevi vs Workflow Hevi</h2>')
    parts.append('<div class="hevi-compare">')

    for s in SLOT_ORDER:
        matches = slot_matches.get(s, [])
        items_html = ""
        if matches:
            for m in matches:
                matched = m.get("matched")
                cls = "match" if matched else "miss"
                label = "MATCH" if matched else "MISS"
                reason_html = ""
                if m.get("reason"):
                    reason_html = f'<div style="font-size:10px;color:#86868b;margin-left:16px;">↳ {e(m.get("reason",""))}</div>'
                items_html += f'<div class="item">{badge(label, cls)} {e(m.get("reference_item",""))}{reason_html}</div>'
        else:
            items_html = '<div class="empty">— no reference items —</div>'

        parts.append(f'''<div class="slot">
      <h4 style="color:{SLOT_COLORS[s]};">{SLOT_LABELS[s]}</h4>
      {items_html}
    </div>''')

    parts.append('</div></div>')
    return '\n'.join(parts)


# ---- Evidence ----

def build_evidence(cs_evidence, ss_evidence):
    parts = [f'<div class="card" id="evidence">']
    parts.append(f'<h2>📚 Retrieved evidence</h2>')

    # CS Evidence
    parts.append(f'<h3>CS evidence <span style="font-weight:400;font-size:11px;color:#86868b;">({len(cs_evidence)} papers)</span></h3>')
    for ev in cs_evidence:
        title = e(ev.get("title", ""))
        abstract = e(ev.get("abstract", ""))
        rel = ev.get("evidence_relevance", {})
        sup = ev.get("evidence_support", {})
        parts.append(f'''<div class="ev-row">
    <div class="ev-title">📄 {title}</div>
    <div class="ev-blocks">
      <div class="ev-block">
        <div class="ev-block-label">Abstract</div>
        <div class="ev-block-content">{abstract}</div>
      </div>
      <div class="ev-block">
        <div class="ev-block-label">Relevance {badge(rel.get("label","?"), rel.get("label","low"))}</div>
        <div class="ev-block-content"><span style="font-size:10px;">{e(rel.get("reason",""))}</span></div>
      </div>
      <div class="ev-block">
        <div class="ev-block-label">Support {badge(sup.get("label","?"), sup.get("label","low"))}</div>
        <div class="ev-block-content"><span style="font-size:10px;">{e(sup.get("reason",""))}</span></div>
      </div>
    </div>
  </div>''')

    # SS Evidence
    parts.append(f'<h3 style="margin-top:20px;">SS Evidence <span style="font-weight:400;font-size:11px;color:#86868b;">({len(ss_evidence)} papers)</span></h3>')
    for ev in ss_evidence:
        title = e(ev.get("title", ""))
        abstract = e(ev.get("abstract", ""))
        rel = ev.get("evidence_relevance", {})
        sup = ev.get("evidence_support", {})
        parts.append(f'''<div class="ev-row">
    <div class="ev-title">📄 {title}</div>
    <div class="ev-blocks">
      <div class="ev-block">
        <div class="ev-block-label">Abstract</div>
        <div class="ev-block-content">{abstract}</div>
      </div>
      <div class="ev-block">
        <div class="ev-block-label">Relevance {badge(rel.get("label","?"), rel.get("label","low"))}</div>
        <div class="ev-block-content"><span style="font-size:10px;">{e(rel.get("reason",""))}</span></div>
      </div>
      <div class="ev-block">
        <div class="ev-block-label">Support {badge(sup.get("label","?"), sup.get("label","low"))}</div>
        <div class="ev-block-content"><span style="font-size:10px;">{e(sup.get("reason",""))}</span></div>
      </div>
    </div>
  </div>''')

    parts.append('</div>')
    return '\n'.join(parts)


# ---- JS ----

def build_js(trace):
    js_trace = []
    for t in trace:
        js_trace.append({
            "round": t.get("round"),
            "cs_score": t.get("cs_self_score") or t.get("r_A"),
            "ss_score": t.get("ss_self_score") or t.get("r_B"),
        })

    return f'''<script>
const _TRACE = {json.dumps(js_trace, ensure_ascii=False)};
(function() {{
  if (_TRACE.length > 0) {{
    const rounds = _TRACE.map((_,i) => i+1);
    const csScores = _TRACE.map(t => t.cs_score);
    const ssScores = _TRACE.map(t => t.ss_score);
    Plotly.newPlot("chart-trajectory", [
      {{ type: "scatter", x: rounds, y: csScores, mode: "lines+markers", name: "CS", line: {{ color: "#0071e3", width: 2.5 }}, marker: {{ size: 12 }} }},
      {{ type: "scatter", x: rounds, y: ssScores, mode: "lines+markers", name: "SS", line: {{ color: "#ff9f0a", width: 2.5 }}, marker: {{ size: 12 }} }},
    ], {{
      margin: {{ t: 10, r: 20, b: 30, l: 50 }},
      xaxis: {{ title: "Round", dtick: 1 }},
      yaxis: {{ title: "Self-Score", range: [0, 1.05] }},
      shapes: [{{ type: "line", x0: 0.5, x1: _TRACE.length+0.5, y0: 0.8, y1: 0.8, line: {{ dash: "dot", color: "#ccc", width: 1 }} }}],
      annotations: [{{ text: "θ = 0.8", x: _TRACE.length, y: 0.81, xref: "x", yref: "y", showarrow: false, font: {{ size: 10, color: "#86868b" }} }}],
    }}, {{ responsive: true }});
  }}
  document.querySelectorAll('.section-nav a').forEach(a => {{
    a.addEventListener('click', e => {{
      e.preventDefault();
      const el = document.querySelector(e.target.getAttribute('href'));
      if (el) el.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
    }});
  }});
}})();
</script>'''


# ---- HTML template ----

CSS = '''<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f5f7; color: #1d1d1f; line-height: 1.5; -webkit-hyphens: auto; hyphens: auto; text-align: justify; }
.container { max-width: 960px; margin: 0 auto; padding: 24px 20px 60px; }
.hero { background: #fff; border-radius: 16px; padding: 28px 32px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
.hero h1 { font-size: 22px; font-weight: 700; line-height: 1.3; padding-right: 90px; }
.hero .pid { font-size: 12px; color: #86868b; margin-top: 4px; }
.recall-ring { display: inline-flex; align-items: center; justify-content: center; width: 72px; height: 72px; border-radius: 50%; font-size: 24px; font-weight: 800; float: right; margin-left: 16px; }
.recall-ring.high { background: #d1f7d9; color: #1a7d2e; }
.recall-ring.mid { background: #fff3cd; color: #856404; }
.recall-ring.low { background: #ffe0e0; color: #c41e1e; }
.meta-bar { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }
.meta-chip { background: #fff; border-radius: 10px; padding: 8px 14px; font-size: 11px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }
.meta-chip strong { font-size: 15px; display: block; }
.card { background: #fff; border-radius: 12px; padding: 20px 24px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
.card h2 { font-size: 16px; margin-bottom: 12px; font-weight: 700; }
.card h3 { font-size: 14px; margin: 16px 0 8px; color: #1d1d1f; font-weight: 600; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 640px) { .cols { grid-template-columns: 1fr; } }
.text-box { background: #f9f9fb; border-radius: 8px; padding: 10px 12px; font-size: 12px; overflow-wrap: break-word; max-height: 160px; overflow-y: auto; line-height: 1.6; }

/* Query chips */
.query-chips { display: flex; flex-wrap: wrap; gap: 5px; }
.query-chip { display: inline-block; background: #e8f2ff; color: #0055b3; padding: 3px 8px; border-radius: 5px; font-size: 12px; font-weight: 500; white-space: nowrap; border: 1px solid #cce0ff; }

/* Ref HEVI */
.ref-hevi-grid { display: grid; grid-template-columns: 1fr; gap: 6px; }
.ref-slot { background: #f9f9fb; border-radius: 8px; padding: 8px 12px; border-left: 2px solid; }
.ref-slot-label { font-size: 11px; font-weight: 700; margin-bottom: 4px; }
.ref-item { font-size: 12px; padding: 2px 0; }
.empty-slot { color: #bbb; font-style: italic; }

/* CS Proposal */
.claim-item { font-size: 13px; padding: 4px 0; }
.nexus-card { border: 1px solid #e5e5e5; border-radius: 8px; padding: 10px; margin: 6px 0; font-size: 13px; }
.nexus-label { font-size: 10px; font-weight: 700; color: #86868b; letter-spacing: 0.5px; margin-bottom: 4px; }
.field-label { font-size: 11px; font-weight: 600; color: #86868b; display: block; margin-bottom: 2px; margin-top: 6px; }

/* SS Response */
.nexus-response-card { border: 1px solid #e5e5e5; border-radius: 8px; padding: 12px; margin: 8px 0; }
.nr-header { margin-bottom: 6px; display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }

/* Badges */
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 600; white-space: nowrap; }
.badge-match { background: #d1f7d9; color: #1a7d2e; }
.badge-miss { background: #ffe0e0; color: #c41e1e; }
.badge-accept { background: #d1f7d9; color: #1a7d2e; }
.badge-revise { background: #fff3cd; color: #856404; }
.badge-reject { background: #ffe0e0; color: #c41e1e; }
.badge-high { background: #d1f7d9; color: #1a7d2e; }
.badge-medium { background: #fff3cd; color: #856404; }
.badge-low { background: #ffe0e0; color: #c41e1e; }
.badge-none { background: #f5f5f5; color: #86868b; }

/* Round timeline */
.round-timeline { display: flex; gap: 10px; margin: 12px 0; }
.round-node { flex: 1; padding: 12px; border-radius: 8px; border: 2px solid #e5e5e5; text-align: center; }
.round-node.converged { border-color: #34c759; background: #f0fff4; }
.round-node .score { font-size: 22px; font-weight: 700; }
.round-node .rn { font-size: 11px; color: #86868b; }
.critique-block { margin: 8px 0; padding: 10px; border-radius: 6px; font-size: 12px; }
.critique-cs { background: #e8f2ff; border-left: 3px solid #0071e3; }
.critique-ss { background: #fff8e8; border-left: 3px solid #ff9f0a; }

/* Chains */
.chain-card { border: 1px solid #e5e5e5; border-radius: 10px; padding: 16px; margin-bottom: 10px; }
.chain-card summary { cursor: pointer; font-weight: 600; font-size: 13px; }
.chain-card summary:hover { color: #0071e3; }
.chain-idx { display: inline-block; background: #e8f2ff; color: #0071e3; padding: 1px 7px; border-radius: 4px; font-size: 11px; font-weight: 700; margin-right: 6px; }
.chain-body { margin-top: 12px; padding-top: 12px; border-top: 1px solid #f0f0f5; }
.chain-slot { margin: 4px 0; font-size: 12px; }
.chain-slot strong { font-size: 11px; display: block; color: #86868b; margin-top: 4px; }
.evidence-item { font-size: 11px; padding: 6px 8px; background: #f9f9fb; border-radius: 6px; margin: 3px 0; }
.evidence-item.cs { border-left: 3px solid #0071e3; }
.evidence-item.ss { border-left: 3px solid #ff9f0a; }

/* Comparison */
.hevi-compare { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.hevi-compare .slot { border: 1px solid #e5e5e5; border-radius: 8px; padding: 10px; }
.hevi-compare .slot h4 { font-size: 11px; margin-bottom: 6px; color: #1d1d1f; font-weight: 700; }
.hevi-compare .item { font-size: 12px; padding: 3px 0; border-bottom: 1px solid #f0f0f5; }
.hevi-compare .item:last-child { border-bottom: none; }
.hevi-compare .empty { color: #bbb; font-style: italic; font-size: 11px; }

/* Evidence */
.ev-row { margin-bottom: 12px; border: 1px solid #e5e5e5; border-radius: 8px; overflow: hidden; }
.ev-title { font-size: 13px; font-weight: 600; padding: 8px 12px; background: #f9f9fb; border-bottom: 1px solid #e5e5e5; }
.ev-blocks { display: grid; grid-template-columns: 2fr 1fr 1fr; gap: 0; }
@media (max-width: 640px) { .ev-blocks { grid-template-columns: 1fr; } }
.ev-block { padding: 8px 10px; border-right: 1px solid #f0f0f5; font-size: 11px; }
.ev-block:last-child { border-right: none; }
.ev-block-label { font-size: 10px; font-weight: 600; color: #86868b; margin-bottom: 4px; }
.ev-block-content { font-size: 11px; color: #555; max-height: 120px; overflow-y: auto; overflow-wrap: break-word; }

details summary { list-style: none; cursor: pointer; }
details summary::-webkit-details-marker { display: none; }
details summary::before { content: '▸ '; display: inline-block; transition: transform .2s; width: 14px; }
details[open] summary::before { transform: rotate(90deg); }

.section-nav { position: sticky; top: 0; background: #f5f5f7; padding: 8px 0; z-index: 10; display: flex; gap: 8px; flex-wrap: wrap; border-bottom: 1px solid #e5e5e5; margin-bottom: 16px; }
.section-nav a { font-size: 11px; color: #86868b; text-decoration: none; padding: 4px 8px; border-radius: 4px; }
.section-nav a:hover { color: #0071e3; background: #e8f2ff; }
.chart-container { min-height: 250px; }
</style>'''


def render_paper_html(paper: dict) -> str:
    pid = paper["id"]
    ref = paper["ref"]
    cs = paper["cs"]
    ss = paper["ss"]
    con = paper["consensus"]
    cmp = paper["compare"]

    title = ref["title"]
    abstract = ref.get("abstract", "")
    impact = ref.get("impact", "")
    ref_hevi = ref.get("ref_hevi", {})

    cs_query = cs["cs_query_terms"]
    cs_evidence = cs.get("cs_evidence", [])
    cs_hazard = cs.get("hazard", [])
    cs_nexus = cs.get("nexus_candidates", [])
    cs_self_score = float(cs.get("self_score", 0))

    ss_query = ss["ss_query_terms"]
    ss_evidence = ss.get("ss_evidence", [])
    ss_nexus_resp = ss.get("nexus_responses", [])
    ss_self_score = float(ss.get("self_score", 0))

    rounds = con.get("rounds", 0)
    converged = con.get("converged", False)
    cs_final = float(con.get("cs_self_score_final") or con.get("r_A_final", 0))
    ss_final = float(con.get("ss_self_score_final") or con.get("r_B_final", 0))
    trace = con.get("consensus_trace", [])
    chains = con.get("chains", [])

    item_recall = cmp.get("item_recall", 0)
    ref_item_count = cmp.get("reference_item_count", 0)
    matched_count = cmp.get("matched_item_count", 0)
    slot_matches = cmp.get("slot_matches", {})

    sections = [
        build_hero(title, pid, item_recall, matched_count, ref_item_count, rounds, converged),
        build_meta_bar(cs_final, ss_final, rounds, len(chains), len(cs_evidence), len(ss_evidence)),
        build_context(abstract, impact),
        build_ref_hevi(ref_hevi),
        build_cs_proposal(cs_query, cs_hazard, cs_nexus, cs_self_score),
        build_ss_response(ss_nexus_resp, ss_self_score, ss_query),
        build_consensus(trace, rounds, converged, cs_final, ss_final),
        build_chains(chains),
        build_comparison(slot_matches),
        build_evidence(cs_evidence, ss_evidence),
    ]

    body_html = '\n'.join(sections)
    js_html = build_js(trace)
    short_title = title[:60]

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HEVI · {e(short_title)}</title>
<script src="https://cdn.plot.ly/plotly-3.0.0.min.js"></script>
{CSS}
</head>
<body>
<div class="container">
<div class="section-nav">
  <a href="#context">Context</a>
  <a href="#ref-hevi">Ref HEVI</a>
  <a href="#cs-proposal">CS Agent</a>
  <a href="#ss-response">SS Agent</a>
  <a href="#consensus">Consensus</a>
  <a href="#chains">Chains</a>
  <a href="#comparison">Comparison</a>
  <a href="#evidence">Evidence</a>
</div>
{body_html}
</div>
{js_html}
</body>
</html>'''


def main():
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    targets = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    paper_dirs = sorted(d for d in OUTPUT_DIR.iterdir() if d.is_dir() and d.name.startswith("icml_"))

    generated = 0
    for d in paper_dirs:
        if targets and d.name not in targets:
            continue
        paper = load_paper(d)
        if not paper:
            print(f"  SKIP {d.name} (missing files)")
            continue
        html = render_paper_html(paper)
        out_path = VIZ_DIR / f"{d.name}.html"
        out_path.write_text(html, encoding="utf-8")
        recall = paper["compare"]["item_recall"]
        chains_n = len(paper["consensus"]["chains"])
        print(f"  ✅ {d.name}.html  (recall={recall:.0%}, rounds={paper['consensus']['rounds']}, chains={chains_n})")
        generated += 1

    print(f"\nGenerated {generated} files → {VIZ_DIR}/")
    if generated > 0:
        first = sorted(VIZ_DIR.glob("*.html"))[0]
        print(f"Open: file://{first}")


if __name__ == "__main__":
    main()
