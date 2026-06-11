#!/usr/bin/env python3
"""Extract v2 prompts from run_hevi_pipeline_v2.py to prompts_v2/"""
import ast
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts_v2"

with open("scripts/run_hevi_pipeline_v2.py") as f:
    source_lines = f.readlines()

with open("scripts/run_hevi_pipeline_v2.py") as f:
    tree = ast.parse(f.read())

prompts = [
    ("extract_nexus", "01_extract_nexus.md", "Step 1: Nexus 提取", "看全论文+ref_hevi → 1-2 条因果链"),
    ("build_cs_query_v2", "02_cs_query.md", "Step 2a: CS 检索查询", "title+abstract+Nexus → CS query terms"),
    ("complete_cs", "03_cs_complete.md", "Step 2b: CS 侧补全", "CS 文献 → 补空 hazard/exposure，锁定已有"),
    ("build_ss_query_v2", "04_ss_query.md", "Step 3: SS 检索查询", "完整 CS 侧+Nexus → SS query terms"),
    ("complete_ss", "05_ss_complete.md", "Step 4: SS 侧补全", "SS 文献 → 补空 vuln/impact/KCN，锁定已有"),
    ("synthesize_dr_v2", "06_dr_synthesis.md", "Step 5: DR 合成", "拼接 CS+SS 段 → dose_response"),
]

for func_name, filename, title, desc in prompts:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            start = node.lineno - 1
            end = node.end_lineno
            source = ''.join(source_lines[start:end])
            content = f"# {title}\n\n**作用**: {desc}\n\n**位置**: `scripts/run_hevi_pipeline_v2.py → {func_name}()`\n\n```python\n{source}\n```\n"
            (PROMPTS_DIR / filename).write_text(content, encoding='utf-8')
            print(f"  {filename} — {title}")
            break

print(f"Done: {len(prompts)} prompts → prompts_v2/")
