#!/usr/bin/env python3
"""Extract all LLM prompts from the HEVI pipeline and save to prompts/ directory."""
import ast, json, re, sys
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
PROMPTS_DIR.mkdir(exist_ok=True)

def extract_string_literals(node):
    """Recursively extract all string literals from an AST node."""
    strings = []
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            strings.append(child.value)
        elif isinstance(child, ast.JoinedStr):
            # f-strings: extract the constant parts
            parts = []
            for v in child.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    parts.append(v.value)
            if parts:
                strings.append(''.join(parts))
    return strings

def find_prompt_function(tree, func_name):
    """Find a function by name and extract its system/user prompt patterns."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            source_lines = []
            # Get the raw source lines for the function
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name) and target.id in ('system', 'user'):
                            # Try to get the string content
                            strings = extract_string_literals(stmt.value)
                            if strings:
                                source_lines.append(f"# {target.id} = ")
                                for s in strings:
                                    source_lines.append(s)
            return '\n'.join(source_lines)
    return None

# ── 01: CS propose ──────────────────────────────────────
with open("hipporag/hevi_workflow/agents.py") as f:
    agents_tree = ast.parse(f.read())

with open("hipporag/hevi_workflow/pipeline.py") as f:
    pipeline_tree = ast.parse(f.read())

with open("hipporag/hevi_workflow/hit_report.py") as f:
    hit_tree = ast.parse(f.read())

with open("hipporag/hevi_workflow/hevi_compare.py") as f:
    compare_tree = ast.parse(f.read())

prompts = []

# 01 - CS propose_bilateral
for node in ast.walk(agents_tree):
    if isinstance(node, ast.FunctionDef) and node.name == 'propose_bilateral':
        # Get raw source
        with open("hipporag/hevi_workflow/agents.py") as f:
            lines = f.readlines()
        start = node.lineno - 1
        end = node.end_lineno
        source = ''.join(lines[start:end])
        prompts.append(("01_cs_propose.md", "CS Agent 提案", "Stage 2", "agents.py → CSAgent.propose_bilateral", source))
        break

# 02 - CS/SS evidence judge (same function, both sides)
for node in ast.walk(hit_tree):
    if isinstance(node, ast.FunctionDef) and node.name == 'judge_hits_with_llm':
        with open("hipporag/hevi_workflow/hit_report.py") as f:
            lines = f.readlines()
        start = node.lineno - 1
        end = node.end_lineno
        source = ''.join(lines[start:end])
        prompts.append(("02_evidence_judge.md", "证据裁决 (CS+SS 共用)", "Stage 2 & 3", "hit_report.py → judge_hits_with_llm", source))
        break

# 03 - SS query planning
for node in ast.walk(pipeline_tree):
    if isinstance(node, ast.FunctionDef) and node.name == 'build_ss_queries':
        with open("hipporag/hevi_workflow/pipeline.py") as f:
            lines = f.readlines()
        start = node.lineno - 1
        end = node.end_lineno
        source = ''.join(lines[start:end])
        prompts.append(("03_ss_query_planning.md", "SS 检索查询规划", "Stage 3", "pipeline.py → build_ss_queries", source))
        break

# 04 - SS respond_bilateral
for node in ast.walk(agents_tree):
    if isinstance(node, ast.FunctionDef) and node.name == 'respond_bilateral':
        with open("hipporag/hevi_workflow/agents.py") as f:
            lines = f.readlines()
        start = node.lineno - 1
        end = node.end_lineno
        source = ''.join(lines[start:end])
        prompts.append(("04_ss_respond.md", "SS Agent 响应", "Stage 3", "agents.py → SSAgent.respond_bilateral", source))
        break

# 05 - CS critique_ss
for node in ast.walk(agents_tree):
    if isinstance(node, ast.FunctionDef) and node.name == 'critique_ss':
        with open("hipporag/hevi_workflow/agents.py") as f:
            lines = f.readlines()
        start = node.lineno - 1
        end = node.end_lineno
        source = ''.join(lines[start:end])
        prompts.append(("05_cs_critique_ss.md", "CS Agent 评议 SS", "Stage 4 (共识)", "agents.py → CSAgent.critique_ss", source))
        break

# 06 - SS critique_cs
for node in ast.walk(agents_tree):
    if isinstance(node, ast.FunctionDef) and node.name == 'critique_cs':
        with open("hipporag/hevi_workflow/agents.py") as f:
            lines = f.readlines()
        start = node.lineno - 1
        end = node.end_lineno
        source = ''.join(lines[start:end])
        prompts.append(("06_ss_critique_cs.md", "SS Agent 评议 CS", "Stage 4 (共识)", "agents.py → SSAgent.critique_cs", source))
        break

# 07 - CS revise_proposal
for node in ast.walk(agents_tree):
    if isinstance(node, ast.FunctionDef) and node.name == 'revise_proposal':
        with open("hipporag/hevi_workflow/agents.py") as f:
            lines = f.readlines()
        start = node.lineno - 1
        end = node.end_lineno
        source = ''.join(lines[start:end])
        prompts.append(("07_cs_revise.md", "CS Agent 修订提案", "Stage 4 (共识)", "agents.py → CSAgent.revise_proposal", source))
        break

# 08 - SS revise_response
for node in ast.walk(agents_tree):
    if isinstance(node, ast.FunctionDef) and node.name == 'revise_response':
        with open("hipporag/hevi_workflow/agents.py") as f:
            lines = f.readlines()
        start = node.lineno - 1
        end = node.end_lineno
        source = ''.join(lines[start:end])
        prompts.append(("08_ss_revise.md", "SS Agent 修订响应", "Stage 4 (共识)", "agents.py → SSAgent.revise_response", source))
        break

# 09 - DR synthesis
for node in ast.walk(pipeline_tree):
    if isinstance(node, ast.FunctionDef) and node.name == '_synthesize_dose_response':
        with open("hipporag/hevi_workflow/pipeline.py") as f:
            lines = f.readlines()
        start = node.lineno - 1
        end = node.end_lineno
        source = ''.join(lines[start:end])
        prompts.append(("09_dr_synthesis.md", "Dose-Response 合成", "Stage 4", "pipeline.py → _synthesize_dose_response", source))
        break

# 10 - HEVI compare
for node in ast.walk(compare_tree):
    if isinstance(node, ast.FunctionDef) and node.name == 'compare':
        with open("hipporag/hevi_workflow/hevi_compare.py") as f:
            lines = f.readlines()
        start = node.lineno - 1
        end = node.end_lineno
        source = ''.join(lines[start:end])
        prompts.append(("10_hevi_compare.md", "HEVI 对比评测", "Stage 5", "hevi_compare.py → HEVIComparator.compare", source))
        break

for filename, title, stage, location, source in prompts:
    # Clean up: extract the system and user prompt strings
    # The source is the raw Python code; we save it as-is with markdown wrapper
    content = f"""# {title}

**Stage**: {stage}
**Location**: `{location}`

```python
{source}
```
"""
    path = PROMPTS_DIR / filename
    path.write_text(content, encoding='utf-8')
    print(f"  {filename} — {title}")

print(f"\nDone: {len(prompts)} prompts → prompts/")
