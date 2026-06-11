from .ner_paper import one_shot_paper_text, one_shot_paper_output
from ...utils.llm_utils import convert_format_to_template


paper_re_system = """You are an expert research knowledge graph extraction system.
Your task is to convert an academic paper title and abstract into research-oriented triples.

Return a JSON object with exactly one key: "triples".
Each triple must be a list of three strings:
["subject", "predicate", "object"]

Use triples that capture the scientific contribution, technical mechanism, evaluation evidence, and potential impact of the paper.

Prefer these predicate types when supported by the text:
- proposes
- introduces
- addresses
- studies
- uses
- extends
- improves
- reduces
- increases
- outperforms
- evaluates_on
- applies_to
- measures
- enables
- depends_on
- causes
- mitigates
- affects
- benefits
- risks
- targets

Requirements:
- Each triple should be directly supported by the text.
- Prefer entities from the provided named entity list as subjects or objects.
- Resolve pronouns such as "it", "they", "this method", and "our approach" to the specific paper entity when possible.
- Use concise predicates, preferably from the predicate list above.
- Do not invent datasets, results, impacts, or claims that are not stated in the text.
- Do not include vague triples such as ["paper", "proposes", "method"].
- If the abstract contains no clear relation, return {"triples": []}.
"""


paper_re_frame = """Convert the paper text into a JSON dict containing research triples.

Paper text:
```
{passage}
```

Named entities:
{named_entity_json}
"""


paper_re_input = paper_re_frame.format(
    passage=one_shot_paper_text,
    named_entity_json=one_shot_paper_output
)


paper_re_output = """{"triples": [
    ["FrameQuant", "proposes", "flexible low-bit quantization"],
    ["FrameQuant", "applies_to", "Transformers"],
    ["Transformers", "are_used_for", "foundation models"],
    ["foundation models", "applies_to", "Vision tasks"],
    ["foundation models", "applies_to", "Natural Language Processing tasks"],
    ["Transformers", "have", "large compute footprint"],
    ["Transformers", "have", "large memory footprint"],
    ["Post-Training Quantization", "quantizes", "pretrained model"],
    ["Post-Training Quantization", "improves", "compute efficiency"],
    ["Post-Training Quantization", "improves", "memory efficiency"],
    ["Post-Training Quantization", "improves", "latency efficiency"],
    ["FrameQuant", "evaluates_on", "language tasks"],
    ["FrameQuant", "evaluates_on", "vision tasks"]
]}"""


prompt_template = [
    {"role": "system", "content": paper_re_system},
    {"role": "user", "content": paper_re_input},
    {"role": "assistant", "content": paper_re_output},
    {"role": "user", "content": convert_format_to_template(original_string=paper_re_frame, placeholder_mapping=None, static_values=None)}
]
