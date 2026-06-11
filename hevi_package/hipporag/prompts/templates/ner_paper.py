ner_paper_system = """You are an expert information extraction system for academic paper titles and abstracts.
Your task is to extract research-relevant named entities from the given paper text.

Return a JSON object with exactly one key: "named_entities".

Extract entities that are useful for building a research knowledge graph, including:
- Methods, algorithms, frameworks, systems, models, and techniques
- Tasks, problems, benchmarks, datasets, metrics, and evaluation settings
- Application domains, affected groups, deployment contexts, risks, benefits, and impact targets
- Important scientific concepts or variables that are central to the paper

Do not extract:
- Generic words such as "paper", "method", "approach", "result", "experiment", or "model" unless part of a specific name
- Pronouns such as "we", "our", "this work", or "this paper"
- Author names, venue names, or citation markers
- Duplicate entities

Keep entity strings concise and faithful to the text. Preserve original capitalization for named methods, datasets, and acronyms.
"""


one_shot_paper_text = """Title: FrameQuant Flexible Low-Bit Quantization for Transformers
Abstract: Transformers are the backbone of powerful foundation models for many Vision and Natural Language Processing tasks. However, their compute and memory footprint is large. Post-Training Quantization modifies a pretrained model and quantizes it to eight bits or lower, improving compute, memory, and latency efficiency. FrameQuant introduces a flexible low-bit quantization method for transformer models and evaluates it on language and vision tasks."""


one_shot_paper_output = """{"named_entities": [
    "FrameQuant",
    "low-bit quantization",
    "Transformers",
    "foundation models",
    "Vision tasks",
    "Natural Language Processing tasks",
    "compute footprint",
    "memory footprint",
    "Post-Training Quantization",
    "pretrained model",
    "latency efficiency",
    "language tasks",
    "vision tasks"
]}"""


prompt_template = [
    {"role": "system", "content": ner_paper_system},
    {"role": "user", "content": one_shot_paper_text},
    {"role": "assistant", "content": one_shot_paper_output},
    {"role": "user", "content": "${passage}"}
]
