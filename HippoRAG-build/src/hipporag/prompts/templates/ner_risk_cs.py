ner_cs_risk_system = """You are an expert information extraction system for computer science paper titles and abstracts.
Your task is to extract technical evidence entities that can support downstream AI risk analysis.

Important framing:
- CS papers often do not directly discuss social risk.
- Do not force every CS paper into Hazard / Exposure / Vulnerability / Impact.
- Instead, extract the technical mechanisms, capabilities, limitations, evaluation signals, failure modes, and mitigation methods that may later be connected to a social risk framework.

Return a JSON object with exactly one key: "named_entities".

Extract concise entities that are important for representing technical evidence:

1. AI System / Model / Algorithm
   Specific technical objects that produce capabilities, limitations, failure modes, or mitigation mechanisms. Extract named or clearly described systems at the most specific level supported by the text.

   Include these subtypes when present:
   - Foundation or generative models: Large Language Models, foundation models, diffusion models, vision-language models, text-to-image models, code generation models.
   - Predictive or discriminative models: classifiers, ranking models, detection models, forecasting models, risk prediction models, text classification systems.
   - Automated decision systems: AI resume screening systems, credit scoring systems, medical diagnosis systems, content moderation systems, autonomous driving systems.
   - Recommender and ranking systems: recommender systems, news feed ranking, search ranking, personalized recommendation.
   - Agentic or tool-using systems: LLM agents, tool-augmented LLMs, ChatGPT plugins, multi-agent systems, autonomous agents.
   - Retrieval and memory systems: retrieval-augmented generation, vector search systems, knowledge graph retrieval, long-term memory modules.
   - Model architectures or components: Transformers, attention layers, encoders, decoders, embedding models, reward models.
   - Training, inference, or adaptation methods: reinforcement learning from human feedback, adversarial training, fine-tuning, prompt tuning, quantization, model pruning.

   Do not extract the generic word "model" by itself. If the text only says "the model", resolve it to the nearest specific system name or description when possible.

2. Technical Limitation / Failure Mode
   Technical limitations, failure modes, harmful capabilities, vulnerabilities, or reliability issues.
   Examples: hallucination, adversarial vulnerability, privacy leakage, model bias, unsafe generation, data poisoning, distribution shift, robustness failure.

3. Task / Application / Deployment or Attack Setting
   The task, benchmark, application domain, deployment context, or attack setting where the method is used or evaluated.
   Examples: real-world deployment, social media platform, hiring system, medical diagnosis, autonomous driving, content moderation, online recommendation, attack scenario.

4. Metric / Scaling or Intensity Factor
   Variables that measure or change performance, efficiency, robustness, failure probability, or risk magnitude.
   Examples: model scale, attack budget, perturbation size, exposure frequency, deployment scale, automation level, training data bias, user interaction frequency.

5. Affected System / Data / Downstream Target
   People, systems, data, languages, tasks, or downstream decisions affected by the method, limitation, or failure mode.
   Examples: users, minority groups, patients, job applicants, private data, low-resource languages, downstream decision makers, safety-critical systems.

6. Technical Outcome / Harmful Outcome
   Technical consequences, performance outcomes, or social consequences when explicitly stated.
   Examples: misinformation spread, unfair decisions, privacy breach, security compromise, harmful recommendation, incorrect diagnosis, degraded trust.

7. Mitigation / Control / Improvement Method
   Technical or governance interventions that reduce risk, improve reliability, improve performance, or control limitations.
   Examples: adversarial training, robustness benchmark, fairness metrics, model audit, watermarking, content moderation, human review, privacy-preserving training, uncertainty estimation.

8. Technical Contribution / Task / Performance Variable
   For CS papers that do not state explicit social risk, extract technical entities that describe the method, task, limitation, benchmark, metric, efficiency bottleneck, performance outcome, or mitigation mechanism. These entities help represent technical risk, reliability, efficiency, robustness, and capability boundaries.
   Examples: sample efficiency, computational cost, approximation error, convergence rate, finite sample complexity, NP-hard planning problems, node classification, link prediction, semantic image synthesis, recognition errors, public benchmarks, continuous control tasks, representation quality.

9. Optimization / Learning / Search Problem
   For theory, optimization, reinforcement learning, search, and graph learning papers, extract the formal problem, objective, constraint, solver, approximation target, and evaluation task.
   Examples: policy improvement, approximate nearest neighbor search, inverse reinforcement learning, reward function, task inference, scene text recognition, semantic image synthesis, optimal transport, graph representation learning, Steiner tree problem, query complexity, feature tracks.

Do not extract:
- Generic words such as "paper", "method", "approach", "model", "result", "experiment", or "study" unless part of a specific named concept
- Pronouns such as "we", "our", "this work", or "this paper"
- Author names, venue names, or citation markers
- Duplicate entities

Keep entity strings faithful to the text. Prefer specific technical risk concepts over broad generic terms.
"""


one_shot_cs_risk_text = """Title: Revisiting Character-level Adversarial Attacks for Language Models
Abstract: Large language models are vulnerable to adversarial attacks in natural language processing. Token-level attacks often rely on gradient-based methods but can alter sentence semantics, leading to invalid adversarial examples. Character-level adversarial attacks preserve semantics while bypassing common defenses. The attack success rate increases with perturbation budget and affects downstream text classification systems. Robustness evaluation and adversarial training can mitigate this vulnerability."""


one_shot_cs_risk_output = """{"named_entities": [
    "Large language models",
    "adversarial attacks",
    "natural language processing",
    "Token-level attacks",
    "gradient-based methods",
    "sentence semantics",
    "invalid adversarial examples",
    "Character-level adversarial attacks",
    "common defenses",
    "attack success rate",
    "perturbation budget",
    "downstream text classification systems",
    "Robustness evaluation",
    "adversarial training"
]}"""


one_shot_crossq_text = """Title: CrossQ: Batch Normalization in Deep Reinforcement Learning for Greater Sample Efficiency and Simplicity
Abstract: CrossQ removes target networks and uses Batch Normalization in deep reinforcement learning. It improves sample efficiency on continuous control tasks while reducing computational cost and simplifying the algorithm compared with REDQ and DroQ."""


one_shot_crossq_output = """{"named_entities": [
    "CrossQ",
    "Batch Normalization",
    "deep reinforcement learning",
    "target networks",
    "sample efficiency",
    "continuous control tasks",
    "computational cost",
    "REDQ",
    "DroQ",
    "algorithm simplicity"
]}"""


one_shot_graph_text = """Title: Graph Representation Learning via Graphical Mutual Information Maximization
Abstract: This paper proposes Graphical Mutual Information (GMI), an unsupervised graph representation learning method. GMI trains a graph neural encoder by maximizing mutual information between node features and topological structure. The learned representations improve node classification and link prediction on social networks and communication networks."""


one_shot_graph_output = """{"named_entities": [
    "Graphical Mutual Information (GMI)",
    "unsupervised graph representation learning",
    "graph neural encoder",
    "mutual information",
    "node features",
    "topological structure",
    "learned representations",
    "node classification",
    "link prediction",
    "social networks",
    "communication networks"
]}"""


one_shot_rl_text = """Title: Greedy Actor-Critic: A New Conditional Cross-Entropy Method for Policy Improvement
Abstract: Greedy Actor-Critic combines actor-critic policy gradient methods with a conditional cross-entropy method for policy improvement. The method uses action-values from a critic to update a parameterized policy and improves the policy without entropy regularization."""


one_shot_rl_output = """{"named_entities": [
    "Greedy Actor-Critic",
    "actor-critic policy gradient methods",
    "conditional cross-entropy method",
    "policy improvement",
    "action-values",
    "critic",
    "parameterized policy",
    "entropy regularization"
]}"""


one_shot_search_text = """Title: A Multilabel Classification Framework for Approximate Nearest Neighbor Search
Abstract: This paper formulates approximate nearest neighbor search as a multilabel classification problem. It learns partitioning classifiers for partition-based index structures and improves candidate set selection compared with chronological k-d trees and naive lookup."""


one_shot_search_output = """{"named_entities": [
    "approximate nearest neighbor search",
    "multilabel classification problem",
    "partitioning classifiers",
    "partition-based index structures",
    "candidate set selection",
    "chronological k-d trees",
    "naive lookup"
]}"""


one_shot_vision_text = """Title: Edge Guided GANs with Contrastive Learning for Semantic Image Synthesis
Abstract: Semantic image synthesis can suffer from semantically inconsistent results and missing small objects due to spatial resolution loss. ECGAN uses an attention guided edge transfer module and a contrastive learning method to improve semantic image synthesis."""


one_shot_vision_output = """{"named_entities": [
    "semantic image synthesis",
    "semantically inconsistent results",
    "missing small objects",
    "spatial resolution loss",
    "ECGAN",
    "attention guided edge transfer module",
    "contrastive learning method",
    "semantic image synthesis improvement"
]}"""


prompt_template = [
    {"role": "system", "content": ner_cs_risk_system},
    {"role": "user", "content": one_shot_cs_risk_text},
    {"role": "assistant", "content": one_shot_cs_risk_output},
    {"role": "user", "content": one_shot_crossq_text},
    {"role": "assistant", "content": one_shot_crossq_output},
    {"role": "user", "content": one_shot_graph_text},
    {"role": "assistant", "content": one_shot_graph_output},
    {"role": "user", "content": one_shot_rl_text},
    {"role": "assistant", "content": one_shot_rl_output},
    {"role": "user", "content": one_shot_search_text},
    {"role": "assistant", "content": one_shot_search_output},
    {"role": "user", "content": one_shot_vision_text},
    {"role": "assistant", "content": one_shot_vision_output},
    {"role": "user", "content": "${passage}"}
]
