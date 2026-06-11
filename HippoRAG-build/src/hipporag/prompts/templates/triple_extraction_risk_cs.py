from .ner_risk_cs import one_shot_cs_risk_text, one_shot_cs_risk_output
from ...utils.llm_utils import convert_format_to_template


cs_risk_re_system = """You are an expert research knowledge graph extraction system for computer science papers.
Your task is to convert a CS paper title and abstract into triples that capture technical evidence for downstream AI risk analysis.

Important framing:
- CS papers often provide technical evidence rather than direct social-risk claims.
- Do not force every CS paper into a full Hazard / Exposure / Vulnerability / Impact chain.
- Extract method-task-metric-limitation-improvement relationships first.
- Extract explicit risk, security, privacy, bias, robustness, safety, or failure-mode relationships when they are stated.

The extracted triples should support these evidence roles:
- AI System / Model / Algorithm
- Method / Technical Mechanism
- Task / Application / Benchmark
- Metric / Performance Variable
- Technical Limitation / Failure Mode
- Mitigation / Control / Improvement Method
- Explicit Risk / Impact when stated

Return a JSON object with exactly one key: "triples".
Each triple must be a list of three strings:
["subject", "predicate", "object"]

Use triples to capture technical evidence chains:
method/model -> applies_to -> task/application
method/model -> improves/reduces -> metric/cost/error
method/model -> evaluated_on -> benchmark/dataset
limitation/failure mode -> affects -> task/performance
mitigation/control method -> mitigates/reduces -> limitation/failure mode
AI system -> creates_hazard/has_failure_mode -> explicit risk when stated

For CS papers that do not describe explicit social risk, also extract technical reliability and performance relationships. Treat limitations, approximation error, computational cost, sample efficiency, convergence, robustness, recognition error, bias, scalability, and task performance as technical risk or control-relevant variables.

When a CS abstract mainly describes an algorithm, theory result, optimization problem, search method, reinforcement learning method, vision model, or representation learning method, still extract method-task-performance triples. Do not return an empty triple list merely because the paper is not about social harm.

Preferred predicates:
- creates_hazard
- has_failure_mode
- has_limitation
- is_vulnerable_to
- causes
- increases_risk_of
- amplifies
- occurs_in
- deployed_in
- exposed_to
- affects
- harms
- improves
- reduces
- increases
- measured_by
- evaluated_on
- uses
- proposes
- applies_to
- leads_to
- triggered_by
- depends_on
- mitigated_by
- mitigates
- reduced_by
- controlled_by
- improves_robustness_against
- improves_efficiency_of
- improves_accuracy_of
- reduces_cost_of
- reduces_error_in
- reduces_bias_in
- reduces_privacy_risk
- reduces_harm_to

Requirements:
- Each triple must be directly supported by the text.
- Each triple should contain at least one, but preferably two, of the named entities in the provided named entity list.
- If a necessary concept is missing from the named entity list, you may use a concise concept from the passage, but still try to keep at least one side of the triple anchored to the named entity list.
- Resolve pronouns such as "it", "they", "this system", "this attack", and "this method" to specific entities.
- Use concise predicates, preferably from the predicate list above.
- Do not invent hazards, impacts, datasets, metrics, affected groups, or interventions not stated in the text.
- Do not include vague triples such as ["paper", "studies", "risk"].
- For CS abstracts, aim to extract 3 to 8 high-quality triples when the text contains relationships among methods, tasks, limitations, metrics, benchmarks, performance, robustness, efficiency, or mitigation mechanisms.
- Return {"triples": []} only when the abstract contains no clear method-task, method-metric, limitation-mitigation, or technical risk/performance relationship.
"""


cs_risk_re_frame = """Convert the CS paper text into a JSON dict containing technical risk-framework triples.

Paper text:
```
{passage}
```

Named entities:
{named_entity_json}
"""


cs_risk_re_input = cs_risk_re_frame.format(
    passage=one_shot_cs_risk_text,
    named_entity_json=one_shot_cs_risk_output
)


cs_risk_re_output = """{"triples": [
    ["Large language models", "is_vulnerable_to", "adversarial attacks"],
    ["adversarial attacks", "occurs_in", "natural language processing"],
    ["Token-level attacks", "uses", "gradient-based methods"],
    ["Token-level attacks", "causes", "altered sentence semantics"],
    ["altered sentence semantics", "leads_to", "invalid adversarial examples"],
    ["Character-level adversarial attacks", "preserves", "sentence semantics"],
    ["Character-level adversarial attacks", "bypasses", "common defenses"],
    ["perturbation budget", "amplifies", "attack success rate"],
    ["Character-level adversarial attacks", "affects", "downstream text classification systems"],
    ["Robustness evaluation", "measured_by", "attack success rate"],
    ["Character-level adversarial attacks", "mitigated_by", "adversarial training"],
    ["adversarial training", "improves_robustness_against", "adversarial attacks"]
]}"""


one_shot_crossq_text = """Title: CrossQ: Batch Normalization in Deep Reinforcement Learning for Greater Sample Efficiency and Simplicity
Abstract: CrossQ removes target networks and uses Batch Normalization in deep reinforcement learning. It improves sample efficiency on continuous control tasks while reducing computational cost and simplifying the algorithm compared with REDQ and DroQ."""


one_shot_crossq_entities = """{"named_entities": [
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


one_shot_crossq_output = """{"triples": [
    ["CrossQ", "uses", "Batch Normalization"],
    ["CrossQ", "applies_to", "deep reinforcement learning"],
    ["CrossQ", "reduces", "target networks"],
    ["CrossQ", "improves", "sample efficiency"],
    ["CrossQ", "evaluated_on", "continuous control tasks"],
    ["CrossQ", "reduces_cost_of", "computational cost"],
    ["CrossQ", "improves", "algorithm simplicity"],
    ["CrossQ", "compares_to", "REDQ"],
    ["CrossQ", "compares_to", "DroQ"]
]}"""


one_shot_gnn_text = """Title: Learning NP-Hard Multi-Agent Assignment Planning using GNN
Abstract: This paper applies Graph Neural Networks to multi-agent assignment planning, including multi-robot reward collection and ride-sharing. The method uses mean-field inference and an order-transferable Q-function estimator to solve NP-hard planning problems and improves generalization to larger random graphs."""


one_shot_gnn_entities = """{"named_entities": [
    "Graph Neural Networks",
    "multi-agent assignment planning",
    "multi-robot reward collection",
    "ride-sharing",
    "mean-field inference",
    "order-transferable Q-function estimator",
    "NP-hard planning problems",
    "larger random graphs",
    "generalization"
]}"""


one_shot_gnn_output = """{"triples": [
    ["Graph Neural Networks", "applies_to", "multi-agent assignment planning"],
    ["multi-agent assignment planning", "includes", "multi-robot reward collection"],
    ["multi-agent assignment planning", "includes", "ride-sharing"],
    ["NP-hard planning problems", "creates_hazard", "computational difficulty"],
    ["mean-field inference", "mitigates", "NP-hard planning problems"],
    ["order-transferable Q-function estimator", "improves", "generalization"],
    ["generalization", "evaluated_on", "larger random graphs"]
]}"""


one_shot_graph_text = """Title: Graph Representation Learning via Graphical Mutual Information Maximization
Abstract: This paper proposes Graphical Mutual Information (GMI), an unsupervised graph representation learning method. GMI trains a graph neural encoder by maximizing mutual information between node features and topological structure. The learned representations improve node classification and link prediction on social networks and communication networks."""


one_shot_graph_entities = """{"named_entities": [
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


one_shot_graph_output = """{"triples": [
    ["Graphical Mutual Information (GMI)", "proposes", "unsupervised graph representation learning"],
    ["Graphical Mutual Information (GMI)", "uses", "graph neural encoder"],
    ["Graphical Mutual Information (GMI)", "depends_on", "mutual information"],
    ["mutual information", "connects", "node features"],
    ["mutual information", "connects", "topological structure"],
    ["learned representations", "improves_accuracy_of", "node classification"],
    ["learned representations", "improves_accuracy_of", "link prediction"],
    ["Graphical Mutual Information (GMI)", "evaluated_on", "social networks"],
    ["Graphical Mutual Information (GMI)", "evaluated_on", "communication networks"]
]}"""


one_shot_rl_text = """Title: Greedy Actor-Critic: A New Conditional Cross-Entropy Method for Policy Improvement
Abstract: Greedy Actor-Critic combines actor-critic policy gradient methods with a conditional cross-entropy method for policy improvement. The method uses action-values from a critic to update a parameterized policy and improves the policy without entropy regularization."""


one_shot_rl_entities = """{"named_entities": [
    "Greedy Actor-Critic",
    "actor-critic policy gradient methods",
    "conditional cross-entropy method",
    "policy improvement",
    "action-values",
    "critic",
    "parameterized policy",
    "entropy regularization"
]}"""


one_shot_rl_output = """{"triples": [
    ["Greedy Actor-Critic", "extends", "actor-critic policy gradient methods"],
    ["Greedy Actor-Critic", "uses", "conditional cross-entropy method"],
    ["conditional cross-entropy method", "applies_to", "policy improvement"],
    ["critic", "produces", "action-values"],
    ["action-values", "updates", "parameterized policy"],
    ["Greedy Actor-Critic", "reduces", "entropy regularization"],
    ["Greedy Actor-Critic", "improves", "policy improvement"]
]}"""


one_shot_search_text = """Title: A Multilabel Classification Framework for Approximate Nearest Neighbor Search
Abstract: This paper formulates approximate nearest neighbor search as a multilabel classification problem. It learns partitioning classifiers for partition-based index structures and improves candidate set selection compared with chronological k-d trees and naive lookup."""


one_shot_search_entities = """{"named_entities": [
    "approximate nearest neighbor search",
    "multilabel classification problem",
    "partitioning classifiers",
    "partition-based index structures",
    "candidate set selection",
    "chronological k-d trees",
    "naive lookup"
]}"""


one_shot_search_output = """{"triples": [
    ["approximate nearest neighbor search", "modeled_as", "multilabel classification problem"],
    ["partitioning classifiers", "applies_to", "partition-based index structures"],
    ["partitioning classifiers", "improves", "candidate set selection"],
    ["candidate set selection", "affects", "approximate nearest neighbor search"],
    ["partitioning classifiers", "compares_to", "chronological k-d trees"],
    ["partitioning classifiers", "compares_to", "naive lookup"]
]}"""


one_shot_irl_text = """Title: Active Task-Inference-Guided Deep Inverse Reinforcement Learning
Abstract: Active Task-Inference-Guided Deep Inverse Reinforcement Learning learns reward functions for temporally extended tasks. It uses a task inference module to identify subgoals and a reward learning module to improve inverse reinforcement learning in Markov decision processes."""


one_shot_irl_entities = """{"named_entities": [
    "Active Task-Inference-Guided Deep Inverse Reinforcement Learning",
    "inverse reinforcement learning",
    "reward functions",
    "temporally extended tasks",
    "task inference module",
    "subgoals",
    "reward learning module",
    "Markov decision processes"
]}"""


one_shot_irl_output = """{"triples": [
    ["Active Task-Inference-Guided Deep Inverse Reinforcement Learning", "applies_to", "inverse reinforcement learning"],
    ["inverse reinforcement learning", "learns", "reward functions"],
    ["reward functions", "occurs_in", "Markov decision processes"],
    ["temporally extended tasks", "depends_on", "subgoals"],
    ["task inference module", "identifies", "subgoals"],
    ["reward learning module", "improves", "inverse reinforcement learning"]
]}"""


one_shot_vision_text = """Title: Edge Guided GANs with Contrastive Learning for Semantic Image Synthesis
Abstract: Semantic image synthesis can suffer from semantically inconsistent results and missing small objects due to spatial resolution loss. ECGAN uses an attention guided edge transfer module and a contrastive learning method to improve semantic image synthesis."""


one_shot_vision_entities = """{"named_entities": [
    "semantic image synthesis",
    "semantically inconsistent results",
    "missing small objects",
    "spatial resolution loss",
    "ECGAN",
    "attention guided edge transfer module",
    "contrastive learning method"
]}"""


one_shot_vision_output = """{"triples": [
    ["semantic image synthesis", "has_failure_mode", "semantically inconsistent results"],
    ["semantic image synthesis", "has_failure_mode", "missing small objects"],
    ["spatial resolution loss", "causes", "semantically inconsistent results"],
    ["ECGAN", "uses", "attention guided edge transfer module"],
    ["attention guided edge transfer module", "reduces", "spatial resolution loss"],
    ["ECGAN", "uses", "contrastive learning method"],
    ["contrastive learning method", "improves", "semantic image synthesis"]
]}"""


prompt_template = [
    {"role": "system", "content": cs_risk_re_system},
    {"role": "user", "content": cs_risk_re_input},
    {"role": "assistant", "content": cs_risk_re_output},
    {"role": "user", "content": cs_risk_re_frame.format(passage=one_shot_crossq_text, named_entity_json=one_shot_crossq_entities)},
    {"role": "assistant", "content": one_shot_crossq_output},
    {"role": "user", "content": cs_risk_re_frame.format(passage=one_shot_gnn_text, named_entity_json=one_shot_gnn_entities)},
    {"role": "assistant", "content": one_shot_gnn_output},
    {"role": "user", "content": cs_risk_re_frame.format(passage=one_shot_graph_text, named_entity_json=one_shot_graph_entities)},
    {"role": "assistant", "content": one_shot_graph_output},
    {"role": "user", "content": cs_risk_re_frame.format(passage=one_shot_rl_text, named_entity_json=one_shot_rl_entities)},
    {"role": "assistant", "content": one_shot_rl_output},
    {"role": "user", "content": cs_risk_re_frame.format(passage=one_shot_search_text, named_entity_json=one_shot_search_entities)},
    {"role": "assistant", "content": one_shot_search_output},
    {"role": "user", "content": cs_risk_re_frame.format(passage=one_shot_irl_text, named_entity_json=one_shot_irl_entities)},
    {"role": "assistant", "content": one_shot_irl_output},
    {"role": "user", "content": cs_risk_re_frame.format(passage=one_shot_vision_text, named_entity_json=one_shot_vision_entities)},
    {"role": "assistant", "content": one_shot_vision_output},
    {"role": "user", "content": convert_format_to_template(original_string=cs_risk_re_frame, placeholder_mapping=None, static_values=None)}
]
