from .ner_risk_ss import one_shot_risk_text, one_shot_risk_output
from ...utils.llm_utils import convert_format_to_template


risk_re_system = """You are an expert research knowledge graph extraction system for AI risk analysis.
Your task is to convert an academic paper title and abstract into triples organized by this risk framework:

- Hazard
- Exposure
- Dose-Response
- Vulnerability
- Impact
- Key Control Node

Return a JSON object with exactly one key: "triples".
Each triple must be a list of three strings:
["subject", "predicate", "object"]

Use triples to capture the risk chain:
AI system or capability -> Hazard -> Exposure -> Dose-Response -> Vulnerability -> Impact,
plus Key Control Nodes that intervene in the chain.

For social science papers, also extract variable relationships that explain AI risk, trust, adoption, fairness, accountability, privacy, transparency, user perception, institutional legitimacy, and behavior change. These relationships are valid risk-framework triples even when the paper does not describe a direct physical or cybersecurity harm.

Preferred predicates:
- creates_hazard
- has_hazard
- uses
- occurs_in
- exposes
- exposed_to
- affects
- affects_group
- has_dose_metric
- has_dose_response
- dose_increases
- dose_decreases
- amplifies
- increases
- decreases
- increases_risk_of
- reduces_risk_of
- vulnerable_to
- leads_to
- causes
- mitigates
- controlled_by
- control_targets
- reduces_exposure
- reduces_dose
- reduces_vulnerability
- reduces_impact
- measured_by
- moderates
- mediates
- predicts
- increases_trust
- decreases_trust
- increases_adoption
- decreases_adoption

Requirements:
- Each triple must be directly supported by the text.
- Each triple should contain at least one, but preferably two, of the named entities in the provided named entity list.
- If a necessary concept is missing from the named entity list, you may use a concise concept from the passage, but still try to keep at least one side of the triple anchored to the named entity list.
- Resolve pronouns such as "it", "they", "this system", and "this approach" to specific entities.
- Use concise predicates, preferably from the predicate list above.
- Do not invent hazards, impacts, vulnerable groups, or interventions not stated in the text.
- Do not include vague triples such as ["paper", "studies", "risk"].
- For AI-related social science abstracts, aim to extract 3 to 8 high-quality triples when the text contains relationships among AI systems, user perceptions, trust, fairness, privacy, adoption, governance, or behavioral outcomes.
- Return {"triples": []} only when the abstract contains no AI-related risk, trust, governance, fairness, privacy, adoption, or user-impact relationship.
"""


risk_re_frame = """Convert the paper text into a JSON dict containing risk-framework triples.

Paper text:
```
{passage}
```

Named entities:
{named_entity_json}
"""


risk_re_input = risk_re_frame.format(
    passage=one_shot_risk_text,
    named_entity_json=one_shot_risk_output
)


risk_re_output = """{"triples": [
    ["AI resume screening systems", "creates_hazard", "embedded historical bias"],
    ["AI resume screening systems", "uses", "historical hiring data"],
    ["embedded historical bias", "occurs_in", "large-scale recruitment"],
    ["job applicants", "exposed_to", "automated resume review"],
    ["automated resume review", "reduces", "human assessment"],
    ["proportion of AI-processed applications", "has_dose_response", "wrongful exclusion"],
    ["model bias level", "increases_risk_of", "wrongful exclusion"],
    ["wrongful exclusion", "affects_group", "marginalized applicants"],
    ["Women", "vulnerable_to", "wrongful exclusion"],
    ["minority groups", "vulnerable_to", "wrongful exclusion"],
    ["low-income applicants", "vulnerable_to", "wrongful exclusion"],
    ["unavailable appeal channels", "amplifies", "wrongful exclusion"],
    ["wrongful exclusion", "leads_to", "labor market inequality"],
    ["wrongful exclusion", "leads_to", "structural discrimination"],
    ["Model audits", "mitigates", "embedded historical bias"],
    ["fairness metrics", "measured_by", "model bias level"],
    ["mandatory human review", "reduces_risk_of", "wrongful exclusion"]
]}"""


one_shot_trust_text = """Title: Trust in artificial intelligence: a survey experiment to assess trust in algorithmic decision-making
Abstract: This study uses a pre-registered survey experiment embedded in an online survey to assess trust in AI-based Automated Decision-Making (ADM). Participants were presented with hypothetical decisions in medical diagnoses, hiring, transportation, and financial investments. Decisions supported by human intervention were perceived as more trustworthy than decisions supported by AI-based ADM. Low understanding of AI, high privacy concerns, and closed personality were associated with lower trust in AI, while good understanding of AI and low privacy concerns were associated with higher trust."""


one_shot_trust_entities = """{"named_entities": [
    "AI-based Automated Decision-Making (ADM)",
    "trust in AI",
    "human intervention",
    "medical diagnoses",
    "hiring",
    "transportation",
    "financial investments",
    "low understanding of AI",
    "high privacy concerns",
    "closed personality",
    "good understanding of AI",
    "low privacy concerns"
]}"""


one_shot_trust_output = """{"triples": [
    ["AI-based Automated Decision-Making (ADM)", "occurs_in", "medical diagnoses"],
    ["AI-based Automated Decision-Making (ADM)", "occurs_in", "hiring"],
    ["AI-based Automated Decision-Making (ADM)", "occurs_in", "transportation"],
    ["AI-based Automated Decision-Making (ADM)", "occurs_in", "financial investments"],
    ["human intervention", "increases_trust", "trust in AI"],
    ["low understanding of AI", "decreases_trust", "trust in AI"],
    ["high privacy concerns", "decreases_trust", "trust in AI"],
    ["closed personality", "decreases_trust", "trust in AI"],
    ["good understanding of AI", "increases_trust", "trust in AI"],
    ["low privacy concerns", "increases_trust", "trust in AI"]
]}"""


one_shot_disclosure_text = """Title: Machines vs. Humans: The Impact of Artificial Intelligence Chatbot Disclosure on Customer Purchases
Abstract: This field experiment investigates how chatbot disclosure affects customer purchase behavior. Customers exposed to chatbot disclosure showed reduced purchase rates and shorter calls. The negative disclosure effect was amplified by subjective human perception and was stronger among customers without prior AI experience. Late disclosure timing and customer prior AI experience mitigated the negative disclosure effect."""


one_shot_disclosure_entities = """{"named_entities": [
    "chatbot disclosure",
    "customer purchase behavior",
    "customers",
    "reduced purchase rates",
    "shorter calls",
    "negative disclosure effect",
    "subjective human perception",
    "customers without prior AI experience",
    "late disclosure timing",
    "customer prior AI experience"
]}"""


one_shot_disclosure_output = """{"triples": [
    ["customers", "exposed_to", "chatbot disclosure"],
    ["chatbot disclosure", "decreases", "customer purchase behavior"],
    ["chatbot disclosure", "causes", "reduced purchase rates"],
    ["chatbot disclosure", "causes", "shorter calls"],
    ["subjective human perception", "amplifies", "negative disclosure effect"],
    ["customers without prior AI experience", "vulnerable_to", "negative disclosure effect"],
    ["late disclosure timing", "mitigates", "negative disclosure effect"],
    ["customer prior AI experience", "mitigates", "negative disclosure effect"]
]}"""


one_shot_transparency_text = """Title: Transparency and trust in artificial intelligence systems
Abstract: A behavioral experiment examines how transparency and explanations of AI decisions affect decision makers using an ML-based decision support tool for text classification. The study finds that explanations can shape trust in AI predictions and alter reliance on assistive AI technology."""


one_shot_transparency_entities = """{"named_entities": [
    "transparency",
    "explanations of AI decisions",
    "decision makers",
    "ML-based decision support tool",
    "text classification",
    "trust in AI predictions",
    "reliance on assistive AI technology",
    "assistive AI technology"
]}"""


one_shot_transparency_output = """{"triples": [
    ["ML-based decision support tool", "occurs_in", "text classification"],
    ["decision makers", "exposed_to", "ML-based decision support tool"],
    ["transparency", "increases_trust", "trust in AI predictions"],
    ["explanations of AI decisions", "affects", "trust in AI predictions"],
    ["trust in AI predictions", "affects", "reliance on assistive AI technology"],
    ["explanations of AI decisions", "controlled_by", "transparency"]
]}"""


one_shot_xai_text = """Title: Explainable Artificial Intelligence for facilitating recognition of algorithmic bias
Abstract: This experiment investigates whether an explanation by example XAI approach helps users recognize algorithmic bias caused by non-inclusive datasets. Explanatory examples resembling user input increase perceived incongruence, perceptions of unfairness and exclusion, and user awareness of algorithmic bias. Prior experience with discrimination strengthens the effect, while explanations reduce blind trust in the system."""


one_shot_xai_entities = """{"named_entities": [
    "explanation by example XAI approach",
    "algorithmic bias",
    "non-inclusive datasets",
    "explanatory examples resembling user input",
    "perceived incongruence",
    "perceptions of unfairness and exclusion",
    "user awareness of algorithmic bias",
    "prior experience with discrimination",
    "blind trust in the system"
]}"""


one_shot_xai_output = """{"triples": [
    ["non-inclusive datasets", "creates_hazard", "algorithmic bias"],
    ["explanation by example XAI approach", "mitigates", "blind trust in the system"],
    ["explanatory examples resembling user input", "increases", "perceived incongruence"],
    ["perceived incongruence", "increases", "perceptions of unfairness and exclusion"],
    ["explanatory examples resembling user input", "increases", "user awareness of algorithmic bias"],
    ["prior experience with discrimination", "moderates", "user awareness of algorithmic bias"]
]}"""


one_shot_public_ai_text = """Title: Artificial intelligence vs. public administrators: Public trust, efficiency, and tolerance for error
Abstract: This study examines public reactions to AI mistakes in the public sector. Citizens compare computational-audit scenarios and conversational-advisory scenarios. Improved efficiency can increase support for AI in public administration, but tolerance for errors depends on scenario type. Context-sensitive design approaches and governance practices can reduce distrust in algorithmic governance."""


one_shot_public_ai_entities = """{"named_entities": [
    "AI mistakes",
    "public sector AI",
    "public trust",
    "computational-audit scenarios",
    "conversational-advisory scenarios",
    "improved efficiency",
    "support for AI in public administration",
    "tolerance for errors",
    "scenario type",
    "context-sensitive design approaches",
    "governance practices",
    "distrust in algorithmic governance"
]}"""


one_shot_public_ai_output = """{"triples": [
    ["AI mistakes", "decreases_trust", "public trust"],
    ["AI mistakes", "occurs_in", "public sector AI"],
    ["computational-audit scenarios", "moderates", "tolerance for errors"],
    ["conversational-advisory scenarios", "moderates", "tolerance for errors"],
    ["improved efficiency", "increases_adoption", "support for AI in public administration"],
    ["context-sensitive design approaches", "mitigates", "distrust in algorithmic governance"],
    ["governance practices", "mitigates", "distrust in algorithmic governance"]
]}"""


one_shot_info_text = """Title: The effect of information seeking behaviour on trust in AI in Asia
Abstract: This study examines how information seeking behaviour through social media and legacy media affects trust in AI. Concern about misinformation moderates the relationship between information seeking and trust in AI across Asian societies."""


one_shot_info_entities = """{"named_entities": [
    "information seeking behaviour",
    "social media",
    "legacy media",
    "trust in AI",
    "concern about misinformation",
    "misinformation",
    "Asian societies"
]}"""


one_shot_info_output = """{"triples": [
    ["information seeking behaviour", "affects", "trust in AI"],
    ["social media", "occurs_in", "information seeking behaviour"],
    ["legacy media", "occurs_in", "information seeking behaviour"],
    ["concern about misinformation", "moderates", "relationship between information seeking behaviour and trust in AI"],
    ["misinformation", "creates_hazard", "decreased trust in AI"],
    ["Asian societies", "exposed_to", "misinformation"]
]}"""


prompt_template = [
    {"role": "system", "content": risk_re_system},
    {"role": "user", "content": risk_re_input},
    {"role": "assistant", "content": risk_re_output},
    {"role": "user", "content": risk_re_frame.format(passage=one_shot_trust_text, named_entity_json=one_shot_trust_entities)},
    {"role": "assistant", "content": one_shot_trust_output},
    {"role": "user", "content": risk_re_frame.format(passage=one_shot_disclosure_text, named_entity_json=one_shot_disclosure_entities)},
    {"role": "assistant", "content": one_shot_disclosure_output},
    {"role": "user", "content": risk_re_frame.format(passage=one_shot_transparency_text, named_entity_json=one_shot_transparency_entities)},
    {"role": "assistant", "content": one_shot_transparency_output},
    {"role": "user", "content": risk_re_frame.format(passage=one_shot_xai_text, named_entity_json=one_shot_xai_entities)},
    {"role": "assistant", "content": one_shot_xai_output},
    {"role": "user", "content": risk_re_frame.format(passage=one_shot_public_ai_text, named_entity_json=one_shot_public_ai_entities)},
    {"role": "assistant", "content": one_shot_public_ai_output},
    {"role": "user", "content": risk_re_frame.format(passage=one_shot_info_text, named_entity_json=one_shot_info_entities)},
    {"role": "assistant", "content": one_shot_info_output},
    {"role": "user", "content": convert_format_to_template(original_string=risk_re_frame, placeholder_mapping=None, static_values=None)}
]
