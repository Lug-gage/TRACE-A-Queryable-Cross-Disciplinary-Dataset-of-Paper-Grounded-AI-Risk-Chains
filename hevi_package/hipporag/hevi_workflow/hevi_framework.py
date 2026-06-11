"""
HEVI framework definitions — grounded in Turner et al. (2003) vulnerability analysis.
https://www.pnas.org/doi/10.1073/pnas.1231335100

Turner defines vulnerability of coupled human-environment systems as a function of:
  Exposure (what faces the hazard)
  × Sensitivity (how easily affected)
  × Resilience/Coping Capacity (ability to respond and adapt)

HEVI adapts this for AI risk assessment with six slots:
"""

HEVI_FRAMEWORK = """
HEVI RISK FRAMEWORK (adapted from Turner et al. 2003 vulnerability analysis):

A coupled human-environment system faces HAZARDS (perturbations/stresses) that act on
EXPOSED elements (who/what is in harm's way). The system's SENSITIVITY (VULNERABILITY)
determines how easily harm occurs, while COPING CAPACITY (KEY CONTROL NODES) provides
response options at multiple levels — individual, institutional, and policy.
The DOSE-RESPONSE relationship translates hazard scale/intensity into IMPACT magnitude.

SIX HEVI SLOTS:

1. HAZARD — A perturbation or stress introduced or amplified by the paper's method.
   - A technical capability, behavior, or mechanism that could cause harm.
   - Hazards can originate INSIDE the system (method-introduced) or OUTSIDE (amplified by use).
   - State at capability level — not narrowed to a single scenario.
   - ≤30 words. Single sentence. NOT the problem the paper solves.

2. EXPOSURE — Who or what system elements face the hazard.
   - Specific people, groups, systems, or coupled system components in harm's way.
   - EXPOSURE ≠ SENSITIVITY: exposure is WHO faces it; sensitivity (vulnerability) is HOW easily affected.
   - Format: 'who' + 'doing what / in what setting'. ≤15 words.
   - No vulnerability conditions. No impact descriptions.

3. DOSE_RESPONSE — How scale, frequency, or intensity of the hazard translates into impact magnitude.
   - The causal translation: more/faster/wider deployment → more/worse harm.
   - Almost never discussed by authors in impact statements.
   - ≤40 words. Single sentence.

4. VULNERABILITY — Conditions that increase the system's SENSITIVITY to the hazard.
   - Gaps, weaknesses, or system properties that make harm MORE LIKELY or MORE SEVERE.
   - Operates at multiple scales: individual, institutional, infrastructural, societal.
   - NOT the research gap the paper fills. NOT generic 'lack of regulation'.
   - Each item ≤30 words.

5. IMPACT — Negative social, ethical, economic, or safety CONSEQUENCES.
   - The realized harm when hazard meets exposure under vulnerable conditions.
   - Impacts can ripple through coupled systems: human subsystem, technical subsystem,
     and their interactions (e.g., biased decisions → eroded trust → reduced adoption).
   - NOT benefits. NOT generic categories without specifics.
   - Each item ≤30 words.

6. KEY_CONTROL_NODES — Intervention points that BLOCK or REDUCE the risk chain.
   - MULTI-LEVEL (Turner coping capacity hierarchy):
     (a) Individual/autonomous: user vigilance, adversarial training, input validation.
     (b) Institutional/organizational: audit processes, human-in-the-loop, red-teaming.
     (c) Policy/societal: regulation, standards, ethical review, public code release.
   - Name only, ≤10 words each. NOT the paper's method. NOT research proposals.

CROSS-CUTTING PRINCIPLES:
- PLACE-BASED: vulnerability varies by scenario/context. The same hazard affects different
  "places" (deployment settings) differently. Each chain is anchored in a specific Nexus.
- COUPLED SYSTEM FEEDBACK: human and technical subsystems interact. A response in one
  (e.g., adding a filter) may affect the other (e.g., over-reliance on the filter).
- MULTI-SCALE: consequences operate at individual, organizational, and societal scales
  simultaneously. Analysis should acknowledge scale interactions.
"""

# Short version for injection into system prompts
HEVI_SHORT = (
    "HEVI framework (Turner et al. 2003): Hazard→Exposure→Vulnerability→Impact, "
    "with Key Control Nodes as coping capacity and Dose-Response as scale→impact translation. "
    "Exposure≠Sensitivity. Place-based. Multi-scale. Coupled system feedback."
)
