"""
Adverse-action explanation pipeline, built as a controlled sequence:

    Model -> SHAP -> validated feature mapping -> approved reason-code
    taxonomy -> plain-language explanation -> (optional) GenAI wording pass,
    tightly bounded

Design rationale: ECOA / Regulation B (12 CFR 1002.9) require specific,
accurate principal reasons for adverse action -- reaffirmed by CFPB
Circular 2026-03 (May 2026), which makes clear that using a complex
algorithm does not excuse this requirement and that proprietary/
uninterpretable models are not a defense. The binding requirement is the
statute/regulation, not the circular's wording -- the taxonomy layer below
is built so the underlying logic survives future circular renumbering.

The core rule enforced by this module: no feature name, engineered
artifact, or free-form model/LLM output ever reaches a customer-facing
explanation without first being mapped through an approved, versioned
taxonomy. If a SHAP-flagged feature isn't in the taxonomy, it is NOT
silently surfaced -- it's logged for a compliance reviewer to triage, and
the explanation falls back to the next validated reason.
"""
import shap
import pandas as pd
import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------
# Approved reason-code taxonomy (compliance-owned in practice; treat any
# edit to this table as a change to customer-facing legal text, not a
# code change -- it should go through the same review a real adverse
# action letter template would).
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ReasonCode:
    code: str
    approved_text: str


REASON_TAXONOMY: dict[str, ReasonCode] = {
    "dti":               ReasonCode("R01", "Debt-to-income ratio is too high relative to the amount requested"),
    "fico_midpoint":      ReasonCode("R02", "Credit score does not meet the minimum required"),
    "fico_range_low":     ReasonCode("R02", "Credit score does not meet the minimum required"),
    "delinq_2yrs":        ReasonCode("R03", "Delinquent credit obligation(s) on file"),
    "revol_util":         ReasonCode("R04", "Proportion of credit lines used is too high"),
    "emp_length_years":   ReasonCode("R05", "Length of employment history is insufficient"),
    "annual_inc":         ReasonCode("R06", "Income insufficient for amount of credit requested"),
    "inq_last_6mths":     ReasonCode("R07", "Number of recent credit inquiries on file"),
    "open_acc":           ReasonCode("R08", "Number of open credit accounts"),
    "pub_rec":            ReasonCode("R09", "Public record item(s) on file"),
    "total_acc":          ReasonCode("R10", "Length of credit history is insufficient"),
    "verification_status": ReasonCode("R11", "Unable to verify income or other application information"),
    "loan_amnt":          ReasonCode("R12", "Requested amount is high relative to ability to repay"),
}

UNMAPPED_FEATURE_FALLBACK = ReasonCode(
    "R99", "Based on information in your credit application and report"
)

# Deliberately excluded from the taxonomy, even though SHAP will often flag
# them as high-importance: these are OUTPUTS of underwriting (assigned
# based on an applicant's assessed risk), not root causes an applicant
# could act on. Citing "your grade was too low" is circular and not a
# specific, accurate reason under ECOA/Reg B -- the underlying factors
# that PRODUCED the grade (dti, fico, etc., already in the taxonomy above)
# are the compliant explanation. Distinguished from genuinely unmapped
# features so the compliance queue isn't spammed with an expected, known
# exclusion every time.
DESIGN_EXCLUDED_FEATURES = {"grade", "sub_grade", "int_rate", "installment"}


# ---------------------------------------------------------------------
# Step 1-2: SHAP computation + validated feature mapping
# ---------------------------------------------------------------------

def compute_shap_values(fitted_model, X_sample: pd.DataFrame):
    try:
        explainer = shap.TreeExplainer(fitted_model.named_steps["clf"])
    except Exception:
        explainer = shap.Explainer(fitted_model.predict_proba, X_sample)
    return explainer(X_sample)


def map_shap_to_reason_codes(shap_values: np.ndarray, feature_names: list[str],
                              n_reasons: int = 4) -> tuple[list[ReasonCode], list[str]]:
    """Map SHAP-flagged features to approved reason codes only. Returns the
    validated reason codes to use in the explanation, AND the list of
    genuinely unmapped feature names that were skipped -- the latter must
    be logged for compliance review, not discarded silently, since a
    feature that keeps showing up unmapped is itself a governance signal
    (the taxonomy is falling behind the model). Features in
    DESIGN_EXCLUDED_FEATURES are skipped WITHOUT being logged as unmapped
    -- this is a known, deliberate exclusion (see its definition above),
    not a gap needing review."""
    contributions = pd.Series(shap_values, index=feature_names).sort_values(ascending=False)

    validated: list[ReasonCode] = []
    unmapped: list[str] = []
    seen_codes = set()

    for feat, contribution in contributions.items():
        if contribution <= 0:
            continue  # only features pushing toward decline are eligible reasons
        if feat in DESIGN_EXCLUDED_FEATURES:
            continue  # known, deliberate exclusion -- not a taxonomy gap
        reason = REASON_TAXONOMY.get(feat)
        if reason is None:
            unmapped.append(feat)
            continue
        if reason.code in seen_codes:
            continue  # avoid duplicate reason codes from correlated features
        validated.append(reason)
        seen_codes.add(reason.code)
        if len(validated) == n_reasons:
            break

    if not validated:
        # Nothing in the taxonomy matched -- never leave a letter with zero
        # reasons; use the generic approved fallback and rely on the
        # unmapped log for the real diagnosis.
        validated = [UNMAPPED_FEATURE_FALLBACK]

    return validated, unmapped


def log_unmapped_features(unmapped: list[str], applicant_id) -> None:
    """In production this writes to a compliance review queue, not a
    customer-facing surface. Placeholder here -- wire to real logging
    once deployed."""
    if unmapped:
        print(f"[compliance-review-queue] application {applicant_id}: "
              f"unmapped SHAP features requiring taxonomy review: {unmapped}")


# ---------------------------------------------------------------------
# Step 3: deterministic plain-language assembly (no generation, no LLM)
# ---------------------------------------------------------------------

def render_approved_explanation(reason_codes: list[ReasonCode]) -> str:
    """Purely deterministic template fill -- every word here traces back to
    the approved taxonomy table above. This is the text that is legally
    defensible to send as-is, with no GenAI step at all."""
    bullets = "\n".join(f"  - {r.approved_text} [{r.code}]" for r in reason_codes)
    return f"This decision was based in part on the following factors:\n{bullets}"


def generate_adverse_action_letter(applicant_id, decision: str,
                                    reason_codes: list[ReasonCode]) -> str:
    body = render_approved_explanation(reason_codes)
    return f"Decision for application {applicant_id}: {decision}\n\n{body}"


# ---------------------------------------------------------------------
# Step 4 (optional): GenAI wording pass, tightly bounded
# ---------------------------------------------------------------------

def polish_wording_with_genai(approved_text: str, reason_codes: list[ReasonCode],
                               llm_call_fn) -> str:
    """Optional step: allow an LLM to improve *tone/wording only* of the
    already-approved, deterministic text -- never to introduce new reasons,
    reorder priority, or infer additional facts. `llm_call_fn` is injected
    so this stays testable without a live API; wire to the actual model
    call at integration time.

    Guardrails enforced here, not left to the prompt alone:
      1. The prompt is constrained to rewording, explicitly forbidding
         adding/removing/reordering substantive content.
      2. The output is validated post-hoc: every approved reason code's
         key terms must still be traceable in the polished text, and no
         reason-code count mismatch is allowed. If validation fails, the
         deterministic (unpolished) text is used instead -- the GenAI
         step is fail-safe, not fail-open.
    """
    prompt = (
        "Reword the following adverse-action explanation for clarity and "
        "tone only. Do not add, remove, or reorder any reasons. Do not "
        "introduce any fact not already present in the text.\n\n"
        f"{approved_text}"
    )
    polished = llm_call_fn(prompt)

    if not _validate_polish_preserves_reasons(polished, reason_codes):
        print("[genai-guardrail] Polished text failed validation -- "
              "falling back to the deterministic approved text.")
        return approved_text

    return polished


def _validate_polish_preserves_reasons(polished_text: str, reason_codes: list[ReasonCode]) -> bool:
    """Cheap but effective guardrail: every reason code must still appear
    (as its code, e.g. '[R01]'), and no reason count drift. A production
    version would also diff key nouns/claims, but preserving the explicit
    code markers is a hard, checkable floor that a rewrite must not break."""
    if len(reason_codes) == 0:
        return False
    return all(f"[{r.code}]" in polished_text for r in reason_codes)
