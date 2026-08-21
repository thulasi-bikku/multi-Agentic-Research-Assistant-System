"""
Reviewer 3, point 7: Table 7 scored competitor tools with no stated
criteria, no assessment date, no evidence, while MARAS got a checkmark on
every row. This module enforces a citation/evidence field for every cell;
anything without one prints as "not verified" rather than a blank ("no")
or a dash, because a blank reads to most people as "confirmed absent",
which is exactly the promotional framing the reviewer objected to.
"""
from dataclasses import dataclass, field
from datetime import date
import pandas as pd


@dataclass
class FeatureClaim:
    system: str
    feature: str
    status: str          # "yes" / "no" / "partial" / "not_verified"
    evidence: str = ""   # citation, URL, or "observed on <date> via <query>"
    assessed_on: str = field(default_factory=lambda: str(date.today()))

    def __post_init__(self):
        if self.status != "not_verified" and not self.evidence:
            raise ValueError(
                f"Claim '{self.system} / {self.feature} = {self.status}' has no evidence. "
                "Either supply a citation/observation, or set status='not_verified'."
            )


def build_comparison_table(claims: list) -> pd.DataFrame:
    df = pd.DataFrame([c.__dict__ for c in claims])
    table = df.pivot(index="feature", columns="system", values="status")
    return table


def build_evidence_appendix(claims: list) -> pd.DataFrame:
    """A separate, citable appendix table -- reviewers can trace every
    claimed cell back to its source, which Table 7 as published did not
    allow."""
    return pd.DataFrame([c.__dict__ for c in claims])


# ---- example usage (fill in with your own verified observations only) ----
EXAMPLE_CLAIMS = [
    FeatureClaim("MARAS", "Multi-step retrieval pipeline", "yes",
                 evidence="measured directly in this codebase"),
    FeatureClaim("Semantic Scholar", "Citation graph search", "not_verified"),
    FeatureClaim("Elicit", "Methodology extraction", "not_verified"),
    # Add real, dated, cited entries here before publishing a comparison table.
    # If you can't verify a competitor's feature set without violating their
    # ToS or without an account, either omit the row or run the head-to-head
    # harness below on a shared query set instead of guessing.
]


def head_to_head_harness_stub(shared_queries: list, tool_call_fns: dict):
    """
    tool_call_fns: {system_name: callable(query:str) -> dict-like result}
    Runs the SAME query set through every tool you have API/UI access to
    and records raw outputs for manual, evidenced comparison -- this is
    the reviewer's suggested replacement for the checkmark table. Left as
    a stub because it requires credentials/API access to each competitor
    tool that this environment doesn't have.
    """
    results = {}
    for system, call_fn in tool_call_fns.items():
        results[system] = {q: call_fn(q) for q in shared_queries}
    return results
