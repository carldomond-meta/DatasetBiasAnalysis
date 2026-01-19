"""
Pydantic schemas for the Duplicate Taxonomies App.ß
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Source(BaseModel):
    """Identifies the origin of each annotation for traceability."""

    file_name: str
    task_id: str
    rogue_step_number: str


class FactCheckerEvidence(BaseModel):
    """Evidence extracted from the rogue behavior step for fact checking."""

    plan: Optional[str] = None
    code: Optional[str] = None
    analysis: Optional[str] = None
    term_out: Optional[str] = None


class FactCheckerInput(BaseModel):
    """Input to the Fact Checker component."""

    source: Source
    rogue_behavior_type: str
    rogue_behavior_explanation: str
    rogue_behavior_origin: str
    evidence: FactCheckerEvidence


class FactCheckerVerdict(BaseModel):
    """LLM-generated verdict on whether the rogue behavior claim is valid."""

    is_valid: bool = Field(
        description="Whether the rogue behavior explanation is actually exhibited in the evidence"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score from 0 to 1"
    )
    reasoning: str = Field(
        description="Explanation of why the claim is valid or invalid"
    )
    discrepancy_details: Optional[str] = Field(
        default=None, description="Details of the mismatch (if invalid)"
    )


class FactCheckerOutput(BaseModel):
    """Complete output from the Fact Checker for a single trace."""

    source: Source
    rogue_behavior_type: str
    rogue_behavior_origin: str
    verdict: FactCheckerVerdict



class NecessaryFields(BaseModel):
    """The necessary fields for identical behavior analysis."""

    rogue_behavior_type: str
    rogue_behavior_explanation: str


class IdenticalBehaviorInput(BaseModel):
    """Input to the Identical Behavior Analyzer for a single trace."""

    source: Source
    necessary_fields: NecessaryFields


class TraceAnalysis(BaseModel):
    """Per-trace analysis of what behavior this trace exhibits."""

    task_id: str = Field(description="Unique identifier for the trace")
    rogue_behavior_type: str = Field(description="The taxonomy label for this trace")
    rogue_behavior_explanation: str = Field(
        description="Vendor's original explanation for the rogue behavior"
    )
    behavior_exhibited: str = Field(
        description="How the behavior pattern manifests within this trace's unique context. "
        "Use format: 'Despite [this trace's unique context], the agent [how it exhibited the behavior pattern]'"
    )


class IdenticalBehaviorOutput(BaseModel):
    """Complete output from the Identical Behavior Analyzer for a taxonomy."""

    taxonomy: str = Field(description="The rogue behavior taxonomy being analyzed")
    total_instances: int = Field(
        description="Total number of traces with this taxonomy"
    )
    is_singular_behavior: bool = Field(
        description="Whether this taxonomy is represented by a singular behavior consistently"
    )
    similarity_analysis: str = Field(
        description="Are the traces showing the SAME behavior consistently, or DIFFERENT behaviors"
    )
    trace_analyses: list[TraceAnalysis] = Field(
        description="Per-trace behavior analysis"
    )


# =============================================================================
# REPORT GENERATION SCHEMAS (LLM-enhanced narratives)
# =============================================================================


class TaxonomyNarrative(BaseModel):
    """LLM-generated narrative for a taxonomy's findings in the report."""

    taxonomy: str = Field(description="The taxonomy being described")

    training_coverage_assessment: str = Field(
        description="Assess what this training data covers for this taxonomy. "
        "If diverse: Explain that the model will learn to recognize multiple manifestations. "
        "If singular: Explain that the model will learn this one pattern well, but may not generalize to other manifestations."
    )

    pattern_summary: str = Field(
        description="If singular: Describe the ONE consistent pattern observed across all instances. "
        "If diverse: Summarize the different methods observed and how they contrast with each other."
    )


class ReportNarratives(BaseModel):
    """Collection of LLM-generated narratives for all analyzed taxonomies."""

    taxonomy_narratives: list[TaxonomyNarrative] = Field(
        description="Narrative for each taxonomy that was analyzed"
    )
