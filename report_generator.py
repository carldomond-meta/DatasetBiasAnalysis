"""
Markdown Report Generator for the Duplicate Taxonomies App.
"""

import json
from datetime import datetime
from pathlib import Path

from config import (
    DATA_DIR,
    IDENTICAL_BEHAVIOR_OUTPUT_DIR,
    OUTPUT_DIR,
)
from data_extractor import (
    extract_all_for_identical_behavior,
    group_by_taxonomy,
    load_json_file,
    detect_format,
)
from llama_client import LlamaClient
from schemas import ReportNarratives, TaxonomyNarrative


NARRATIVE_SYSTEM_PROMPT = """You are a data quality analyst writing a report on training data coverage for rogue behavior detection models.

Your task is to generate clear, interpretive narratives that explain what the training data covers for each taxonomy.

Focus on:
1. What manifestations of the behavior the training data covers
2. What a model trained on this data will learn to recognize
3. Whether the coverage is diverse (multiple methods) or singular (one pattern)

Do NOT:
- Suggest what's missing or recommend collecting more data
- Use overly technical jargon
- Make claims about the "full spectrum" since we only know what we have"""


def build_narrative_prompt(ib_results: list[dict]) -> str:
    """Build the user prompt for generating report narratives."""

    prompt_parts = ["Generate narratives for the following taxonomy analyses:\n"]

    for result in ib_results:
        taxonomy = result.get("taxonomy", "Unknown")
        is_singular = result.get("is_singular_behavior", False)
        similarity_analysis = result.get("similarity_analysis", "")
        trace_analyses = result.get("trace_analyses", [])
        total_instances = result.get("total_instances", 0)

        prompt_parts.append(f"\n## {taxonomy}")
        prompt_parts.append(f"Instances: {total_instances}")
        prompt_parts.append(f"Is Singular Behavior: {is_singular}")
        prompt_parts.append(f"Similarity Analysis: {similarity_analysis}")
        prompt_parts.append("Behaviors Exhibited:")

        for trace in trace_analyses:
            behavior = trace.get("behavior_exhibited", "")
            prompt_parts.append(f"  - {behavior}")

    prompt_parts.append("\n\nFor each taxonomy, provide:")
    prompt_parts.append("1. training_coverage_assessment: What does this data cover? What will a model learn?")
    prompt_parts.append("2. pattern_summary: Describe the pattern(s) observed - one consistent pattern if singular, or the contrasting methods if diverse.")

    return "\n".join(prompt_parts)


def generate_narratives_with_llm(ib_results: list[dict]) -> dict[str, TaxonomyNarrative]:
    """
    Use LLM to generate interpretive narratives for each taxonomy.

    Returns a dict mapping taxonomy name to its narrative.
    """
    if not ib_results:
        return {}

    try:
        client = LlamaClient()
        user_prompt = build_narrative_prompt(ib_results)

        response = client.generate_structured(
            schema=ReportNarratives,
            system_prompt=NARRATIVE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        return {n.taxonomy: n for n in response.taxonomy_narratives}

    except Exception as e:
        print(f"Warning: LLM narrative generation failed: {e}")
        print("Falling back to template-based narratives.")
        return {}


def generate_fallback_narrative(result: dict) -> TaxonomyNarrative:
    """Generate a template-based narrative when LLM is unavailable."""
    taxonomy = result.get("taxonomy", "Unknown")
    is_singular = result.get("is_singular_behavior", False)
    similarity_analysis = result.get("similarity_analysis", "")
    trace_analyses = result.get("trace_analyses", [])
    total_instances = result.get("total_instances", 0)

    if is_singular:
        training_coverage = (
            f"This training data covers one specific manifestation of {taxonomy}. "
            f"A model trained on this data will reliably detect this pattern, "
            f"but may not generalize to other forms of this behavior."
        )
        pattern_summary = similarity_analysis
    else:
        methods = [t.get("behavior_exhibited", "") for t in trace_analyses]
        training_coverage = (
            f"This training data covers {total_instances} distinct manifestations of {taxonomy}. "
            f"A model trained on this data will learn to recognize varied forms of this behavior."
        )
        pattern_summary = similarity_analysis

    return TaxonomyNarrative(
        taxonomy=taxonomy,
        training_coverage_assessment=training_coverage,
        pattern_summary=pattern_summary,
    )


def load_identical_behavior_results(ib_dir: Path) -> list | None:
    """Load identical behavior analyzer results if available."""
    results_file = ib_dir / "identical_behavior_results.json"
    if results_file.exists():
        with open(results_file, "r") as f:
            return json.load(f)
    return None


def load_identical_behavior_summary(ib_dir: Path) -> dict | None:
    """Load identical behavior summary if available."""
    summary_file = ib_dir / "identical_behavior_summary.json"
    if summary_file.exists():
        with open(summary_file, "r") as f:
            return json.load(f)
    return None


def get_data_extraction_stats(data_dir: Path) -> dict:
    """
    Get statistics about the data extraction process.
    """
    stats = {
        "total_json_files": 0,
        "integrity_format_count": 0,
        "factuality_format_count": 0,
        "unknown_format_count": 0,
        "extracted_annotations": 0,
        "filtered_out": 0,
        "taxonomies_found": {},
    }

    json_files = list(data_dir.glob("*.json"))
    stats["total_json_files"] = len(json_files)

    for file_path in json_files:
        data = load_json_file(file_path)
        if data:
            format_type = detect_format(data)
            if format_type == "INTEGRITY":
                stats["integrity_format_count"] += 1
            elif format_type == "FACTUALITY":
                stats["factuality_format_count"] += 1
            else:
                stats["unknown_format_count"] += 1

    all_annotations = list(extract_all_for_identical_behavior(data_dir))
    stats["extracted_annotations"] = len(all_annotations)
    stats["filtered_out"] = stats["total_json_files"] - stats["extracted_annotations"]

    grouped = group_by_taxonomy(all_annotations)
    stats["taxonomies_found"] = {k: len(v) for k, v in grouped.items()}

    return stats
ß

def generate_report(
    data_dir: Path = DATA_DIR,
    ib_dir: Path = IDENTICAL_BEHAVIOR_OUTPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
    use_llm: bool = True,
) -> str:
    """
    Generate a comprehensive markdown report from analysis outputs.

    Args:
        data_dir: Directory containing the batch of JSON files
        ib_dir: Directory containing Identical Behavior Analyzer output
        output_dir: Directory to save the report
        use_llm: Whether to use LLM for enhanced narratives

    Returns the markdown content as a string.
    """
    # Load data
    ib_results = load_identical_behavior_results(ib_dir)
    ib_summary = load_identical_behavior_summary(ib_dir)
    extraction_stats = get_data_extraction_stats(data_dir)

    # Generate LLM narratives if enabled
    narratives = {}
    if use_llm and ib_results:
        print("Generating LLM-enhanced narratives...")
        narratives = generate_narratives_with_llm(ib_results)

    # Build report
    report = []

    # ==========================================================================
    # HEADER
    # ==========================================================================
    report.append("# Taxonomy Coverage Analysis Report")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n**Batch Source:** `{data_dir}`\n")

    # ==========================================================================
    # CONTEXT / OVERVIEW
    # ==========================================================================
    report.append("## Overview\n")
    report.append(
        "This report analyzes the quality and coverage of injected rogue behavior annotations. "
        "To train an effective rogue behavior detection model, the training data must capture "
        "diverse manifestations of each taxonomy — not just one repeated pattern.\n"
    )
    report.append(
        "**Goal:** Ensure that for each rogue behavior taxonomy, the injected examples exhibit "
        "the behavior through varied methods. This diversity enables the model to recognize "
        "the full spectrum of how a taxonomy can manifest, rather than overfitting to a single pattern.\n"
    )
    report.append(
        "**Methodology:** We analyze injected rogue behaviors grouped by taxonomy. For each taxonomy "
        "with 2+ instances, we assess whether the instances demonstrate the behavior through the "
        "same method (singular — low coverage) or different methods (diverse — good coverage).\n"
    )

    # ==========================================================================
    # DATA OVERVIEW (BATCH CONTEXT)
    # ==========================================================================
    report.append("## Batch Data Overview\n")
    report.append(f"From this batch of **{extraction_stats['total_json_files']}** JSON files:\n")

    report.append("### Extraction Summary\n")
    report.append("| Metric | Count |")
    report.append("|--------|-------|")
    report.append(f"| Total JSON Files in Batch | {extraction_stats['total_json_files']} |")
    report.append(f"| Injected Rogue Behaviors Extracted | {extraction_stats['extracted_annotations']} |")
    report.append(f"| Filtered Out | {extraction_stats['filtered_out']} |")
    report.append("")

    # Explain what was filtered out and WHY
    report.append("### What Was Filtered Out\n")
    report.append(
        f"**{extraction_stats['filtered_out']}** files were excluded from analysis for the following reasons:\n"
    )
    report.append(
        "- **Non-injected behaviors:** Only `introduced` or `injected` rogue behaviors are analyzed. "
        "Behaviors marked as `identified` (naturally occurring) are excluded since we are specifically "
        "assessing vendor-injected training data quality.\n"
    )
    report.append(
        "- **No rogue behavior present:** Some trajectory files do not contain annotated rogue behaviors.\n"
    )

    # Taxonomies found with eligibility
    if extraction_stats['taxonomies_found']:
        report.append("### Taxonomies Identified\n")
        report.append("| Taxonomy | Instances | Eligible for Analysis |")
        report.append("|----------|-----------|----------------------|")

        for taxonomy, count in sorted(extraction_stats['taxonomies_found'].items(), key=lambda x: -x[1]):
            eligible = "✅ Yes" if count >= 2 else "❌ No (single instance)"
            report.append(f"| {taxonomy} | {count} | {eligible} |")
        report.append("")

        # Explain why single-instance taxonomies are excluded
        single_instance = [t for t, c in extraction_stats['taxonomies_found'].items() if c == 1]
        if single_instance:
            report.append(
                f"**Why single-instance taxonomies are excluded:** {len(single_instance)} taxonomy/taxonomies "
                f"have only 1 instance in this batch. Coverage analysis requires comparing multiple instances "
                f"to determine if the behavior is exhibited consistently (same method) or diversely (varied methods). "
                f"A single sample cannot be compared against anything.\n"
            )

        # Explain what IS eligible
        multi_instance = [t for t, c in extraction_stats['taxonomies_found'].items() if c >= 2]
        if multi_instance:
            report.append(
                f"**Eligible for analysis:** {len(multi_instance)} taxonomy/taxonomies with 2+ instances "
                f"will be analyzed in the Behavioral Diversity Analysis section below.\n"
            )

    report.append("## Behavioral Diversity Analysis\n")

    if ib_results:
        for result in ib_results:
            taxonomy = result.get("taxonomy", "Unknown")
            is_singular = result.get("is_singular_behavior", False)
            trace_analyses = result.get("trace_analyses", [])
            total_instances = result.get("total_instances", 0)

            # Get narrative (LLM or fallback)
            if taxonomy in narratives:
                narrative = narratives[taxonomy]
            else:
                narrative = generate_fallback_narrative(result)

            # Header with status
            status_icon = "⚠️" if is_singular else "✅"
            status_text = "Low Coverage (Singular Pattern)" if is_singular else "Good Coverage (Diverse Methods)"

            report.append(f"### {status_icon} {taxonomy}\n")
            report.append(f"**Instances:** {total_instances} | **Coverage:** {status_text}\n")

            # Training Coverage Assessment (LLM-enhanced)
            report.append("**Training Coverage Assessment:**\n")
            report.append(f"> {narrative.training_coverage_assessment}\n")

            # Pattern Summary (LLM-enhanced)
            report.append("**Pattern Summary:**\n")
            report.append(f"> {narrative.pattern_summary}\n")

            # Show the distinct behaviors observed
            if is_singular:
                report.append("**Evidence (all instances show the same pattern):**\n")
            else:
                report.append("**Distinct Methods Observed:**\n")

            for i, trace in enumerate(trace_analyses, 1):
                behavior = trace.get("behavior_exhibited", "")
                report.append(f"- **Instance {i}:** {behavior}")
            report.append("")

    else:
        report.append(
            "*No behavioral diversity analysis results found. "
            "Run `python identical_behavior_analyzer.py` first.*\n"
        )

    report.append("## Summary\n")

    if ib_summary:
        total_taxonomies = ib_summary.get("total_taxonomies_analyzed", 0)
        singular_count = ib_summary.get("singular_behavior_count", 0)
        diverse_count = ib_summary.get("diverse_behavior_count", 0)

        report.append(f"**Taxonomies Analyzed:** {total_taxonomies}\n")

        if diverse_count > 0:
            report.append(
                f"- ✅ **{diverse_count}** with good coverage — training data shows diverse manifestations"
            )
        if singular_count > 0:
            report.append(
                f"- ⚠️ **{singular_count}** with low coverage — training data shows only one pattern"
            )
        report.append("")

        # Final interpretation
        if singular_count == 0 and diverse_count > 0:
            report.append(
                "**Conclusion:** This batch demonstrates good training data quality. "
                "The analyzed taxonomies show diverse behavioral patterns, which will help "
                "the model generalize to varied manifestations of rogue behaviors.\n"
            )
        elif singular_count > 0 and diverse_count == 0:
            report.append(
                "**Conclusion:** This batch shows limited diversity. All analyzed taxonomies "
                "exhibit singular behavioral patterns. The model may only learn to recognize "
                "these specific manifestations and struggle to generalize.\n"
            )
        elif singular_count > 0 and diverse_count > 0:
            report.append(
                f"**Conclusion:** Mixed results. {diverse_count} taxonomy/taxonomies show good diversity, "
                f"while {singular_count} show singular patterns that may limit model generalization.\n"
            )

    report.append("---")
    report.append("*Report generated by Taxonomy Coverage Analysis Tool*")

    # Join and return
    report_content = "\n".join(report)

    # Save to file
    report_file = output_dir / "taxonomy_coverage_report.md"
    with open(report_file, "w") as f:
        f.write(report_content)

    print(f"Report saved to {report_file}")

    return report_content


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Taxonomy Coverage Analysis Report"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing the batch of trajectory JSON files",
    )
    parser.add_argument(
        "--ib-dir",
        type=Path,
        default=IDENTICAL_BEHAVIOR_OUTPUT_DIR,
        help="Directory containing Identical Behavior Analyzer output",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to save the report",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM-enhanced narratives (use template-based instead)",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_report",
        help="Print report to console",
    )

    args = parser.parse_args()

    report = generate_report(
        data_dir=args.data_dir,
        ib_dir=args.ib_dir,
        output_dir=args.output_dir,
        use_llm=not args.no_llm,
    )

    if args.print_report:
        print("\n" + "=" * 60)
        print(report)
