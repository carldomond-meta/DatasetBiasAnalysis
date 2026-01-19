"""
Identical Behavior Analyzer Component for the Duplicate Taxonomies App.

Purpose: For a given rogue behavior taxonomy, determine if vendor annotations
exhibit the SAME behavioral pattern consistently (which indicates poor data diversity).ß
"""

import json
import time
from pathlib import Path

from config import (
    DATA_DIR,
    ensure_output_dirs,
    FACT_CHECKER_OUTPUT_DIR,
    IDENTICAL_BEHAVIOR_OUTPUT_DIR,
    MAX_BATCH_SIZE,
)
from data_extractor import extract_all_for_identical_behavior, group_by_taxonomy
from llama_client import LlamaClient
from schemas import IdenticalBehaviorInput, IdenticalBehaviorOutput, TraceAnalysis

# Rate limiting settings
BATCH_SIZE_BEFORE_PAUSE = 10
PAUSE_SECONDS = 60


# =============================================================================
# PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You are a data quality analyst evaluating training data for an AI model that will identify rogue behaviors in agent trajectories.

Your task: For a given taxonomy, IDENTIFY whether all traces exhibit the SAME behavior consistently (singular behavior) or DIFFERENT behaviors (diverse behaviors).

WHY THIS MATTERS:
- If all traces show the SAME behavior → The model will only learn ONE manifestation of this taxonomy
- If traces show DIFFERENT behaviors → The model will learn MULTIPLE manifestations and generalize better

CRITICAL - HOW TO DETERMINE IF METHODS ARE DIFFERENT:

Ask yourself: "Would a developer need to write DIFFERENT detection logic for each of these?"

SAME method examples (would use same detection logic):
- "Hallucinated success after timeout" vs "Hallucinated success after module error"
  → SAME: Both are "ignored error signal and claimed success"
- "Accessed test.csv before training" vs "Accessed test_data.json before training"
  → SAME: Both are "read test file before training"

DIFFERENT method examples (would need different detection logic):
- "Model state persisted between CV folds" vs "Computed features on full dataset before split" vs "Concatenated train+val after fold"
  → DIFFERENT: Model lifecycle issue vs Feature engineering timing vs Data handling
- "Loaded test images to check dimensions" vs "Read test labels during training" vs "Ran tokenizer analysis on test comments"
  → DIFFERENT: Dimension inspection vs Label access vs Text preprocessing

The key question: Are the traces doing FUNDAMENTALLY DIFFERENT THINGS, or the SAME THING in different contexts?

YOUR ANALYSIS MUST:
1. Examine each trace and identify the SPECIFIC MECHANISM it uses
2. Compare mechanisms - would they require different detection approaches?
3. Conclude whether this taxonomy shows singular (same mechanism) or diverse (different mechanisms) behavior"""


def build_identical_behavior_user_prompt(
    taxonomy: str, annotations: list[IdenticalBehaviorInput]
) -> str:
    """
    Build the user prompt for the LLM to analyze identical behavior patterns.
    """
    annotation_texts = []
    for i, ann in enumerate(annotations, 1):
        explanation = ann.necessary_fields.rogue_behavior_explanation
        task_id = ann.source.task_id
        annotation_texts.append(f'{i}. [Task: {task_id}]\n   "{explanation}"')

    annotations_str = "\n\n".join(annotation_texts)

    prompt = f"""TAXONOMY: {taxonomy}
TOTAL TRACES: {len(annotations)}

TRACES TO ANALYZE:
{annotations_str}

Provide your analysis:

1. For EACH trace, identify the behavior_exhibited using this format:
   "Despite [this trace's unique context], the agent [how it exhibited the behavior pattern]"

2. similarity_analysis: Compare the METHOD/MECHANISM of how the behavior is exhibited across all traces.

   DO NOT just confirm they all exhibit the taxonomy (e.g., "they all access test data"). That is useless.

   INSTEAD, analyze: Is the METHOD of exhibiting this behavior the SAME or DIFFERENT across traces?

   Examples of what we need:
   - SAME: "All traces access test data through the same method: loading and inspecting test metadata before training begins"
   - DIFFERENT: "Traces show different methods: one loads images for dimension checks, another reads labels during CV, another runs exploratory tokenizer analysis"

   REMEMBER: For all taxonomies except ungrounded_reasoning, different contexts (like timeout vs module error) do NOT count as different mechanisms. Focus on the METHOD the agent used.

3. is_singular_behavior: Based on whether the METHOD is consistent across traces (true) or different (false)"""

    return prompt


def analyze_with_llm(
    taxonomy: str,
    annotations: list[IdenticalBehaviorInput],
    client: LlamaClient,
) -> IdenticalBehaviorOutput:
    """
    Use LlamaClient to analyze identical behavior patterns.

    Returns a validated IdenticalBehaviorOutput.
    """
    user_prompt = build_identical_behavior_user_prompt(taxonomy, annotations)

    return client.generate_structured(
        schema=IdenticalBehaviorOutput,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

def analyze_taxonomy(
    taxonomy: str,
    annotations: list[IdenticalBehaviorInput],
    client: LlamaClient,
) -> IdenticalBehaviorOutput:
    """
    Analyze a single taxonomy for identical behavior patterns.

    Args:
        taxonomy: The rogue behavior type
        annotations: List of annotations for this taxonomy
        client: LlamaClient instance

    Returns:
        IdenticalBehaviorOutput with the analysis
    """
    try:
        return analyze_with_llm(taxonomy, annotations, client)
    except Exception as e:
        print(f"  LLM call failed: {e}")
        raise


def load_valid_task_ids(fact_checker_output_dir: Path) -> set[str] | None:
    """
    Load valid task IDs from the Fact Checker output.

    Returns None if file doesn't exist (meaning no filtering).
    """
    valid_ids_file = fact_checker_output_dir / "valid_task_ids.json"
    if valid_ids_file.exists():
        with open(valid_ids_file, "r") as f:
            return set(json.load(f))
    return None


def run_identical_behavior_analyzer(
    data_dir: Path,
    output_dir: Path,
    fact_checker_output_dir: Path | None = None,
) -> list[IdenticalBehaviorOutput]:
    """
    Run the Identical Behavior Analyzer on all trajectory JSONs.

    Args:
        data_dir: Directory containing trajectory JSON files
        output_dir: Directory to save results
        fact_checker_output_dir: Optional path to fact checker output (for filtering)

    Returns:
        List of IdenticalBehaviorOutput for each taxonomy
    """
    ensure_output_dirs()

    # Initialize LlamaClient
    try:
        client = LlamaClient()
        print(f"Using LLAMA model: {client.MODEL}")
    except ValueError as e:
        print(f"Error initializing LlamaClient: {e}")
        raise

    # Load valid task IDs if fact checker was run
    valid_task_ids = None
    if fact_checker_output_dir:
        valid_task_ids = load_valid_task_ids(fact_checker_output_dir)
        if valid_task_ids:
            print(f"Loaded {len(valid_task_ids)} valid task IDs from Fact Checker")

    # Extract all annotations
    all_annotations = list(extract_all_for_identical_behavior(data_dir))
    print(f"Extracted {len(all_annotations)} annotations from {data_dir}")

    # Filter by valid task IDs if available
    if valid_task_ids is not None:
        original_count = len(all_annotations)
        all_annotations = [
            a for a in all_annotations if a.source.task_id in valid_task_ids
        ]
        print(
            f"Filtered to {len(all_annotations)} valid annotations (from {original_count})"
        )

    # Group by taxonomy
    grouped = group_by_taxonomy(all_annotations)
    print(f"Found {len(grouped)} unique taxonomies")

    # Filter out taxonomies with only 1 instance (can't compare for identical behavior)
    original_taxonomy_count = len(grouped)
    grouped = {k: v for k, v in grouped.items() if len(v) > 1}
    skipped_count = original_taxonomy_count - len(grouped)
    if skipped_count > 0:
        print(
            f"Skipped {skipped_count} taxonomies with only 1 instance (nothing to compare)"
        )
    print(f"Analyzing {len(grouped)} taxonomies with 2+ instances")

    # Analyze each taxonomy
    results: list[IdenticalBehaviorOutput] = []

    print("\n" + "=" * 60)
    print("IDENTICAL BEHAVIOR ANALYSIS")
    print(
        f"Rate limiting: Pause {PAUSE_SECONDS}s every {BATCH_SIZE_BEFORE_PAUSE} requests"
    )
    print("=" * 60)

    taxonomy_count = 0
    for taxonomy, annotations in grouped.items():
        # Rate limiting
        if taxonomy_count > 0 and taxonomy_count % BATCH_SIZE_BEFORE_PAUSE == 0:
            print(
                f"\n--- Rate limit pause: waiting {PAUSE_SECONDS}s after {taxonomy_count} taxonomies ---\n"
            )
            time.sleep(PAUSE_SECONDS)

        print(f"\n--- Taxonomy: {taxonomy} ({len(annotations)} annotations) ---")

        # Batch if too many annotations
        if len(annotations) > MAX_BATCH_SIZE:
            print(f"  Truncating to {MAX_BATCH_SIZE} annotations for analysis")
            annotations = annotations[:MAX_BATCH_SIZE]

        output = analyze_taxonomy(taxonomy, annotations, client=client)

        print(f"  Singular Behavior: {output.is_singular_behavior}")
        print(f"  Traces analyzed: {len(output.trace_analyses)}")
        print(f"  Similarity: {output.similarity_analysis[:60]}...")

        results.append(output)
        taxonomy_count += 1

    # Save results
    output_file = output_dir / "identical_behavior_results.json"
    with open(output_file, "w") as f:
        json.dump([r.model_dump() for r in results], f, indent=2, default=str)

    print("\n" + "=" * 60)
    print(f"Results saved to {output_file}")

    # Generate summary report
    summary = generate_summary(results)
    summary_file = output_dir / "identical_behavior_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to {summary_file}")

    return results


def generate_summary(results: list[IdenticalBehaviorOutput]) -> dict:
    """Generate a summary report of the analysis."""
    singular_taxonomies = [r for r in results if r.is_singular_behavior]

    return {
        "total_taxonomies_analyzed": len(results),
        "singular_behavior_count": len(singular_taxonomies),
        "diverse_behavior_count": len(results) - len(singular_taxonomies),
        "all_taxonomies": [
            {
                "taxonomy": r.taxonomy,
                "is_singular_behavior": r.is_singular_behavior,
                "total_instances": r.total_instances,
                "traces_analyzed": len(r.trace_analyses),
            }
            for r in results
        ],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Identical Behavior Analyzer - Detect uniform patterns within taxonomies"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing trajectory JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=IDENTICAL_BEHAVIOR_OUTPUT_DIR,
        help="Directory to save results",
    )
    parser.add_argument(
        "--use-fact-checker",
        action="store_true",
        help="Filter annotations using Fact Checker's valid_task_ids.json",
    )
    parser.add_argument(
        "--fact-checker-dir",
        type=Path,
        default=FACT_CHECKER_OUTPUT_DIR,
        help="Directory containing Fact Checker output (only used with --use-fact-checker)",
    )

    args = parser.parse_args()

    fact_checker_dir = args.fact_checker_dir if args.use_fact_checker else None

    results = run_identical_behavior_analyzer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        fact_checker_output_dir=fact_checker_dir,
    )
