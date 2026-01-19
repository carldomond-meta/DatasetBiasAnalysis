"""
Data extraction utilities for the Duplicate Taxonomies App.

Supports TWO JSON formats:
1. INTEGRITY (wrapped): Root is dict with task_id, trajectory, rogue_behavior_step_number
2. FACTUALITY (raw): Root is list [[steps], [steps]] - trajectory is the root itself

Extracts data for:
1. Fact Checker - needs evidence (plan, code, analysis, term_out)
2. Identical Behavior Analyzer - needs only the 3 necessary fields
"""

import json
import re
from pathlib import Path
from typing import Generator

from schemas import (
    FactCheckerEvidence,
    FactCheckerInput,
    IdenticalBehaviorInput,
    NecessaryFields,
    Source,
)


# =============================================================================
# FORMAT DETECTION & LOADING
# =============================================================================


def load_json_file(file_path: Path) -> dict | list | None:
    """Load a JSON file and return its content (dict or list)."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading {file_path}: {e}")
        return None


def detect_format(data: dict | list) -> str:
    """
    Detect whether the JSON is INTEGRITY (wrapped) or FACTUALITY (raw) format.

    Returns:
        "INTEGRITY" if root is dict with trajectory field
        "FACTUALITY" if root is list
        "UNKNOWN" otherwise
    """
    if isinstance(data, dict) and "trajectory" in data:
        return "INTEGRITY"
    elif isinstance(data, list):
        return "FACTUALITY"
    return "UNKNOWN"


def extract_task_id_from_filename(file_name: str) -> str:
    """
    Extract a unique identifier from the filename for FACTUALITY format.

    Examples:
    - "08-09-fix-ops_all_maverick_aide_latest___alaska2-image-steganalysis___5__2025-08-10_12-58-57-343517__branch_3_of_5_annotated.json"
      -> "alaska2-image-steganalysis___5__branch_3_of_5"
    """
    # Remove .json extension
    name = file_name.replace(".json", "")

    # Try to extract meaningful parts
    # Pattern: contains task name and branch info
    parts = name.split("___")
    if len(parts) >= 2:
        # Get task name and any branch info
        task_part = parts[1] if len(parts) > 1 else parts[0]
        branch_part = ""
        for part in parts:
            if "branch" in part.lower():
                branch_part = part
                break

        if branch_part:
            return f"{task_part}___{branch_part}"
        return task_part

    # Fallback: use the whole filename as ID
    return name


# =============================================================================
# TRAJECTORY PARSING
# =============================================================================


def parse_trajectory(data: dict | list, format_type: str) -> list:
    """
    Parse trajectory based on format type.

    For INTEGRITY: trajectory is in data["trajectory"] and may be a JSON string
    For FACTUALITY: the data itself IS the trajectory (list of lists)
    """
    if format_type == "INTEGRITY":
        trajectory = data.get("trajectory", [])
        if isinstance(trajectory, str):
            try:
                return json.loads(trajectory)
            except json.JSONDecodeError:
                return []
        return trajectory
    elif format_type == "FACTUALITY":
        # Data is already the trajectory
        return data if isinstance(data, list) else []
    return []


def find_step_by_number(trajectory: list, step_number: str) -> dict | None:
    """
    Find a step by its step number in a trajectory.

    Handles both:
    - Flat structure: [{'data': {...}}, ...]
    - Nested structure (INTEGRITY): [[{'step': N, 'data': {...}}, ...], ...]
    """
    for item in trajectory:
        # Handle nested list structure (INTEGRITY: list of branches)
        if isinstance(item, list):
            for step in item:
                if isinstance(step, dict):
                    # Check step number at top level of step
                    if str(step.get("step")) == str(step_number):
                        return step.get("data", step)
                    # Check step number inside 'data'
                    if "data" in step:
                        step_data = step["data"]
                        if str(step_data.get("step")) == str(step_number):
                            return step_data
        # Handle flat structure (step directly in trajectory)
        elif isinstance(item, dict):
            if "data" in item:
                step_data = item["data"]
                if str(step_data.get("step")) == str(step_number):
                    return step_data
    return None


def find_rogue_behavior_step(trajectory: list) -> dict | None:
    """
    Find the first step that contains rogue_behavior_type.
    Used for FACTUALITY format where we don't have a top-level step number.
    """
    for step in trajectory:
        if isinstance(step, dict) and "data" in step:
            step_data = step["data"]
            if step_data.get("rogue_behavior_type"):
                return step_data
    return None


def normalize_rogue_behavior_type(value) -> str:
    """
    Normalize rogue_behavior_type which can be a string or list.

    INTEGRITY uses: "Result Hallucination"
    FACTUALITY uses: ["Incorrect Data Analysis"]
    """
    if isinstance(value, list):
        return value[0] if value else ""
    return value if isinstance(value, str) else ""


# =============================================================================
# MAIN EXTRACTION FUNCTIONS
# =============================================================================


def extract_for_fact_checker(file_path: Path) -> FactCheckerInput | None:
    """
    Extract data needed for the Fact Checker from a trajectory JSON.

    Supports both INTEGRITY (wrapped) and FACTUALITY (raw) formats.

    IMPORTANT: Only returns traces with rogue_behavior_origin = "injected".
    Identified behaviors are filtered out.

    Returns FactCheckerInput with:
    - source identifiers
    - rogue_behavior_type
    - rogue_behavior_explanation
    - evidence (plan, code, analysis, term_out)
    """
    data = load_json_file(file_path)
    if not data:
        return None

    format_type = detect_format(data)
    if format_type == "UNKNOWN":
        return None

    # Parse trajectory
    trajectories = parse_trajectory(data, format_type)
    if not trajectories:
        return None

    # Find the rogue step based on format
    rogue_step_data = None
    rogue_step_number = ""
    task_id = ""

    if format_type == "INTEGRITY":
        # INTEGRITY: Get task_id and step number from top-level
        task_id = data.get("task_id", "")
        rogue_step_number = str(data.get("rogue_behavior_step_number", ""))

        if not task_id or not rogue_step_number:
            return None

        # Search trajectories for the rogue step
        for trajectory in trajectories:
            rogue_step_data = find_step_by_number(trajectory, rogue_step_number)
            if rogue_step_data and rogue_step_data.get("rogue_behavior_type"):
                break

    elif format_type == "FACTUALITY":
        # FACTUALITY: Derive task_id from filename, search for rogue step
        task_id = extract_task_id_from_filename(file_path.name)

        # Search all trajectories for a step with rogue_behavior_type
        for trajectory in trajectories:
            rogue_step_data = find_rogue_behavior_step(trajectory)
            if rogue_step_data:
                rogue_step_number = str(rogue_step_data.get("step", ""))
                break

    if not rogue_step_data:
        return None

    # FILTER: Only include INJECTED/INTRODUCED rogue behaviors (not "identified")
    rogue_behavior_origin = rogue_step_data.get("rogue_behavior_origin", "").lower()
    if rogue_behavior_origin not in ("injected", "introduced"):
        return None

    # Extract and normalize rogue behavior fields
    rogue_behavior_type = normalize_rogue_behavior_type(
        rogue_step_data.get("rogue_behavior_type", "")
    )
    rogue_behavior_explanation = rogue_step_data.get("rogue_behavior_explanation", "")

    if not rogue_behavior_type or not rogue_behavior_explanation:
        return None

    # Build source
    source = Source(
        file_name=file_path.name,
        task_id=task_id,
        rogue_step_number=rogue_step_number,
    )

    # Extract evidence
    evidence = FactCheckerEvidence(
        plan=rogue_step_data.get("plan"),
        code=rogue_step_data.get("code"),
        analysis=rogue_step_data.get("analysis"),
        term_out=rogue_step_data.get("term_out"),
    )

    return FactCheckerInput(
        source=source,
        rogue_behavior_type=rogue_behavior_type,
        rogue_behavior_explanation=rogue_behavior_explanation,
        rogue_behavior_origin=rogue_behavior_origin,
        evidence=evidence,
    )


def extract_for_identical_behavior(file_path: Path) -> IdenticalBehaviorInput | None:
    """
    Extract data needed for the Identical Behavior Analyzer from a trajectory JSON.

    Supports both INTEGRITY (wrapped) and FACTUALITY (raw) formats.

    IMPORTANT: Only returns traces with rogue_behavior_origin = "injected".
    Identified behaviors are filtered out.

    Returns IdenticalBehaviorInput with:
    - source identifiers (file_name, task_id, rogue_step_number)
    - necessary_fields (rogue_behavior_type, rogue_behavior_explanation)
    """
    data = load_json_file(file_path)
    if not data:
        return None

    format_type = detect_format(data)
    if format_type == "UNKNOWN":
        return None

    # Parse trajectory
    trajectories = parse_trajectory(data, format_type)
    if not trajectories:
        return None

    # Find the rogue step based on format
    rogue_step_data = None
    rogue_step_number = ""
    task_id = ""

    if format_type == "INTEGRITY":
        # INTEGRITY: Get task_id and step number from top-level
        task_id = data.get("task_id", "")
        rogue_step_number = str(data.get("rogue_behavior_step_number", ""))

        if not task_id or not rogue_step_number:
            return None

        # Search trajectories for the rogue step
        for trajectory in trajectories:
            rogue_step_data = find_step_by_number(trajectory, rogue_step_number)
            if rogue_step_data and rogue_step_data.get("rogue_behavior_type"):
                break

    elif format_type == "FACTUALITY":
        # FACTUALITY: Derive task_id from filename, search for rogue step
        task_id = extract_task_id_from_filename(file_path.name)

        # Search all trajectories for a step with rogue_behavior_type
        for trajectory in trajectories:
            rogue_step_data = find_rogue_behavior_step(trajectory)
            if rogue_step_data:
                rogue_step_number = str(rogue_step_data.get("step", ""))
                break

    if not rogue_step_data:
        return None

    # FILTER: Only include INJECTED/INTRODUCED rogue behaviors (not "identified")
    rogue_behavior_origin = rogue_step_data.get("rogue_behavior_origin", "").lower()
    if rogue_behavior_origin not in ("injected", "introduced"):
        return None

    # Extract and normalize the necessary fields
    rogue_behavior_type = normalize_rogue_behavior_type(
        rogue_step_data.get("rogue_behavior_type", "")
    )
    rogue_behavior_explanation = rogue_step_data.get("rogue_behavior_explanation", "")

    if not rogue_behavior_type or not rogue_behavior_explanation:
        return None

    # Build source
    source = Source(
        file_name=file_path.name,
        task_id=task_id,
        rogue_step_number=rogue_step_number,
    )

    necessary_fields = NecessaryFields(
        rogue_behavior_type=rogue_behavior_type,
        rogue_behavior_explanation=rogue_behavior_explanation,
    )

    return IdenticalBehaviorInput(
        source=source,
        necessary_fields=necessary_fields,
    )


# =============================================================================
# BATCH EXTRACTION
# =============================================================================


def extract_all_for_fact_checker(
    data_dir: Path, pattern: str = "*.json"
) -> Generator[FactCheckerInput, None, None]:
    """Extract Fact Checker inputs from all JSON files in a directory."""
    for file_path in data_dir.glob(pattern):
        result = extract_for_fact_checker(file_path)
        if result:
            yield result


def extract_all_for_identical_behavior(
    data_dir: Path, pattern: str = "*.json"
) -> Generator[IdenticalBehaviorInput, None, None]:
    """Extract Identical Behavior Analyzer inputs from all JSON files in a directory."""
    for file_path in data_dir.glob(pattern):
        result = extract_for_identical_behavior(file_path)
        if result:
            yield result


def group_by_taxonomy(
    inputs: list[IdenticalBehaviorInput],
) -> dict[str, list[IdenticalBehaviorInput]]:
    """Group Identical Behavior inputs by their rogue_behavior_type."""
    grouped: dict[str, list[IdenticalBehaviorInput]] = {}
    for inp in inputs:
        taxonomy = inp.necessary_fields.rogue_behavior_type
        if taxonomy not in grouped:
            grouped[taxonomy] = []
        grouped[taxonomy].append(inp)
    return grouped


# =============================================================================
# CLI for testing extraction
# =============================================================================

if __name__ == "__main__":
    import sys

    from config import DATA_DIR

    print("=" * 60)
    print("DATA EXTRACTOR - Testing Extraction")
    print("=" * 60)

    # Find JSON files
    json_files = list(DATA_DIR.glob("*.json"))
    print(f"\nFound {len(json_files)} JSON files in {DATA_DIR}")

    if not json_files:
        print("No JSON files found. Exiting.")
        sys.exit(1)

    # Test extraction on each file
    integrity_count = 0
    factuality_count = 0
    failed_count = 0

    for test_file in json_files[:10]:  # Test first 10 files
        print(f"\n--- Testing: {test_file.name[:60]}... ---")

        # Detect format
        data = load_json_file(test_file)
        if data:
            format_type = detect_format(data)
            print(f"  Format: {format_type}")

            if format_type == "INTEGRITY":
                integrity_count += 1
            elif format_type == "FACTUALITY":
                factuality_count += 1

        # Test Fact Checker extraction
        fc_input = extract_for_fact_checker(test_file)
        if fc_input:
            print(f"  [FC] Task ID: {fc_input.source.task_id[:50]}...")
            print(f"  [FC] Type: {fc_input.rogue_behavior_type}")
            print(f"  [FC] Explanation: {fc_input.rogue_behavior_explanation[:60]}...")
        else:
            print("  [FC] Extraction failed")
            failed_count += 1

        # Test Identical Behavior extraction
        ib_input = extract_for_identical_behavior(test_file)
        if ib_input:
            print(f"  [IB] Type: {ib_input.necessary_fields.rogue_behavior_type}")
        else:
            print("  [IB] Extraction failed")

    print("\n" + "=" * 60)
    print(
        f"Summary: INTEGRITY={integrity_count}, FACTUALITY={factuality_count}, Failed={failed_count}"
    )
