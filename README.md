# Taxonomy Coverage Analysis App

**GitHub Repository:** https://github.com/carldomond-meta/DatasetBiasAnalysis

This app analyzes batches of trajectory data to determine whether rogue behavior taxonomies are exhibited consistently or through varied methods. Using the Llama API with structured JSON output and prompt engineering, it identifies potential dataset bias in training data.

## Why This Matters

For a rogue behavior detection model to generalize well, training data must capture diverse manifestations of each taxonomy—not just one repeated pattern. This app answers the question: "Are vendors annotating the same behavior pattern over and over, or are they capturing the full range of how a taxonomy can manifest?"

## How It Works

The app processes a batch of trajectory JSON files through three stages:

1. **Extract** — Parse trajectories and pull out rogue behavior annotations, grouped by taxonomy
2. **Analyze** — Use the Llama API to determine if behaviors within each taxonomy are exhibited through the same method or different methods
3. **Report** — Generate a markdown report summarizing coverage quality per taxonomy

## Core Components

| File | Purpose |
|------|---------|
| `config.py` | Configuration settings, taxonomy definitions, API settings, file paths |
| `data_extractor.py` | Parses trajectory JSONs, extracts rogue behavior fields, groups by taxonomy |
| `identical_behavior_analyzer.py` | Prompts Llama API to analyze behavioral diversity within each taxonomy |
| `report_generator.py` | Builds the final markdown report with LLM-enhanced narratives |
| `schemas.py` | Pydantic models for structured JSON output |
| `llama_client.py` | Wrapper for Llama API with structured output support |
