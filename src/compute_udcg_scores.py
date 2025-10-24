"""
Compute UDCG (Utility and Distraction-aware Cumulative Gain) Scores

This script computes UDCG scores for question-passage pairs. The UDCG metric
evaluates context quality by combining relevance information with distracting
scores to predict how well a language model will perform on a question-answering task.

The metric uses:
- Relevance labels (is_relevant: True/False) for each passage
- Abstention scores (no_res_prob) that indicate how likely the model is to abstain
  when given only that passage

The UDCG score is computed as:
  UDCG = (mean of (1 - no_res_prob) for relevant passages) -
         (weight * mean of (1 - no_res_prob) for irrelevant passages)

Input format:
  A JSON file containing a list of items, each with:
  - "question": the question text
  - "passages": list of passage objects, each with:
    - "is_relevant": boolean indicating if passage is relevant
    - "no_res_prob": abstention score for that model
    - other optional fields

Output format:
  A JSON file with the same structure, plus additional fields:
  - "udcg": the UDCG score for that item
"""

import os
import math
import json
import argparse

from copy import deepcopy
from utils import read_json, write_json


def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def get_no_res_prob(passage: dict, model_id: str) -> float:
    """Extract no_res_prob from passage dictionary."""
    return passage.get('models_info', {}).get(model_id, {}).get('no_res_prob', None)


def compute_udcg_score(
    passages: list[dict], 
    model_id: str,
    relevant_weight: float = 1.0,
    irrelevant_weight: float = -1/3
) -> float:
    """
    Compute UDCG (Utility and Distraction-aware Cumulative Gain) score.

    Args:
        passages: List of passage dictionaries with 'is_relevant' and abstention scores
        model_id: Model ID
        relevant_weight: Weight for relevant passages (default: 1.0)
        irrelevant_weight: Negative weight for irrelevant passages (default: -1/3)

    Returns:
        UDCG score (higher is better)
    """
    if len(passages) == 0:
        return 0.0

    score = 0.0
    for passage in passages:
        no_res_prob = get_no_res_prob(passage, model_id)

        # Check if passage has the required distracting score
        if no_res_prob is None:
            print(f"Warning: Passage missing no_res_prob, skipping")
            continue
        
        if 'is_relevant' not in passage:
            print(f"Warning: Passage missing is_relevant field, skipping")
            continue

        is_relevant = passage.get('is_relevant', False)

        # Convert abstention score to utility score (1 - no_res_prob)
        doc_score = 1.0 - no_res_prob

        # Apply weights based on relevance
        if is_relevant:
            score += doc_score * relevant_weight
        else:
            score += doc_score * irrelevant_weight

    # Average over all passages
    score = score / len(passages)

    return sigmoid(score)


def update_item_with_udcg_score(
    item: dict,
    model_id: str,
    udcg_score: float
) -> None:
    """Update item dictionary with UDCG score information."""
    if 'udcg' not in item:
        item['udcg'] = {}

    item['udcg'][model_id] = udcg_score


def process_dataset(
    dataset: list[dict],
    model_id: str,
    relevant_weight: float = 1.0,
    irrelevant_weight: float = -1/3
) -> list[dict]:
    """
    Process dataset and add UDCG scores.

    Args:
        dataset: List of items with questions and passages
        model_id: Key to identify model scores
        relevant_weight: Weight for relevant passages
        irrelevant_weight: Weight for irrelevant passages

    Returns:
        Dataset with UDCG scores added
    """
    results = []

    for item_idx, item in enumerate(dataset):
        passages = item.get('passages', [])

        # Create a copy of the item
        result_item = deepcopy(item)

        # Compute UDCG score
        udcg_score = compute_udcg_score(
            passages,
            model_id,
            relevant_weight,
            irrelevant_weight
        )

        # Add UDCG score to item
        update_item_with_udcg_score(
            result_item,
            model_id,
            udcg_score
        )

        print(f"Item {item_idx + 1}/{len(dataset)}: UDCG = {udcg_score:.4f}")

        results.append(result_item)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute UDCG scores for question-passage datasets"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/sample_dataset_with_ds.json",
        help="Path to input JSON file with passages and distracting scores"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sample_output.json",
        help="Path to output JSON file with UDCG scores"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Hugging Face model name (e.g., meta-llama/Llama-3.2-3B-Instruct)"
    )
    parser.add_argument(
        "--relevant_weight",
        type=float,
        default=1.0,
        help="Weight for relevant passages (default: 1.0)"
    )
    parser.add_argument(
        "--irrelevant_weight",
        type=float,
        default=-1/3,
        help="Weight for irrelevant passages (default: -0.333)"
    )

    args = parser.parse_args()

    print(f"Loading dataset from: {args.input}")
    dataset = read_json(args.input)

    print(f"Loaded {len(dataset)} items")
    print(f"Using model: {args.model_id}")
    print(f"Relevant weight: {args.relevant_weight}")
    print(f"Irrelevant weight: {args.irrelevant_weight}")

    # Process dataset
    results = process_dataset(
        dataset,
        args.model_id,
        args.relevant_weight,
        args.irrelevant_weight
    )

    # Save results
    print(f"Saving results to: {args.output}")
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    write_json(results, args.output)

    # Print summary statistics
    udcg_scores = [item['udcg'][args.model_id] for item in results if 'udcg' in item and args.model_id in item['udcg']]

    if udcg_scores:
        import statistics
        print("\nUDCG Score Statistics:")
        print(f"  Mean: {statistics.mean(udcg_scores):.4f}")
        print(f"  Median: {statistics.median(udcg_scores):.4f}")
        print(f"  Min: {min(udcg_scores):.4f}")
        print(f"  Max: {max(udcg_scores):.4f}")
        if len(udcg_scores) > 1:
            print(f"  Std Dev: {statistics.stdev(udcg_scores):.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
