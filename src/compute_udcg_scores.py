"""
Compute UDCG (Utility and Distraction-aware Cumulative Gain) Scores

This script computes UDCG scores for question-passage pairs. The UDCG metric
evaluates context quality by combining relevance information with distracting
scores to predict how well a language model will perform on a question-answering task.

The metric uses:
- Relevance labels (is_relevant: True/False) for each passage
- Distracting scores (no_res_prob) that indicate how likely the model is to abstain
  when given only that passage

The UDCG score is computed as:
  UDCG = (mean of (1 - no_res_prob) for relevant passages) -
         (weight * mean of (1 - no_res_prob) for irrelevant passages)

Input format:
  A JSON file containing a list of items, each with:
  - "question": the question text
  - "passages": list of passage objects, each with:
    - "is_relevant": boolean indicating if passage is relevant
    - "no_res_prob_{model_key}": distracting score for that model
    - other optional fields

Output format:
  A JSON file with the same structure, plus additional fields:
  - "udcg_{model_key}": the UDCG score for that item
"""

import json
import argparse
import os
from typing import List, Dict
import math


def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def compute_udcg_score(passages: List[Dict], model_key: str,
                       relevant_weight: float = 1.0,
                       irrelevant_weight: float = -1/3) -> float:
    """
    Compute UDCG (Utility and Distraction-aware Cumulative Gain) score.

    Args:
        passages: List of passage dictionaries with 'is_relevant' and distracting scores
        model_key: Key to identify which distracting score to use (e.g., 'llama-3.2-3b')
        relevant_weight: Weight for relevant passages (default: 1.0)
        irrelevant_weight: Weight for irrelevant passages (default: -1/3)

    Returns:
        UDCG score (higher is better)
    """
    if len(passages) == 0:
        return 0.0

    no_res_prob_key = f"no_res_prob_{model_key}"

    score = 0.0
    for passage in passages:
        # Check if passage has the required distracting score
        if no_res_prob_key not in passage:
            print(f"Warning: Passage missing {no_res_prob_key}, skipping")
            continue

        is_relevant = passage.get('is_relevant', False)
        no_res_prob = passage[no_res_prob_key]

        # Skip if score is None
        if no_res_prob is None:
            continue

        # Convert distracting score to utility score (1 - no_res_prob)
        doc_score = 1.0 - no_res_prob

        # Apply weights based on relevance
        if is_relevant:
            score += doc_score * relevant_weight
        else:
            score += doc_score * irrelevant_weight

    # Average over all passages
    score = score / len(passages)

    return sigmoid(score)


def process_dataset(
    dataset: List[Dict],
    model_key: str,
    relevant_weight: float = 1.0,
    irrelevant_weight: float = -1/3
) -> List[Dict]:
    """
    Process dataset and add UDCG scores.

    Args:
        dataset: List of items with questions and passages
        model_key: Key to identify model scores
        relevant_weight: Weight for relevant passages
        irrelevant_weight: Weight for irrelevant passages

    Returns:
        Dataset with UDCG scores added
    """
    results = []

    for item_idx, item in enumerate(dataset):
        passages = item.get('passages', [])

        # Create a copy of the item
        result_item = json.loads(json.dumps(item))

        # Compute UDCG score
        udcg_score = compute_udcg_score(
            passages,
            model_key,
            relevant_weight,
            irrelevant_weight
        )

        # Add UDCG score to item
        udcg_key = f"udcg_{model_key}"
        result_item[udcg_key] = udcg_score

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
        default="../data/sample_dataset_with_ds.json",
        help="Path to input JSON file with passages and distracting scores"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../data/sample_output.json",
        help="Path to output JSON file with UDCG scores"
    )
    parser.add_argument(
        "--model_key",
        type=str,
        default="llama-3.2-3b",
        help="Model key to identify which distracting scores to use (e.g., llama-3.2-3b). Must match the one used in the previous step"
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
    with open(args.input, 'r') as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} items")
    print(f"Using model_key: {args.model_key}")
    print(f"Relevant weight: {args.relevant_weight}")
    print(f"Irrelevant weight: {args.irrelevant_weight}")

    # Process dataset
    results = process_dataset(
        dataset,
        args.model_key,
        args.relevant_weight,
        args.irrelevant_weight
    )

    # Save results
    print(f"Saving results to: {args.output}")
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary statistics
    udcg_key = f"udcg_{args.model_key}"
    udcg_scores = [item[udcg_key] for item in results if udcg_key in item]

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
