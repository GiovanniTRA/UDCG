"""
Compute NO-RESPONSE probability for Question-Passage Pairs

This script computes the "NO-RESPONSE probability" for each passage
in a dataset. The NO-RESPONSE probability measures how likely a language model is to abstain
from answering a question when given only that passage as context.

Input format:
  A JSON file containing a list of items, each with:
  - "question": the question text
  - "passages": list of passage objects, each with:
    - "text": passage content
    - other optional fields (doc_id, title, etc.)

Output format:
  Same structure as input, but each passage gets additional fields:
  - "no_res_prob": the NO-RESPONSE probability for that passage
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import random
import numpy as np

from utils import seed_everything, read_json, write_json


def format_abstention_prompt(question: str, passage_text: str, tokenizer: AutoTokenizer) -> str:
    """Format prompt for abstention probability calculation with a single passage."""
    document_text = f"Document: {passage_text}\n"

    system_message = (
        "You are given a question and you must respond based on the provided documents. "
        "Respond directly without providing any premise or explanation. "
        "If none of the documents contain the answer, please respond with NO-RESPONSE. "
        "Do not try to respond based on your own knowledge."
    )

    user_message = f"{document_text}Question: {question}\nAnswer:"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return prompt


def compute_no_response_probability(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str
) -> float:
    """
    Compute the probability of generating 'NO-RESPONSE' token.

    This score indicates how likely the model is to abstain from answering based on
    the given context.
    """
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Get the token ID for "NO-RESPONSE"
    no_response_tokens = tokenizer.encode("NO-RESPONSE", add_special_tokens=False)
    if len(no_response_tokens) == 0:
        raise ValueError("Could not tokenize 'NO-RESPONSE'")

    # Use the first token of "NO-RESPONSE"
    no_response_token_id = no_response_tokens[0]

    with torch.no_grad():
        # Get model outputs (logits for next token prediction)
        outputs = model(**inputs)
        logits = outputs.logits

        # Get logits for the last position (next token to generate)
        next_token_logits = logits[0, -1, :]

        # Apply softmax to get probabilities
        probabilities = torch.softmax(next_token_logits, dim=-1)

        # Get probability of "NO-RESPONSE" token
        no_response_prob = probabilities[no_response_token_id].item()

    return no_response_prob


def update_passage_with_no_res_prob(
    passage: dict,
    model_id: str,
    no_res_prob: float
) -> None:
    """Update passage dictionary with no-response probability information."""
    if 'models_info' not in passage:
        passage['models_info'] = {}

    passage['models_info'][model_id] = {
        'no_res_prob': no_res_prob
    }


def process_dataset(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: list[dict],
    model_id: str
) -> list[dict]:
    """Process dataset and add no-response probabilities for each passage."""
    results = []

    for item_idx, item in enumerate(dataset):
        question = item['question']
        passages = item.get('passages', [])

        print(f"Processing item {item_idx + 1}/{len(dataset)}")

        # Create a deep copy of the item
        result_item = json.loads(json.dumps(item))

        # Process each passage
        for passage_idx, passage in enumerate(passages):
            try:
                if 'text' not in passage:
                    raise ValueError("Passage missing 'text' field")

                passage_text = passage.get('text', '')
                prompt = format_abstention_prompt(question, passage_text, tokenizer)
                no_res_prob = compute_no_response_probability(model, tokenizer, prompt)

                update_passage_with_no_res_prob(
                    result_item['passages'][passage_idx],
                    model_id,
                    no_res_prob
                )

            except Exception as e:
                print(f"  Error processing passage {passage_idx}: {e}")
                continue

        results.append(result_item)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute distracting scores (NO-RESPONSE probabilities) for passages"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/sample_dataset.json",
        help="Path to input JSON file with questions and passages"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sample_dataset_with_ds.json",
        help="Path to output JSON file with distracting scores"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Hugging Face model name (e.g., meta-llama/Llama-3.2-3B-Instruct)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype"
    )

    args = parser.parse_args()

    # Set random seed
    seed_everything()

    print(f"Loading model: {args.model_id}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        padding_side="left",
        truncation_side="left"
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    # Load model
    torch_dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map=args.device
    )

    print(f"Loading dataset from: {args.input}")
    dataset = read_json(args.input)
    print(f"Loaded {len(dataset)} items")

    # Process dataset
    results = process_dataset(model, tokenizer, dataset, args.model_id)

    # Save results
    print(f"Saving results to: {args.output}")
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    write_json(results, args.output)

    print("Done!")


if __name__ == "__main__":
    main()
