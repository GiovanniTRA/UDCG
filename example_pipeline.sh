#!/bin/bash

# Example pipeline for computing UDCG scores
# This script demonstrates the complete workflow

set -e  # Exit on error

echo "======================================"
echo "UDCG Score Computation Pipeline"
echo "======================================"
echo ""

# Configuration
INPUT_FILE="data/sample_dataset.json"
SCORES_FILE="data/dataset_with_scores.json"
FINAL_FILE="data/dataset_with_udcg.json"
MODEL="meta-llama/Llama-3.2-3B-Instruct"
MODEL_KEY="llama-3.2-3b"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE not found!"
    echo "Please ensure the sample dataset exists or create your own."
    exit 1
fi

echo "Step 1: Computing distracting scores..."
echo "Model: $MODEL"
echo "Input: $INPUT_FILE"
echo "Output: $SCORES_FILE"
echo ""

python src/compute_distracting_scores.py \
    --input "$INPUT_FILE" \
    --output "$SCORES_FILE" \
    --model "$MODEL" \
    --model_key "$MODEL_KEY" \
    --device auto \
    --dtype float16

echo ""
echo "Step 1 completed!"
echo ""

echo "Step 2: Computing UDCG scores..."
echo "Input: $SCORES_FILE"
echo "Output: $FINAL_FILE"
echo ""

python src/compute_udcg_scores.py \
    --input "$SCORES_FILE" \
    --output "$FINAL_FILE" \
    --model_key "$MODEL_KEY" \
    --relevant_weight 1.0 \
    --irrelevant_weight -0.333

echo ""
echo "======================================"
echo "Pipeline completed successfully!"
echo "======================================"
echo ""
echo "Results saved to: $FINAL_FILE"
echo ""
echo "You can now inspect the UDCG scores in the output file."
