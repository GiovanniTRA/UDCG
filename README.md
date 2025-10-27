# UDCG: Utility and Distraction-aware Cumulative Gain

This repository contains code to compute UDCG (Utility and Distraction-aware Cumulative Gain), a metric for evaluating context quality in question-answering tasks with language models.

## Overview

The UDCG metric evaluates how well a set of passages will help a language model answer a question. It combines two key signals:

1. **Relevance labels** - which passages contain the answer
2. **Utility scores** - how passages are expected to help the LLM to answer the query


## Citation

If you use this code in your research, please cite our paper:
**"Redefining Retrieval Evaluation in the Era of LLMs"**
```bibtex
@misc{trappolini2025redefiningretrievalevaluationera,
      title={Redefining Retrieval Evaluation in the Era of LLMs}, 
      author={Giovanni Trappolini and Florin Cuconasu and Simone Filice and Yoelle Maarek and Fabrizio Silvestri},
      year={2025},
      eprint={2510.21440},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.21440}, 
}
```

## Installation

### Requirements

- Python 3.11
- PyTorch
- Transformers (Hugging Face)
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

The pipeline consists of two main steps:

### 1. Compute Abstention Scores

First, compute how likely a language model is to abstain (respond with "NO-RESPONSE") when given each passage individually:

```bash
python src/compute_abstention_probability.py \
  --input data/sample_dataset.json \
  --output data/dataset_with_scores.json \
  --model_id meta-llama/Llama-3.2-3B-Instruct \
```

**Arguments:**
- `--input`: Input JSON file with questions and passages (look at sample_dataset.json to see an example)
- `--output`: Output file to save results
- `--mode_id`: Hugging Face model name
- `--device`: Device to use (auto, cuda, cpu) [default: auto]
- `--dtype`: Model precision (float16, bfloat16, float32) [default: float16]

### 2. Compute UDCG Scores

Then, combine relevance labels (we assume you have already collected passage relevance either through human evaluation or llm-as-a-judge) and abstention scores to compute the UDCG metric:

```bash
python src/compute_udcg_scores.py \
  --input data/dataset_with_scores.json \
  --output data/dataset_with_udcg.json \
  --model_id meta-llama/Llama-3.2-3B-Instruct \
```

**Arguments:**
- `--input`: Input file with abstention scores (output file from step 1)
- `--output`: Output file with UDCG scores
- `--model_id`: Hugging Face model name
- `--relevant_weight`: Weight for relevant passages [default: 1.0]
- `--irrelevant_weight`: Weight for irrelevant passages [default: -0.333]

## Input Data Format

Your input JSON file should contain a list of items with this structure:

```json
[
  {
    "example_id": "example_1",
    "question": "What is the capital of France?",
    "passages": [
      {
        "doc_id": "doc_1",
        "text": "France is a country in Western Europe. Its capital is Paris.",
        "is_relevant": true
      },
      {
        "doc_id": "doc_2",
        "text": "Germany is a country in Central Europe.",
        "is_relevant": false
      }
    ]
  }
]
```

**Required fields:**
- `question`: The question text
- `passages`: List of passage objects, each containing:
  - `text`: The passage content
  - `is_relevant`: Boolean indicating if passage contains the answer

**Optional but recommended fields:**
- `example_id`: Unique identifier for tracking
- `doc_id`: Document identifier for each passage

**Not needed:**
- `answers`, `title`, `rank`, or other metadata fields

## Output Format

After running both scripts, each item will have:

1. **Abstention scores** added to each passage:
```json
{
  "passages": [
    {
      "text": "...",
      "is_relevant": true,
      "models_info": {
        "meta-llama/Llama-3.2-3B-Instruct": {
          "no_res_prob": 0.0234
        }
      }
    }
  ]
}
```

2. **UDCG score** added to each item:
```json
{
  "question": "...",
  "passages": [...],
  "udcg": {
    "meta-llama/Llama-3.2-3B-Instruct": 0.4521,
  }
}
```

## Understanding the Metrics

### Abstention Score (`no_res_prob`)

- Range: 0 to 1
- **Higher** values mean the passage is more likely to cause the model to abstain
- Computed by measuring the probability that the model generates "NO-RESPONSE" when given only that passage

### UDCG Score

- Range: sigmoid range (0,1)
- **Higher** values indicate better context quality
- Computed as: `(mean relevant passage utility) - (weight × mean irrelevant passage utility)`
- Where utility = `1 - no_res_prob`

The metric balances two factors:
- **Relevant passages** with high relevance scores (high utility) increase UDCG
- **Irrelevant passages** with high distracting scores (high dis-utility) decrease UDCG

## Example: Complete Pipeline

```bash
# Step 1: Compute abstention scores
python src/compute_abstention_probability.py \
  --input data/sample_dataset.json \
  --output data/dataset_with_scores.json \
  --model_id meta-llama/Llama-3.2-3B-Instruct

# Step 2: Compute UDCG scores
python src/compute_udcg_scores.py \
  --input data/dataset_with_scores.json \
  --output data/dataset_with_udcg.json \
  --model_id meta-llama/Llama-3.2-3B-Instruct

# Results will be in data/dataset_with_udcg.json
```

## Supported Models

The code works with any Hugging Face causal language model that supports chat templates. Tested models include:

- `meta-llama/Llama-3.2-3B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`
- `meta-llama/Llama-3.3-70B-Instruct`
- `google/gemma-3-4b-it`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `Qwen/Qwen2.5-7B-Instruct`

For questions or issues, please open an issue on GitHub.
