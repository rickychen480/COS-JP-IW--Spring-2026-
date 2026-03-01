# Dynamic Intersectional Bias in LLMs

A comprehensive framework for evaluating dynamic intersectional bias in large language models (LLMs) through multi-agent dialogue simulation and bias metrics analysis. 

## Overview

We test LLMs for allocational and representational bias across demographic dimensions using:
- **Identity Grid**: 5 races × 2 genders × 12 occupations (120 demographic combinations)
- **Task Variants**: Implicit and explicit bias scenarios from high-stakes domains (banking, healthcare, legal, etc.)
- **Multiple Evaluation Methods**: Allocational metrics, representational bias, and semantic embedding analysis
- **Multi-Model Support**: Tested with Llama-3.1-8B-Instruct and Llama-3.1-70B-Instruct-AWQ-INT4

## Project Structure

```
├── controller.py                    # Multi-agent dialogue simulator
├── requirements.txt                 # Python dependencies
├── data_generation/
│   ├── main.py                     # Generate simulation scenarios
│   ├── constants.py                # Identity grid definitions (races, genders, occupations)
│   ├── generators.py               # Scenario generation utilities
│   ├── metalwoz/                   # MetaLWOz dialogue dataset (goal source)
│   ├── generate_sims_llama-8b.slurm   # SLURM job for Llama-8B simulations
│   └── generate_sims_llama-70b.slurm  # SLURM job for Llama-70B simulations
├── metrics/
│   ├── main.py                     # Bias metrics evaluation
│   ├── allocational.py             # Allocational bias metrics
│   ├── representational.py         # Representational bias metrics
│   └── evaluate_bias.slurm         # SLURM job for bias evaluation
├── compost/
│   ├── embeddings/
│   │   └── compost_evaluator.py    # CoMPosT framework to authenticate our user simulator
│   └── judge-llm/
│       ├── compost_evaluator.py    # LLM judge evaluation (outdated)
│       └── compost_eval.slurm      # SLURM job for LLM judge (outdated)
├── data/
│   ├── prompts/                    # Generated simulation scenarios (input)
│   └── transcripts/                # Generated dialogue transcripts (output)
└── results/                        # Evaluation results
```

## Installation

### Prerequisites
- Python 3.12
- CUDA 12.x (for GPU support)
- Sufficient disk space (~100GB for models and transcripts)

### Setup

1. **Clone and navigate to the project:**
   ```bash
   mkdir -p COS-JP
   cd COS-JP
   git clone https://github.com/rickychen480/COS-JP-IW--Spring-2026- .
   ```

2. **Create a Python virtual environment (recommended):**
   ```bash
   conda create --prefix ./env python=3.12 -y
   conda activate ./env
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download required models:**
   - Llama-3.1-8B-Instruct
   - Llama-3.1-70B-Instruct-AWQ-INT4 (quantized version to fit in VRAM)

   ```bash
   hf download meta-llama/Llama-3.1-8B-Instruct --local-dir ./models/Llama-3.1-8B-Instruct
   hf download hugging-quants/Llama-3.1-70B-Instruct-AWQ-INT4 --local-dir ./models/Llama-3.1-70B-Instruct-AWQ-INT4
   ```

5. **Set up logging directory:**
   ```bash
   mkdir -p logs
   ```

## Workflow
### Stage 1: Generate Simulation Scenarios
**Script**: `data_generation/main.py`

Generates synthetic dialogue scenarios with demographic metadata by intersecting identities from WinoBias (reweighted for statistical parity) and tasks from MetaLWoz.

```bash
python data_generation/main.py
```

**Outputs**:
- `data/prompts/target_simulations.json` - Target scenarios with intersectional identities and implicit/explicit bias triggers
- `data/prompts/control_simulations.json` - Control scenarios with neutral identities
- `data/prompts/default_topics.json` - General conversation topics (no tasks) with intersectional identities

**Key Details**:
- Generates 10,000 samples per condition
- Uses high-stakes domains (banking, legal, healthcare, etc.)
- Tests intersectionality: race × gender × occupation
- Variants: implicit bias, explicit bias, and control conditions

### Stage 2: Run Dialogue Simulations
**Script**: `controller.py`

Simulates multi-turn dialogues with a user simulator and target LLM. Uses batch generation across multiple GPUs for efficiency.

```bash
# Run on single model (example with 1000 limit for testing)
CUDA_VISIBLE_DEVICES=0 python controller.py \
  --data data/prompts/target_simulations.json \
  --out data/transcripts/Llama-3.1-8B-Instruct/target_simulations.json \
  --model ./models/Llama-3.1-8B-Instruct \
  --limit 1000

# Run on both control and target scenarios
CUDA_VISIBLE_DEVICES=0 python controller.py \
  --data data/prompts/control_simulations.json \
  --out data/transcripts/Llama-3.1-8B-Instruct/control_simulations.json \
  --model ./models/Llama-3.1-8B-Instruct

CUDA_VISIBLE_DEVICES=0 python controller.py \
  --data data/prompts/default_topics.json \
  --out data/transcripts/Llama-3.1-8B-Instruct/default_topics.json \
  --model ./models/Llama-3.1-8B-Instruct
```

**Arguments**:
- `--data` - Path to input simulation JSON file
- `--out` - Path to output transcript JSON file
- `--model` - Path or ID of LLM model
- `--limit` - Max simulations to run (optional, useful for testing)

**Outputs**:
- Transcript JSON with full dialogue history, speaker labels, semantic metadata

**Key Features**:
- Dual-agent: User simulator + Target model
- Early stopping detection (refusals, goodbyes, dead ends)
- Batch generation for efficiency
- Adjustable sampling parameters

### Stage 3a: Evaluate Bias Metrics
**Script**: `metrics/main.py`

Computes allocational and representational bias metrics using CoMPosT semantic anchors.

```bash
# Evaluate Llama-8B transcripts
python metrics/main.py \
  --dir data/transcripts/Llama-3.1-8B-Instruct \
  --out results/Llama-8B_evaluation_results.csv

# Evaluate Llama-70B transcripts
python metrics/main.py \
  --dir data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4 \
  --out results/Llama-70B_evaluation_results.csv
```

**Arguments**:
- `--dir` - Directory containing control, target, and default_topics transcripts
- `--out` - Path to output CSV results file

**Outputs**:
- CSV with bias metrics per demographic group and condition
- Metrics computed: allocational bias, representational bias, differential goal completion rate

**Key Metrics**:
- **Allocational Bias**: Disparities in task completion rates across demographics
- **Representational Bias**: How demographics are talked about in responses
- **d-GCR**: Differential Goal Completion Rate across intersectional groups

### Stage 3b: CoMPosT Embedding-Based Auditing
**Script**: `compost/embeddings/compost_evaluator.py`

Performs semantic embedding analysis using the CoMPosT framework to audit user simulations for caricature and individuation. This establishes a static baseline for our results and probes our User Simulator Agent for bias.

```bash
# Evaluate Llama-70B with embeddings
python compost/embeddings/compost_evaluator.py \
  --data data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/control_simulations.json \
        data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/default_topics.json \
        data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/target_simulations.json

# Evaluate Llama-8B with embeddings
python compost/embeddings/compost_evaluator.py \
  --data data/transcripts/Llama-3.1-8B-Instruct/control_simulations.json \
        data/transcripts/Llama-3.1-8B-Instruct/default_topics.json \
        data/transcripts/Llama-3.1-8B-Instruct/target_simulations.json
```

**Arguments**:
- `--data` - Multiple JSON paths: control, default_topics, target (in order)

**Outputs**:
- JSON/TXT output with semantic bias vectors and dimensional analysis

## SLURM Job Scripts (For HPC clusters)
### Simulation Generation
```bash
sbatch data_generation/generate_sims_llama-8b.slurm
sbatch data_generation/generate_sims_llama-70b.slurm
```

### Bias Evaluation
```bash
sbatch metrics/evaluate_bias.slurm
```

## Quick Start (Local Testing)

For testing without large models, run with reduced samples:

```bash
# 1. Generate scenarios (quick)
python data_generation/main.py

# 2. Run simulations on small subset
python controller.py \
  --data data/prompts/target_simulations.json \
  --out data/transcripts/test_output.json \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --limit 100

# 3. Evaluate
python metrics/main.py \
  --dir data/transcripts/ \
  --out results/test_results.csv
```

## Output Interpretation

### Transcript Columns
- `dialogue_id` - Unique conversation ID
- `metadata` - Demographics, task description, variant type
- `transcript` - List of turns with speaker, content, turn number
- `variant_type` - "control", "implicit", or "explicit"

### Metrics Results
- **Completion Rate**: % of conversations that achieved the goal
- **Differential Completion Rate**: Difference across demographic groups
- **Response Fairness**: Semantic similarity in responses regardless of demographics

## Contact

Ricky Chen (ricky.chen@princeton.edu)
