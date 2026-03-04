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
├── merge_simulations.py             # Merge transcript chunks from distributed runs
├── requirements.txt                 # Python dependencies
├── SIMULATION_JOB_ARRAY.md          # Guide for SLURM job array submissions
├── data_generation/
│   ├── main.py                     # Generate simulation scenarios
│   ├── constants.py                # Identity grid definitions
│   ├── generators.py               # Scenario generation utilities
│   ├── metalwoz/                   # MetaLWOz dialogue dataset
│   ├── generate_sims_llama-8b.slurm   # SLURM job for Llama-8B simulations
│   └── generate_sims_llama-70b.slurm  # SLURM job for Llama-70B simulations
├── metrics/
│   ├── main.py                     # Bias metrics evaluation
│   ├── allocational.py             # Allocational bias metrics
│   ├── representational.py         # Representational bias metrics
│   └── evaluate_bias.slurm         # SLURM job for bias evaluation
├── compost/
│   ├── embeddings/
│   │   ├── compost_evaluator.py    # CoMPosT framework to validate our user simulator
│   │   ├── axis_metrics.py         # Dimensional axis metrics
│   │   ├── intersectional_evaluator.py # Intersectional semantic evaluation
│   │   └── semantic_masking.py     # Semantic masking utilities
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

### Stage 2a: Run Dialogue Simulations
**Script**: `controller.py`

Simulates multi-turn dialogues with a user simulator and target LLM. For large-scale runs, this is parallelized using SLURM job arrays (see `SIMULATION_JOB_ARRAY.md`).

```bash
# Example for a single, non-parallelized run
CUDA_VISIBLE_DEVICES=0 python controller.py \
  --data data/prompts/target_simulations.json \
  --out data/transcripts/Llama-3.1-8B-Instruct/target_simulations.json \
  --model ./models/Llama-3.1-8B-Instruct \
  --limit 1000

# Example for a distributed run (e.g., inside a SLURM job script)
# This command processes chunk 0 out of 12 total chunks.
python controller.py \
  --data data/prompts/target_simulations.json \
  --out data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/target_simulations_chunk_0.json \
  --model ./models/Llama-3.1-70B-Instruct-AWQ-INT4 \
  --quant awq \
  --tp 2 \
  --chunk_index 0 \
  --total_chunks 12
```

**Arguments**:
- `--data`: Path to input simulation JSON.
- `--out`: Path to output transcript JSON.
- `--model`: Path or ID of the LLM model.
- `--quant`: Quantization method (e.g., `awq`).
- `--tp`: Tensor parallel size (number of GPUs).
- `--chunk_index`: Index of the current data chunk for parallel processing.
- `--total_chunks`: Total number of chunks the data is split into.
- `--limit`: Max simulations to run (for testing).

**Outputs**:
- Transcript JSON with full dialogue history, speaker labels, semantic metadata

**Key Features**:
- Dual-agent: User simulator + Target model
- Early stopping detection (refusals, goodbyes, dead ends)
- Batch generation for efficiency
- Adjustable sampling parameters

### Stage 2b: Merge Simulation Chunks
**Script**: `merge_simulations.py`

If you ran simulations in parallel, this script merges the resulting transcript chunks back into single files.

```bash
# Merge chunks for the 70B model
python merge_simulations.py --model 70b

# Merge chunks for the 8B model
python merge_simulations.py --model 8b
```

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
- CSV with bias metrics per demographic group and condition.
- Key Metrics: `d_GCR` (differential goal completion), `d_CCD` (differential self-confidence), `Steering_Score`.
- **Allocational Bias**: Disparities in task completion rates across demographics
- **Representational Bias**: How demographics are talked about in responses

### Stage 3b: CoMPosT Embedding-Based Auditing
**Script**: `compost/embeddings/compost_evaluator.py`

Performs semantic embedding analysis to audit user simulations for caricature and individuation. Features include:
- **Semantic Masking**: NER-based redaction of explicit identity mentions
- **Scenario-Disjoint CV**: GroupKFold prevents data leakage between train/test
- **Intersectional Joint Probabilities**: Treats demographic identities as indivisible units

```bash
# Evaluate Llama-8B with embeddings (single and intersectional axes)
python compost/embeddings/compost_evaluator.py \
  --data data/transcripts/Llama-3.1-8B-Instruct/control_simulations.json \
        data/transcripts/Llama-3.1-8B-Instruct/default_topics.json \
        data/transcripts/Llama-3.1-8B-Instruct/target_simulations.json \
  --enable-semantic-masking \
  --cv-strategy GroupKFold \
  --enable-single-axes-eval \
  --enable-intersectional-eval \
  --output-dir results/compost/embeddings/llama-8b

# Evaluate Llama-70B with embeddings
python compost/embeddings/compost_evaluator.py \
  --data data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/control_simulations.json \
        data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/default_topics.json \
        data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/target_simulations.json \
  --enable-semantic-masking \
  --cv-strategy GroupKFold \
  --enable-single-axes-eval \
  --enable-intersectional-eval \
  --output-dir results/compost/embeddings/llama-70b

# Single-axes evaluation only (legacy approach, may be inflated by confounders)
python compost/embeddings/compost_evaluator.py \
  --data data/transcripts/Llama-3.1-8B-Instruct/control_simulations.json \
        data/transcripts/Llama-3.1-8B-Instruct/default_topics.json \
        data/transcripts/Llama-3.1-8B-Instruct/target_simulations.json \
  --output-dir results/compost/embeddings/llama-8b-single-axes
```

**Arguments**:
- `--data` - Multiple JSON paths: control, default_topics, target (required, in order)
- `--enable-semantic-masking` - Flag to redact explicit demographic/occupational labels in user simulator responses
- `--cv-strategy` - Cross-validation strategy: `random` (legacy), `GroupKFold` (scenario-disjoint, recommended), or `LeaveOneGroupOut`
- `--enable-single-axes-eval` - By default, evaluates each demographic axis individually (race, gender, occupation). Omit flag to skip (faster, for broad scans)
- `--enable-intersectional-eval` - Flag to perform intersectional evaluation treating demographic identities as indivisible combinations
- `--output-dir` - Directory to save detailed results and performance metrics (default: `./compost_results`)

**Outputs**:
- JSON/TXT output with semantic bias vectors and dimensional analysis

## SLURM Job Scripts (For HPC clusters)

Refer to `SIMULATION_JOB_ARRAY.md` for detailed instructions on using SLURM for large-scale runs.

1.  **Generate Scenarios**: `sbatch data_generation/generate_sims_llama-8b.slurm`
2.  **Run Simulations**: The `controller.py` script is run via a job array.
3.  **Merge Transcripts**: `python merge_simulations.py --model <70b|8b>`
4.  **Evaluate Bias**: `sbatch metrics/evaluate_bias.slurm`

## Quick Start (Local Testing)

For testing without large models, run with reduced samples:

```bash
# 1. Generate scenarios
python data_generation/main.py

# 2. Run simulations on a small subset
python controller.py \
  --data data/prompts/target_simulations.json \
  --out data/transcripts/test_output.json \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --limit 100

# 3. Evaluate (assuming you have control and default_topics transcripts)
# Note: For a real evaluation, you need all three transcript types.
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
