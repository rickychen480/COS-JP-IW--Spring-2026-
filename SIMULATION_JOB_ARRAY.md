# SLURM Job Array Parallel Simulations

This guide explains how to parallelize the LLM simulations (controller.py) across 12 independent SLURM jobs, reducing computation time significantly.

## Workflow

1. **Generate prompts** (unchanged): `python data_generation/main.py`
2. **Run parallel simulations** (new): Submit job array scripts
3. **Merge results** (new): Run merge script to recombine chunks

## Quick Start

### Step 1: Generate Prompts

First, generate all prompts using the original script:
```bash
python data_generation/main.py
```

This creates:
- `data/prompts/target_simulations.json`
- `data/prompts/control_simulations.json`
- `data/prompts/default_topics.json`

### Step 2: Submit Job Array for Simulations

For **70B model** (12 parallel simulation chunks):
```bash
sbatch data_generation/generate_sims_llama-70b.slurm
```

For **8B model** (12 parallel simulation chunks):
```bash
sbatch data_generation/generate_sims_llama-8b.slurm
```

Each job array will output:
- Job 0: `target_simulations_chunk_0.json`, `control_simulations_chunk_0.json`, `default_topics_chunk_0.json`
- Job 1: `target_simulations_chunk_1.json`, `control_simulations_chunk_1.json`, `default_topics_chunk_1.json`
- ... (12 jobs total)

### Step 3: Monitor Progress

Check all job statuses:
```bash
squeue -j <JOB_ID>
```

View specific job output:
```bash
tail logs/sim_70b_<JOB_ID>_0.out     # Job 0
tail logs/sim_70b_<JOB_ID>_5.out     # Job 5, etc.
```

### Step 4: Merge Simulation Chunks

After all 12 jobs complete:

For **70B model**:
```bash
python merge_simulations.py --model 70b
```

For **8B model**:
```bash
python merge_simulations.py --model 8b
```

This merges all chunks back into:
- `data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/target_simulations.json`
- `data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/control_simulations.json`
- `data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/default_topics.json`

(or for 8B: `data/transcripts/Llama-3.1-8B-Instruct/`)

## How It Works

### Per-Job Processing

Each job processes a chunk of each dataset sequentially:
- Job 0 runs scenarios 0-999 from target, then 0-999 from control, then 0-999 from default
- Job 1 runs scenarios 1000-1999 from target, then 1000-1999 from control, then 1000-1999 from default
- etc.

### Chunk Calculation

For N scenarios and 12 jobs:
```
chunk_size = ceil(N / 12)
Job K processes scenarios: K * chunk_size to (K + 1) * chunk_size - 1
```

Example for target_simulations (14,000 scenarios):
```
chunk_size = ceil(14000 / 12) = 1167
Job 0: scenarios 0-1166
Job 1: scenarios 1167-2333
...
Job 11: scenarios 13000-13999
```

## Resource Allocation

### 70B Model Array
- Per job: 1 GPU, 8 CPUs, 48GB RAM
- Wall time: 10 hours per chunk
- Total elapsed: ~10 hours (parallel, vs ~30 hours sequential)

### 8B Model Array
- Per job: 1 GPU, 8 CPUs, 32GB RAM
- Wall time: 6 hours per chunk
- Total elapsed: ~6 hours (parallel, vs ~18 hours sequential)

## File Structure

During execution:
```
data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/
├── target_simulations_chunk_0.json
├── target_simulations_chunk_1.json
├── ...
├── target_simulations_chunk_11.json
├── control_simulations_chunk_0.json
├── ...
└── default_topics_chunk_11.json
```

After merging:
```
data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/
├── target_simulations.json          (merged)
├── control_simulations.json         (merged)
├── default_topics.json              (merged)
├── target_simulations_chunk_*.json  (original chunks, can delete)
├── control_simulations_chunk_*.json
└── default_topics_chunk_*.json
```

## Running Sequential Mode (Original)

To run simulations on a single node without job arrays:
```bash
sbatch data_generation/generate_sims_llama-70b.slurm_sequential
sbatch data_generation/generate_sims_llama-8b.slurm_sequential
```

(Note: Original SLURM scripts are preserved with `_sequential` suffix if you prefer the old approach)

## Troubleshooting

### Cancel All Array Jobs
```bash
scancel <JOB_ID>
```

### Resubmit Single Failed Chunk
```bash
sbatch --array=5 data_generation/generate_sims_llama-70b.slurm
```

### Re-merge After Recovering Chunks
```bash
python merge_simulations.py --model 70b
```

### Check Chunk Output
```bash
# Count scenarios in chunk 0
python -c "import json; print(len(json.load(open('data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/target_simulations_chunk_0.json'))))"
```

## Performance Comparison

### Original (Single Node, 3 Simulations in Parallel)
- 70B: ~30 hours (3 GPUs running simultaneously)
- 8B: ~18 hours (3 GPUs running simultaneously)

### New (12-Job Array)
- 70B: ~10 hours (12 jobs sequentially processing chunks)
- 8B: ~6 hours (12 jobs sequentially processing chunks)

**Speedup: 3x faster** (trades parallelism within a job for parallelism across jobs)

## Next Steps

After merging, proceed with evaluation:
```bash
python metrics/main.py
python compost/judge-llm/compost_evaluator.py
```
