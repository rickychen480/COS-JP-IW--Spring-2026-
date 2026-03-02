#!/usr/bin/env python3
"""
Merge simulation chunk files from SLURM job array runs.
Run this after all 12 array jobs complete to recombine the transcripts.

Usage:
    python merge_simulations.py --model 70b
    python merge_simulations.py --model 8b
"""

import json
import glob
import argparse
from pathlib import Path


def merge_simulation_chunks(model_name, dataset_type):
    """
    Merge all simulation chunk files for a given model and dataset type.
    
    Args:
        model_name: "70b" or "8b"
        dataset_type: "target_simulations", "control_simulations", or "default_topics"
    """
    if model_name == "70b":
        transcript_dir = Path("data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4")
    elif model_name == "8b":
        transcript_dir = Path("data/transcripts/Llama-3.1-8B-Instruct")
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    merged_data = []
    
    # Find all chunk files for this dataset type
    chunk_pattern = transcript_dir / f"{dataset_type}_chunk_*.json"
    chunk_files = sorted(glob.glob(str(chunk_pattern)))
    
    if not chunk_files:
        print(f"WARNING: No chunk files found for {model_name} {dataset_type}")
        return 0
    
    print(f"Merging {len(chunk_files)} chunks for {model_name} {dataset_type}...")
    
    for chunk_file in chunk_files:
        with open(chunk_file, "r") as f:
            data = json.load(f)
            merged_data.extend(data)
        print(f"  ✓ Loaded {len(data)} items from {Path(chunk_file).name}")
    
    # Save the merged output (replace _chunk_* with final name)
    output_path = transcript_dir / f"{dataset_type}.json"
    with open(output_path, "w") as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"✓ Merged {len(merged_data)} total items into {dataset_type}.json\n")
    return len(merged_data)


def main():
    parser = argparse.ArgumentParser(description="Merge simulation chunks from job arrays")
    parser.add_argument("--model", type=str, choices=["70b", "8b"], required=True,
                        help="Which model to merge (70b or 8b)")
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Merging {args.model.upper()} simulation chunks...")
    print("=" * 60 + "\n")
    
    total_target = merge_simulation_chunks(args.model, "target_simulations")
    total_control = merge_simulation_chunks(args.model, "control_simulations")
    total_default = merge_simulation_chunks(args.model, "default_topics")
    
    print("=" * 60)
    print("MERGE COMPLETE!")
    print("=" * 60)
    if total_target or total_control or total_default:
        print(f"\nFinal {args.model.upper()} Simulation Summary:")
        if total_target:
            print(f"  - target_simulations.json: {total_target} total items")
        if total_control:
            print(f"  - control_simulations.json: {total_control} total items")
        if total_default:
            print(f"  - default_topics.json: {total_default} total items")
        print(f"\nReady to run evaluation pipeline!")


if __name__ == "__main__":
    main()
