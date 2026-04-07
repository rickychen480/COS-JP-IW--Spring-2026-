import json
import pandas as pd
import glob
import os


def merge_csv_results(results_dir="results/metrics/gpt-4o-mini/"):
    """Merges all chunked CSV files into a single file."""
    chunk_files = glob.glob(
        os.path.join(results_dir, "dynamic_bias_results_chunk_*.csv")
    )
    if not chunk_files:
        print("No chunked CSV files found to merge.")
        return

    df_list = [pd.read_csv(file) for file in chunk_files]
    merged_df = pd.concat(df_list, ignore_index=True)

    merged_df.to_csv(os.path.join(results_dir, "dynamic_bias_results.csv"), index=False)
    print(
        f"Successfully merged {len(chunk_files)} CSV files into dynamic_bias_results.csv"
    )


def merge_json_cache(cache_dir="."):
    """Merges all chunked JSON cache files into a single file."""
    chunk_files = glob.glob(os.path.join(cache_dir, "llm_judge_cache_chunk_*.json"))
    if not chunk_files:
        print("No chunked JSON files found to merge.")
        return

    merged_cache = {}
    for file in chunk_files:
        with open(file, "r") as f:
            data = json.load(f)
            merged_cache.update(data)

    with open(os.path.join(cache_dir, "llm_judge_cache.json"), "w") as f:
        json.dump(merged_cache, f, indent=2)

    print(
        f"Successfully merged {len(chunk_files)} JSON cache files into llm_judge_cache.json"
    )


if __name__ == "__main__":
    merge_csv_results()
    merge_json_cache()
