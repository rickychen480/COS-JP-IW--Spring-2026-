"""
CoMPosT Evaluator

Implements bias audit methodology to measure whether LLMs produce
caricatured outputs for intersectional personas.

Key Components:
1. Individuation Measurement: Binary classifier using scenario-disjoint cross-validation 
   to test if S_{p,t,c} is differentiable from S_{_,t,c} (target vs default-persona simulation).
   Can also use grouped holdout (80/20 split) when evaluation_mode='grouped_holdout'.
   
2. Exaggeration Measurement: Fightin' Words + semantic axes to measure if S_{p,t,c}
   has more persona-defining characteristics than S_{p,_,c} or S_{_,t,c}

Usage:
python compost/compost_evaluator.py \
    --data data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/control_simulations.json \
        data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/default_topics.json \
        data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/target_simulations_masked.json \
    --output-dir results/compost/
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import orjson
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from concurrent.futures import ProcessPoolExecutor
import logging

from intersectional_evaluator import IntersectionalEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download("punkt_tab", quiet=True)

def _process_single_file(path):
    """Helper function to process a single JSON file in a separate CPU process."""
    logger.info(f"Worker started loading {path}...")
    
    with open(path, "rb") as f:
        data = orjson.loads(f.read())
        
    rows = []
    for d in data:
        # Fast dictionary lookups
        meta = d.get("metadata", {})
        persona_dict = meta.get("persona", {})

        demo_val = persona_dict.get("demographic") or persona_dict.get("race") or "Unmarked"
        gender = persona_dict.get("gender", "Unmarked")
        occupation = persona_dict.get("occupation", "Unmarked")

        p_str = "Unmarked" if demo_val == "Unmarked" else f"{demo_val} {gender} {occupation}"

        # Flatten all "User" dialogue turns into a single string for analysis
        user_turns = [t for t in d.get("transcript", []) if t.get("speaker") == "User"]
        user_text = " ".join(t.get("content", "") for t in user_turns)
        masking_applied = any(t.get("masking_applied", False) for t in user_turns)

        rows.append({
            "persona": p_str,
            "race": demo_val,
            "gender": gender,
            "occupation": occupation,
            "topic": meta.get("task_description"),
            "response": user_text,
            "variant_type": d.get("variant_type", "implicit"),
            "scenario_id": meta.get("scenario_id", "unknown"),
            "dialogue_id": d.get("dialogue_id", "unknown"),
            "masking_applied": masking_applied,
        })
        
    return pd.DataFrame(rows)

def load_transcripts_to_dataframe(json_paths):
    """Parses multiple JSON files into a single Pandas DataFrame using multiprocessing.
    
    The returned DataFrame contains separate columns for each demographic
    axis (race, gender, occupation), variant_type, and scenario_id for 
    scenario-disjoint cross-validation.
    
    Args:
        json_paths: List of paths to JSON transcript files
    """
    
    dataframes = []
    
    # Process the files simultaneously using multiple CPU cores
    with ProcessPoolExecutor(max_workers=min(len(json_paths), 4)) as executor:
        for result_df in executor.map(_process_single_file, json_paths):
            dataframes.append(result_df)
            
    # Concatenate all resulting DataFrames efficiently at the very end
    df = pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()

    # Validate scenario_ids
    if "scenario_id" in df.columns and not df.empty:
        df["scenario_id"] = df["scenario_id"].astype(str)
        unique_scen = df["scenario_id"].nunique()
        unique_topics = df["topic"].nunique()
        if unique_scen <= 1:
            logger.warning(f"All scenario_id values are identical or missing. "
                           "Scenario-disjoint CV will collapse to a single fold.")
        else:
            # Check for 1-to-1 mapping between scenarios and topics
            scenario_to_topics = df.groupby("scenario_id")["topic"].nunique()
            topics_to_scenarios = df.groupby("topic")["scenario_id"].nunique()
            
            if scenario_to_topics.max() == 1 and topics_to_scenarios.max() == 1:
                logger.warning("Detected 1-to-1 mapping between scenario_id and topic.")
                logger.warning(f"Each of {unique_scen} scenarios maps to exactly one topic.")
                logger.warning("Scenario-disjoint CV will be equivalent to topic-disjoint CV.")
                logger.warning("Model generalization will not be evaluated across scenarios within the same topic.")
            
            # Check for small scenarios that may cause folding issues
            scenario_counts = df["scenario_id"].value_counts()
            small_scenarios = scenario_counts[scenario_counts < 5]
            if len(small_scenarios) > 0:
                logger.warning(f"Detected {len(small_scenarios)} scenarios with < 5 samples: {small_scenarios.to_dict()}")
                logger.warning("GroupKFold may skip these scenarios if they're too small relative to n_splits.")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CoMPosT Evaluator")
    parser.add_argument(
        "--data",
        type=str,
        nargs="+",
        required=True,
        help="Paths to the transcript JSONs (control, default_topics, target)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./compost_results",
        help="Directory to save detailed results",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir.resolve()}")

    logger.info("=" * 80)
    logger.info("CoMPosT EVALUATOR - Bias Audit Methodology")
    logger.info("=" * 80)
    logger.info("Individuation: scenario-disjoint cross-validation (Random Forest classifier)")
    logger.info("Exaggeration: Fightin' Words + semantic axes (z-score > 1.96)")
    logger.info("=" * 80)

    logger.info("1. Loading Data...")
    df = load_transcripts_to_dataframe(args.data)
    logger.info(f"Loaded {len(df)} total transcripts.")
    
    intersectional_evaluator = IntersectionalEvaluator()
    df = intersectional_evaluator.add_intersectional_column(df)
    logger.info(f"Created intersectional IDs for {df['intersectional_id'].nunique()} groups")
    
    # Warn if sparse groups detected
    group_counts = df["intersectional_id"].value_counts()
    sparse_groups = group_counts[group_counts < 30]
    if len(sparse_groups) > 0:
        logger.warning(f"\n!!! WARNING: {len(sparse_groups)} intersectional groups have < 30 samples !!!")
        logger.warning("This may lead to noisy results in scenario-disjoint CV folds.")

    logger.info("2. Tokenizing sentences...")
    df["sentences"] = df["response"].apply(sent_tokenize)

    logger.info("3. Generating Sentence-BERT Embeddings & Caching...")
    model = SentenceTransformer("all-mpnet-base-v2")
    
    # Pre-encode and cache all unique sentences
    all_sentences = set()
    for sents in df["sentences"]:
        all_sentences.update(sents)
    all_sentences = list(all_sentences)
    
    logger.info(f"Encoding {len(all_sentences)} unique sentences...")
    sentence_embeddings = model.encode(all_sentences, show_progress_bar=True)
    emb_dict = dict(zip(all_sentences, sentence_embeddings))

    # Calculate document-level embeddings via mean-pooling
    def get_doc_embedding(sents):
        if not sents:
            return np.zeros(model.get_sentence_embedding_dimension())
        return np.mean([emb_dict[s] for s in sents], axis=0)

    df["embedding"] = df["sentences"].apply(get_doc_embedding)

    ie = IntersectionalEvaluator()
    X = np.stack(df["embedding"].values)
    
    logger.info("4. Measuring Intersectional Individuation...")
    perf_df = ie.measure_individuation(df, X)
    perf_path = output_dir / "individuation.csv"
    perf_df.to_csv(perf_path, index=False)
    logger.info(f"Saved intersectional individuation to {perf_path}")

    logger.info("5. Measuring Intersectional Exaggeration...")
    exag_df = ie.measure_exaggeration(df, emb_dict, metric="fighting_words")
    exag_path = output_dir / "exaggeration.csv"
    exag_df.to_csv(exag_path, index=False)
    logger.info(f"Saved intersectional exaggeration to {exag_path}")

    comp_df = ie.compare_implicit_vs_explicit(df, X)
    comp_path = output_dir / "implicit_explicit_comparison.csv"
    comp_df.to_csv(comp_path, index=False)
    logger.info(f"Saved implicit vs explicit comparison to {comp_path}")

    # Generate final intersectional disparity reports
    logger.info("\n6. Generating intersectional disparity analysis...")
    
    # Exaggeration Parity Analysis
    if not exag_df.empty and 'exaggeration' in exag_df.columns:
        exag_parity = ie.compute_intersectional_parity(
            performance_df=exag_df,
            metric='exaggeration'
        )
        exag_report = ie.generate_intersectional_report(
            performance_df=exag_df,
            parity_metrics=exag_parity,
            metric_name='exaggeration'
        )
        exag_report_path = output_dir / "exaggeration_report.txt"
        with open(exag_report_path, 'w') as f:
            f.write(exag_report)
        logger.info(f"Saved exaggeration parity report to {exag_report_path}")
    
    # Individuation Parity Analysis
    if not perf_df.empty and 'f1_score' in perf_df.columns:
        perf_parity = ie.compute_intersectional_parity(
            performance_df=perf_df,
            metric='f1_score'
        )
        perf_report = ie.generate_intersectional_report(
            performance_df=perf_df,
            parity_metrics=perf_parity,
            metric_name='f1_score'
        )
        perf_report_path = output_dir / "individuation_report.txt"
        with open(perf_report_path, 'w') as f:
            f.write(perf_report)
        logger.info(f"Saved performance parity report to {perf_report_path}")

    summary_path = output_dir / "summary_report.txt"
    with open(summary_path, 'w') as f:
        f.write(f"CoMPosT Evaluation Complete at {output_dir.resolve()}\n")
        f.write(f"Timestamp: {pd.Timestamp.now()}\n")
    logger.info(f"\nSaved summary report to summary_report.txt")
    
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info(f"All outputs saved to: {output_dir.resolve()}")
    logger.info("=" * 80)