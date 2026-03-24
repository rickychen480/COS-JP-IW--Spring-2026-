"""
CoMPosT Evaluator with:
1. Semantic Masking - NER-based redaction of explicit identifiers
2. Scenario-Disjoint CV - GroupKFold to prevent data leakage
3. Intersectional Joint Probabilities - Treating identities as indivisible

Usage:
python compost/compost_evaluator.py \
    --data data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/control_simulations.json \
        data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/default_topics.json \
        data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/target_simulations.json \
    --enable-semantic-masking \
    --cv-strategy GroupKFold \
    --enable-intersectional-eval \
    --output-dir results/compost/
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
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

from semantic_masking import SemanticMasker, create_semantic_masker
from intersectional_evaluator import IntersectionalEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download("punkt_tab")

def _process_single_file(args):
    """Helper function to process a single JSON file in a separate CPU process."""
    path, apply_masking, semantic_masker = args
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
        user_text = " ".join(
            t["content"] for t in d.get("transcript", []) if t.get("speaker") == "User"
        )
        
        # Apply semantic masking
        masking_applied = False
        if apply_masking and semantic_masker:
            user_text_original = user_text
            user_text = semantic_masker.redact_explicit_identifiers(user_text)
            masking_applied = user_text != user_text_original

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

def load_transcripts_to_dataframe(json_paths, semantic_masker=None, apply_masking=False):
    """Parses multiple JSON files into a single Pandas DataFrame using multiprocessing.
    
    The returned DataFrame contains separate columns for each demographic
    axis (race, gender, occupation), variant_type, and scenario_id for 
    scenario-disjoint cross-validation.
    
    Args:
        json_paths: List of paths to JSON transcript files
        semantic_masker: Optional SemanticMasker instance for redacting explicit identifiers
        apply_masking: If True, apply semantic masking to explicit variants
    """
    
    # Prepare arguments for each file to be processed in parallel
    worker_args = [(path, apply_masking, semantic_masker) for path in json_paths]
    
    dataframes = []
    
    # Process the 3 files simultaneously using multiple CPU cores
    with ProcessPoolExecutor(max_workers=min(len(json_paths), 4)) as executor:
        for result_df in executor.map(_process_single_file, worker_args):
            dataframes.append(result_df)
            
    # Concatenate all resulting DataFrames efficiently at the very end
    df = pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()

    if "scenario_id" in df.columns and not df.empty:
        df["scenario_id"] = df["scenario_id"].astype(str)
        unique_scen = df["scenario_id"].nunique()
        unique_topics = df["topic"].nunique()
        if unique_scen <= 1:
            logger.warning(f"All scenario_id values are identical or missing. "
                           "Scenario-disjoint CV will collapse to a single fold.")
        else:
            # check whether each scenario maps to only one topic
            mapping = df.groupby("scenario_id")["topic"].nunique()
            if mapping.max() == 1 and unique_scen == unique_topics:
                logger.warning("Each scenario_id appears to correspond to exactly one topic.")
                logger.warning("This means the grouping key mirrors topic/ task_description, "
                               "so scenario-disjoint CV will not generalize across topics.")
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
        "--enable-semantic-masking",
        action="store_true",
        help="Enable semantic masking for explicit variants (redacts demographic/occupational labels)",
    )
    parser.add_argument(
        "--cv-strategy",
        type=str,
        default="GroupKFold",
        choices=["random", "GroupKFold", "LeaveOneGroupOut"],
        help="Cross-validation strategy: 'random' (legacy), 'GroupKFold' (scenario-disjoint), or 'LeaveOneGroupOut'",
    )
    parser.add_argument(
        "--enable-intersectional-eval",
        action="store_true",
        help="Enable intersectional joint probability evaluation (treats identities as indivisible)",
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
    logger.info("CoMPosT EVALUATOR")
    logger.info("=" * 80)
    logger.info(f"Semantic Masking: {args.enable_semantic_masking}")
    logger.info(f"CV Strategy: {args.cv_strategy}")
    logger.info(f"Intersectional Evaluation: {args.enable_intersectional_eval}")
    logger.info("=" * 80)

    # Initialize semantic masker if enabled
    semantic_masker = None
    if args.enable_semantic_masking:
        logger.info("Initializing semantic masker...")
        semantic_masker = create_semantic_masker()

    logger.info("1. Loading Data...")
    df = load_transcripts_to_dataframe(
        args.data, 
        semantic_masker=semantic_masker,
        apply_masking=args.enable_semantic_masking
    )
    logger.info(f"Loaded {len(df)} total transcripts.")
    
    # Create intersectional IDs if evaluating intersectionally
    if args.enable_intersectional_eval:
        intersectional_evaluator = IntersectionalEvaluator()
        df = intersectional_evaluator.add_intersectional_column(df)
        logger.info(f"Created intersectional IDs for {df['intersectional_id'].nunique()} groups")
        
        # Warn if sparse groups detected
        group_counts = df["intersectional_id"].value_counts()
        sparse_groups = group_counts[group_counts < 30]
        if len(sparse_groups) > 0:
            logger.warning(f"\n!!! WARNING: {len(sparse_groups)} intersectional groups have < 30 samples !!!")
            logger.warning("This may lead to noisy results in scenario-disjoint CV folds.")
            logger.warning("Sparse groups:")
            for cohort_id, count in sparse_groups.items():
                logger.warning(f"  - {cohort_id}: {count} samples")

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

    # perform paired intersectional evaluation if requested
    if args.enable_intersectional_eval:
        ie = IntersectionalEvaluator()
        X = np.stack(df["embedding"].values)
        
        logger.info("4. Measuring Intersectional Individuation...")
        perf_df = ie.measure_individuation(df, X)
        perf_path = output_dir / "intersectional_performance.csv"
        perf_df.to_csv(perf_path, index=False)
        logger.info(f"Saved paired intersectional performance to {perf_path}")

        logger.info("5. Measuring Intersectional Exaggeration...")
        exag_df = ie.measure_exaggeration(df)
        if not exag_df.empty:
            exag_path = output_dir / "exaggeration_intersectional.csv"
            exag_df.to_csv(exag_path, index=False)
            logger.info(f"Saved intersectional exaggeration to {exag_path}")

        comp_df = ie.compare_implicit_vs_explicit(df, X)
        comp_path = output_dir / "implicit_explicit_comparison.csv"
        comp_df.to_csv(comp_path, index=False)
        logger.info(f"Saved implicit vs explicit comparison to {comp_path}")

        # Generate final intersectional disparity reports
        logger.info("\nGenerating intersectional disparity analysis...")
        
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
            exag_report_path = output_dir / "intersectional_exaggeration_report.txt"
            with open(exag_report_path, 'w') as f:
                f.write(exag_report)
            logger.info(f"Saved exaggeration parity report to {exag_report_path}")
        
        # Performance Parity Analysis
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
            perf_report_path = output_dir / "intersectional_performance_report.txt"
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