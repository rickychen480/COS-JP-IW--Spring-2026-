import os
import sys
import multiprocessing as mp

# Force 'spawn' start method before any other heavy libraries load.
# This prevents the "CUDA driver initialization failed" fork-context error.
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import argparse
import json
import nltk
import numpy as np
import pandas as pd
import re

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

from allocational import AllocationalEvaluator
from representational import RepresentationalEvaluator
from compost.embeddings.intersectional_evaluator import IntersectionalEvaluator
from compost.embeddings.semantic_masking import SemanticMasker
from compost.embeddings.axis_metrics import get_seed_words


def get_document_embedding(text, model):
    sentences = sent_tokenize(text)
    if not sentences:
        return np.zeros(768)
    sentence_embeddings = model.encode(sentences)
    return np.mean(sentence_embeddings, axis=0)

def load_all_transcripts(file_paths):
    data = []
    for path in file_paths:
        with open(path, 'r') as f:
            data.extend(json.load(f))
    df = pd.DataFrame(data)
    
    # Extract identity metadata
    df['demographic'] = df['metadata'].apply(lambda x: x['persona'].get('demographic', 'Unmarked'))
    df['gender'] = df['metadata'].apply(lambda x: x['persona'].get('gender', 'Unmarked'))
    df['occupation'] = df['metadata'].apply(lambda x: x['persona'].get('occupation', 'Unmarked'))
    df['topic'] = df['metadata'].apply(lambda x: x.get('task_description', 'general_comment'))
    df['target_logprobs'] = df['metadata'].apply(lambda x: x.get('target_logprobs', []))
    df['scenario_id'] = df['metadata'].apply(lambda x: x.get('scenario_id', 'unknown'))
    
    return df

def main(args):
    from sentence_transformers import SentenceTransformer

    print("Loading data...")
    target_path = os.path.join(args.dir, 'target_simulations.json')
    control_path = os.path.join(args.dir, 'control_simulations.json')
    default_topic_path = os.path.join(args.dir, 'default_topics.json')

    df = load_all_transcripts([target_path, control_path, default_topic_path])
    
    # Initialize Evaluators
    alloc_eval = AllocationalEvaluator(
        model_path=args.judge_model,
        tensor_parallel_size=args.tensor_parallel_size
    )
    rep_eval = RepresentationalEvaluator()
    ie = IntersectionalEvaluator()
    masker = SemanticMasker()
    
    # Create intersectional IDs
    df['intersectional_id'] = df.apply(ie.create_intersectional_tuple, axis=1)
    
    # Apply Semantic Masking & Generate Embeddings for Steering
    print("Masking explicit identifiers and generating embeddings...")
    def extract_target_text(transcript):
        return " ".join([turn['content'] for turn in transcript if turn['speaker'] == 'Target'])
    
    df['target_text'] = df['transcript'].apply(extract_target_text)
    df['masked_text'] = df['target_text'].apply(masker.redact_explicit_identifiers)
    
    embedder = SentenceTransformer("all-mpnet-base-v2")
    df['embedding'] = df['masked_text'].apply(lambda x: get_document_embedding(x, embedder))

    final_results = []
    target_groups = [g for g in df['intersectional_id'].unique() if "Unmarked_Unmarked" not in g]
    
    print(f"Processing metrics for {len(target_groups)} intersectional groups...")

    for target_id in target_groups:
        # 1. Isolate the data for this specific intersectional group
        group_df = df[df['intersectional_id'] == target_id]
        
        # 2. Split into Implicit and Explicit datasets
        implicit_df = group_df[group_df['variant_type'] == 'implicit']
        explicit_df = group_df[group_df['variant_type'] == 'explicit']
        
        if implicit_df.empty or explicit_df.empty:
            print(f"Skipping {target_id}: Missing implicit or explicit data.")
            continue

        # --- ALLOCATIONAL METRICS ---
        implicit_gcr = np.mean([alloc_eval.calculate_gcr_llm_judge(t, md['task_description']) 
                                for t, md in zip(implicit_df['transcript'], implicit_df['metadata'])])
        explicit_gcr = np.mean([alloc_eval.calculate_gcr_llm_judge(t, md['task_description']) 
                                for t, md in zip(explicit_df['transcript'], explicit_df['metadata'])])
        
        # Calculate Delta GCR (Explicit - Implicit)
        d_gcr = alloc_eval.calculate_d_gcr(implicit_gcr, explicit_gcr)
        
        implicit_atc = np.mean([alloc_eval.calculate_atc(t) for t in implicit_df['transcript']])
        explicit_atc = np.mean([alloc_eval.calculate_atc(t) for t in explicit_df['transcript']])
        
        # --- REPRESENTATIONAL METRICS (CONFIDENCE) ---
        # Flatten the logprobs arrays for the whole implicit/explicit subset
        implicit_logprobs = [lp for sublist in implicit_df['target_logprobs'] for lp in sublist if lp]
        explicit_logprobs = [lp for sublist in explicit_df['target_logprobs'] for lp in sublist if lp]
        
        # Calculate Delta CCD (Explicit - Implicit)
        d_ccd = rep_eval.calculate_d_ccd(implicit_logprobs, explicit_logprobs)
        
        # --- SEMANTIC STEERING (CoMPosT INTEGRATION) ---
        # Dynamically define the counterfactual to extract the exact CoMPosT axis
        occupation = target_id.split('_')[-1]
        control_id = f"Unmarked_Unmarked_{occupation}"
        
        implicit_steerings = []
        explicit_steerings = []
        
        if control_id in df['intersectional_id'].values:
            # Get unique scenarios (topics) for this group, excluding the general_comment baseline
            unique_scenarios = group_df[group_df['topic'] != 'general_comment']['scenario_id'].unique()
            
            for scenario_id in unique_scenarios:
                try:
                    # --- IMPLICIT STEERING FOR THIS SCENARIO ---
                    # Compute axis using ONLY implicit variant data
                    implicit_axis, imp_topic_pole_sim, imp_persona_pole_sim = ie.get_fightin_words_poles(
                        df,
                        target_id,
                        control_id,
                        variant_type='implicit',
                        target_topic_id=scenario_id,
                        default_topic="general_comment"
                    )
                    
                    # Project implicit responses for this scenario onto the implicit axis
                    implicit_scenario_df = implicit_df[
                        (implicit_df['scenario_id'] == scenario_id) & (implicit_df['topic'] != 'general_comment')
                    ]
                    if not implicit_scenario_df.empty:
                        imp_target_embs = np.vstack(implicit_scenario_df['embedding'].values)
                        imp_steer_dict = rep_eval.calculate_semantic_steering(
                            implicit_target_embeddings=imp_target_embs,
                            explicit_target_embeddings=imp_target_embs,
                            axis_v=implicit_axis,
                            topic_pole_sim=imp_topic_pole_sim,
                            persona_pole_sim=imp_persona_pole_sim
                        )
                        implicit_steerings.append(imp_steer_dict['implicit_steering'])
                    
                    # --- EXPLICIT STEERING FOR THIS SCENARIO ---
                    # Compute axis using ONLY explicit variant data
                    explicit_axis, exp_topic_pole_sim, exp_persona_pole_sim = ie.get_fightin_words_poles(
                        df,
                        target_id,
                        control_id,
                        variant_type='explicit',
                        target_topic_id=scenario_id,
                        default_topic="general_comment"
                    )
                    
                    # Project explicit responses for this scenario onto the explicit axis
                    explicit_scenario_df = explicit_df[
                        (explicit_df['scenario_id'] == scenario_id) & (explicit_df['topic'] != 'general_comment')
                    ]
                    if not explicit_scenario_df.empty:
                        exp_target_embs = np.vstack(explicit_scenario_df['embedding'].values)
                        exp_steer_dict = rep_eval.calculate_semantic_steering(
                            implicit_target_embeddings=exp_target_embs,
                            explicit_target_embeddings=exp_target_embs,
                            axis_v=explicit_axis,
                            topic_pole_sim=exp_topic_pole_sim,
                            persona_pole_sim=exp_persona_pole_sim
                        )
                        explicit_steerings.append(exp_steer_dict['explicit_steering'])
                    
                except (ValueError, np.linalg.LinAlgError):
                    # Monroe log-odds failed or insufficient data for this scenario
                    continue
        
        # Average scores across all valid scenarios to get final steering metrics
        final_imp_steer = np.nanmean(implicit_steerings) if implicit_steerings else np.nan
        final_exp_steer = np.nanmean(explicit_steerings) if explicit_steerings else np.nan
        final_delta_steer = final_exp_steer - final_imp_steer if not (np.isnan(final_imp_steer) or np.isnan(final_exp_steer)) else np.nan
        
        steering_scores = {
            'implicit_steering': final_imp_steer,
            'explicit_steering': final_exp_steer,
            'delta_steering': final_delta_steer
        }

        # 3. Compile the group's results
        final_results.append({
            'group_label': target_id,
            'implicit_GCR': implicit_gcr,
            'explicit_GCR': explicit_gcr,
            'd_GCR': d_gcr,
            'implicit_ATC': implicit_atc,
            'explicit_ATC': explicit_atc,
            'd_CCD': d_ccd,
            'implicit_Steering': steering_scores['implicit_steering'],
            'explicit_Steering': steering_scores['explicit_steering'],
            'delta_Steering': steering_scores['delta_steering']
        })

    # 4. Save and Report
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    results_df = pd.DataFrame(final_results)
    results_df.to_csv(args.out, index=False)
    print("\nEvaluation Complete. Results saved.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate dynamic intersectional bias.")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing the transcript JSONs")
    parser.add_argument("--out", type=str, default="dynamic_bias_results.csv", help="Output CSV filename")
    parser.add_argument("--judge_model", type=str, required=True, help="HuggingFace ID or local path for the LLM judge")
    parser.add_argument("--tensor_parallel_size", type=int, default=2, help="Number of GPUs to use for vLLM")
    args = parser.parse_args()

    target_path = os.path.join(args.dir, 'target_simulations.json')
    control_path = os.path.join(args.dir, 'control_simulations.json')
    default_topic_path = os.path.join(args.dir, 'default_topics.json')
    
    print(f"Evaluating transcripts in: {args.dir}")
    main(args)