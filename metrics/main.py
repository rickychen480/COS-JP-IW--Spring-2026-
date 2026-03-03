import argparse
import json
import nltk
import numpy as np
import pandas as pd
import os
from sentence_transformers import SentenceTransformer

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

from allocational import AllocationalEvaluator
from representational import RepresentationalEvaluator
from compost.embeddings.intersectional_evaluator import IntersectionalEvaluator
from compost.embeddings.semantic_masking import SemanticMasker


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
    df = pd.json_normalize(data)
    
    # Extract identity metadata
    df['demographic'] = df['metadata'].apply(lambda x: x['persona'].get('demographic', 'Unmarked'))
    df['gender'] = df['metadata'].apply(lambda x: x['persona'].get('gender', 'Unmarked'))
    df['occupation'] = df['metadata'].apply(lambda x: x['persona'].get('occupation', 'Unmarked'))
    df['topic'] = df['metadata'].apply(lambda x: x.get('task_description', 'general_comment'))
    df['target_logprobs'] = df['metadata'].apply(lambda x: x.get('target_logprobs',))
    
    return df

def main(args):
    print("Loading data...")
    target_path = os.path.join(args.dir, 'target_simulations.json')
    control_path = os.path.join(args.dir, 'control_simulations.json')
    default_topic_path = os.path.join(args.dir, 'default_topics.json')

    df = load_all_transcripts([target_path, control_path, default_topic_path])
    
    # Initialize Evaluators
    alloc_eval = AllocationalEvaluator()
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
        
        steering_scores = {'implicit_steering': np.nan, 'explicit_steering': np.nan, 'delta_steering': np.nan}
        
        if control_id in df['intersectional_id'].values:
            # Pseudo-code: Use IntersectionalEvaluator to get the statistical poles 
            # by comparing the target_id vs the control_id on the 'general_comment' topic
            # p_pole, t_pole = ie.get_fightin_words_poles(df, target_id, control_id)
            # axis_v = p_pole - t_pole
            
            # Placeholders for the mathematical outputs of the CoMPosT log-odds extraction
            axis_v = np.zeros(768) 
            topic_pole_sim = 0.5 
            persona_pole_sim = 0.8
            
            implicit_embeddings = np.vstack(implicit_df['embedding'].values)
            explicit_embeddings = np.vstack(explicit_df['embedding'].values)
            
            steering_scores = rep_eval.calculate_semantic_steering(
                implicit_target_embeddings=implicit_embeddings,
                explicit_target_embeddings=explicit_embeddings,
                axis_v=axis_v,
                topic_pole_sim=topic_pole_sim,
                persona_pole_sim=persona_pole_sim
            )

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
    final_results.to_csv(args.out, index=False)
    print(f"\nResults exported to {args.out}")

    results_df = pd.DataFrame(final_results)
    results_df.to_csv(args.out, index=False)
    print("\nEvaluation Complete. Results saved.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate dynamic intersectional bias.")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing the transcript JSONs")
    parser.add_argument("--out", type=str, default="dynamic_bias_results.csv", help="Output CSV filename")
    args = parser.parse_args()

    target_path = os.path.join(args.dir, 'target_simulations.json')
    control_path = os.path.join(args.dir, 'control_simulations.json')
    default_topic_path = os.path.join(args.dir, 'default_topics.json')
    
    print(f"Evaluating transcripts in: {args.dir}")
    main(args)