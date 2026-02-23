import argparse
import json
import pandas as pd
import os
from allocational import AllocationalEvaluator
from representational import RepresentationalEvaluator

def load_anchor_dictionaries(control_path: str, default_topic_path: str):
    """
    Builds lookup dictionaries for the CoMPosT semantic axes using the transcript files.
    """
    with open(control_path, 'r') as f:
        control_data = json.load(f)
    with open(default_topic_path, 'r') as f:
        default_data = json.load(f)

    # Map by Task Description -> Target's concatenated text
    control_dict = {}
    for d in control_data:
        task_desc = d['metadata']['task_description']
        target_text = " ".join([t['content'] for t in d['transcript'] if t['speaker'] == 'Target'])
        control_dict[task_desc] = target_text

    # Map by (Demographic, Gender, Occupation) -> Target's concatenated text
    persona_dict = {}
    for d in default_data:
        p = d['metadata']['persona']
        persona_key = (p.get('demographic', 'Unknown'), p.get('gender', 'Unknown'), p.get('occupation', 'Unknown'))
        target_text = " ".join([t['content'] for t in d['transcript'] if t['speaker'] == 'Target'])
        persona_dict[persona_key] = target_text

    return control_dict, persona_dict


def process_all_transcripts(target_path: str, control_path: str, default_topic_path: str, trigger_turn: int = 0):
    """
    Runs the evaluation metrics using the CoMPosT anchors mapped from the dataset.
    """
    with open(target_path, 'r') as f:
        target_data = json.load(f)
    with open(control_path, 'r') as f:
        control_data = json.load(f)

    control_dict, persona_dict = load_anchor_dictionaries(control_path, default_topic_path)

    alloc_eval = AllocationalEvaluator()
    rep_eval = RepresentationalEvaluator()
    results = []

    # Process TARGET Data (Intersectional Groups)
    for item in target_data:
        transcript = item.get('transcript', [])
        meta = item.get('metadata', {})
        persona = meta.get('persona', {})
        task_desc = meta.get('task_description', '')
        
        # 1. Allocational
        gcr = alloc_eval.calculate_gcr_llm_judge(transcript)
        rejections = alloc_eval.calculate_rejection_rate(transcript)
        atc = alloc_eval.calculate_atc(transcript, is_successful=bool(gcr))
        
        # 2. Representational (d-CCD)
        logprobs = meta.get('target_logprobs', [])
        confidence = rep_eval.calculate_confidence(logprobs)
        
        # 3. Semantic Steering (CoMPosT)
        persona_key = (persona.get('demographic', 'Unknown'), persona.get('gender', 'Unknown'), persona.get('occupation', 'Unknown'))
        neutral_topic_text = control_dict.get(task_desc, "Fallback neutral text.")
        exaggerated_persona_text = persona_dict.get(persona_key, "Fallback persona text.")
        
        steering = rep_eval.calculate_semantic_steering(
            transcript=transcript,
            trigger_turn=trigger_turn,
            neutral_topic_text=neutral_topic_text,
            exaggerated_persona_text=exaggerated_persona_text
        )
        
        results.append({
            "id": item.get('dialogue_id', 'Unknown'),
            "variant_type": item.get('variant_type', 'Unknown'),
            "demographic": persona.get('demographic', 'Unknown'),
            "gender": persona.get('gender', 'Unknown'),
            "occupation": persona.get('occupation', 'Unknown'),
            "group_label": f"{persona.get('demographic', 'Unknown')}_{persona.get('gender', 'Unknown')}_{persona.get('occupation', 'Unknown')}",
            "GCR": gcr,
            "Rejections": rejections,
            "ATC": atc,
            "Confidence": confidence,
            "Steering_Score": steering
        })

    # Process CONTROL Data (Baselines)
    for item in control_data:
        transcript = item.get('transcript', [])
        meta = item.get('metadata', {})
        persona = meta.get('persona', {})
        
        gcr = alloc_eval.calculate_gcr_llm_judge(transcript)
        rejections = alloc_eval.calculate_rejection_rate(transcript)
        atc = alloc_eval.calculate_atc(transcript, is_successful=bool(gcr))
        
        logprobs = meta.get('target_logprobs', [])
        confidence = rep_eval.calculate_confidence(logprobs)
        
        # We don't calculate steering for the baseline control, so set to 0.0
        steering = 0.0
        
        results.append({
            "id": item.get('dialogue_id', 'Unknown'),
            "variant_type": item.get('variant_type', 'control'),
            "demographic": persona.get('demographic', 'Unmarked'),
            "gender": persona.get('gender', 'Unmarked'),
            "occupation": persona.get('occupation', 'Unmarked'),
            "group_label": "Unmarked_Unmarked_Unmarked",
            "GCR": gcr,
            "Rejections": rejections,
            "ATC": atc,
            "Confidence": confidence,
            "Steering_Score": steering
        })

    return pd.DataFrame(results)

def generate_differential_results(df: pd.DataFrame, baseline_group: str = "Unmarked_Unmarked_Unmarked"):
    """
    Aggregates the raw metrics and computes the differential bias scores (d-GCR, d-CCD)
    relative to the baseline persona, split by variant type.
    """
    grouped = df.groupby(['variant_type', 'group_label']).agg({
        'GCR': 'mean',           
        'Rejections': 'mean',
        'ATC': 'mean',
        'Confidence': 'mean',
        'Steering_Score': 'mean'
    }).reset_index()

    # Extract baseline metrics specifically from the 'control' variant
    baseline_data = grouped[(grouped['group_label'] == baseline_group) & (grouped['variant_type'] == 'control')]
    if baseline_data.empty:
        raise ValueError(f"Baseline group '{baseline_group}' not found in the dataset.")
    
    baseline_gcr = baseline_data['GCR'].values[0]
    baseline_confidence = baseline_data['Confidence'].values[0]

    # Calculate Differentials
    grouped['d_GCR'] = grouped['GCR'] - baseline_gcr
    grouped['d_CCD'] = grouped['Confidence'] - baseline_confidence

    # Filter out the control row from the final differential output (since it would just be all 0s)
    differential_results = grouped[grouped['variant_type'] != 'control'].copy()

    return differential_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate dynamic intersectional bias.")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing the transcript JSONs")
    parser.add_argument("--out", type=str, default="dynamic_bias_results.csv", help="Output CSV filename")
    args = parser.parse_args()

    target_path = os.path.join(args.dir, 'target_simulations.json')
    control_path = os.path.join(args.dir, 'control_simulations.json')
    default_topic_path = os.path.join(args.dir, 'default_topics.json')
    
    print(f"Evaluating transcripts in: {args.dir}")
    raw_df = process_all_transcripts(target_path, control_path, default_topic_path)
    
    print("Aggregating metrics and calculating differentials...")
    final_results = generate_differential_results(raw_df)
    
    print("\n=== Final Bias Evaluation Results ===")
    print(final_results[['variant_type', 'group_label', 'd_GCR', 'd_CCD', 'Rejections', 'Steering_Score']].to_string(index=False))
    
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    final_results.to_csv(args.out, index=False)
    print(f"\nResults exported to {args.out}")