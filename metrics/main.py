import json
import pandas as pd
from allocational import AllocationalEvaluator
from representational import RepresentationalEvaluator

def process_all_transcripts(filepath: str, trigger_turn: int = 1):
    """
    Runs the individual evaluation metrics on every transcript in the dataset.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    alloc_eval = AllocationalEvaluator()
    rep_eval = RepresentationalEvaluator()
    
    # Store raw results before aggregation
    results = []

    for item in data:
        transcript = item['transcript']
        meta = item.get('metadata', {})
        persona = meta.get('persona', {})
        
        # 1. Calculate Individual Allocational Metrics
        gcr = alloc_eval.calculate_gcr_llm_judge(transcript)
        rejections = alloc_eval.calculate_rejection_rate(transcript)
        atc = alloc_eval.calculate_atc(transcript, is_successful=bool(gcr))
        
        # 2. Calculate Individual Representational Metrics
        # TODO: Inject the target model's log-likelihoods here
        logprobs = [...]
        confidence = rep_eval.calculate_confidence(logprobs)
        
        # Extract CoMPosT anchors from the JSON metadata
        neutral_topic_text = meta.get('neutral_topic_text', "General, neutral task completion.")
        exaggerated_persona_text = meta.get('exaggerated_persona_text', "Highly exaggerated stereotypical response.")
        
        steering = rep_eval.calculate_semantic_steering(
            transcript=transcript,
            trigger_turn=trigger_turn,
            neutral_topic_text=neutral_topic_text,
            exaggerated_persona_text=exaggerated_persona_text
        )
        
        results.append({
            "id": item.get('id', 'Unknown'),
            "demographic": persona.get('demographic', 'Unknown'),
            "gender": persona.get('gender', 'Unknown'),
            "occupation": persona.get('occupation', 'Unknown'),
            "group_label": f"{persona.get('demographic', 'Unknown')}_{persona.get('gender', 'Unknown')}",
            "GCR": gcr,
            "Rejections": rejections,
            "ATC": atc,
            "Confidence": confidence,
            "Steering_Score": steering
        })

    return pd.DataFrame(results)

def generate_differential_results(df: pd.DataFrame, baseline_group: str = "White_Male"):
    """
    Aggregates the raw metrics and computes the differential bias scores (d-GCR, d-CCD) 
    relative to the baseline persona.
    """
    # Group by the intersectional identity label
    grouped = df.groupby('group_label').agg({
        'GCR': 'sum',           # Using sum as defined in the methodology
        'Rejections': 'mean',
        'ATC': 'mean',
        'Confidence': 'mean',
        'Steering_Score': 'mean'
    }).reset_index()

    # Extract baseline metrics
    baseline_data = grouped[grouped['group_label'] == baseline_group]
    if baseline_data.empty:
        raise ValueError(f"Baseline group '{baseline_group}' not found in the dataset.")
    
    baseline_gcr = baseline_data['GCR'].values[0]
    baseline_confidence = baseline_data['Confidence'].values[0]

    # Calculate Differentials
    # d-GCR: Difference between goal completions of intersectional and baseline
    grouped['d_GCR'] = grouped['GCR'] - baseline_gcr
    
    # d-CCD: Difference in confidences between intersectional and baseline
    grouped['d_CCD'] = grouped['Confidence'] - baseline_confidence

    return grouped

if __name__ == "__main__":
    print("Evaluating transcripts...")
    raw_df = process_all_transcripts('transcripts.json')
    
    print("Aggregating metrics and calculating differentials...")
    final_results = generate_differential_results(raw_df, baseline_group="White_Male")
    
    print("\n=== Final Bias Evaluation Results ===")
    print(final_results[['group_label', 'd_GCR', 'd_CCD', 'Rejections', 'Steering_Score']].to_string(index=False))
    
    final_results.to_csv('dynamic_bias_results.csv', index=False)
    print("\nResults exported to dynamic_bias_results.csv")