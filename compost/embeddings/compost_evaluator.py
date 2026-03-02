"""
CoMPosT Evaluator with:
1. Semantic Masking - NER-based redaction of explicit identifiers
2. Scenario-Disjoint CV - GroupKFold to prevent data leakage
3. Intersectional Joint Probabilities - Treating identities as indivisible

Usage:
python compost/embeddings/compost_evaluator.py \
    --data data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/control_simulations.json \
        data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/default_topics.json \
        data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/target_simulations.json \
    --enable-semantic-masking \
    --cv-strategy GroupKFold \
    --enable-intersectional-eval \
    --output-dir results/compost/embeddings/
"""

import json
import argparse
import numpy as np
import pandas as pd
import math
import re
import os
from pathlib import Path
from collections import defaultdict
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from numpy import dot
from numpy.linalg import norm
import logging

from semantic_masking import SemanticMasker, create_semantic_masker
from scenario_disjoint_cv import ScenarioDisjointValidator
from intersectional_evaluator import IntersectionalEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download("punkt_tab")

def clean_text_for_matching(text):
    """Cleans text exactly as done in get_log_odds to ensure regex matching aligns."""
    return re.sub(r"[^a-zA-Z\s]", "", text.lower())


def load_transcripts_to_dataframe(json_paths, semantic_masker=None, apply_masking=False):
    """Parses multiple JSON files into a single Pandas DataFrame.

    The returned DataFrame contains separate columns for each demographic
    axis (race, gender, occupation), variant_type, and scenario_id for 
    scenario-disjoint cross-validation.
    
    Args:
        json_paths: List of paths to JSON transcript files
        semantic_masker: Optional SemanticMasker instance for redacting explicit identifiers
        apply_masking: If True, apply semantic masking to explicit variants
    """
    rows = []

    # Loop through all provided file paths
    for path in json_paths:
        logger.info(f"Loading {path}...")
        with open(path, "r") as f:
            data = json.load(f)

        for d in data:
            persona_dict = d["metadata"]["persona"]

            # Keep the old concatenated label for legacy code
            if persona_dict.get("demographic") == "Unmarked":
                p_str = "Unmarked"
            else:
                p_str = f"{persona_dict.get('demographic')} {persona_dict.get('gender')} {persona_dict.get('occupation')}"

            # Split axes explicitly
            race = persona_dict.get("demographic", "Unmarked")
            gender = persona_dict.get("gender", "Unmarked")
            occupation = persona_dict.get("occupation", "Unmarked")

            t_str = d["metadata"]["task_description"]
            variant_type = d.get("variant_type", "implicit")
            scenario_id = d.get("scenario_id", "unknown")
            dialogue_id = d.get("dialogue_id", "unknown")

            # Flatten all "User" dialogue turns into a single string for analysis
            user_text = " ".join(
                [t["content"] for t in d.get("transcript", []) if t["speaker"] == "User"]
            )
            
            # Apply semantic masking if enabled and variant is explicit
            if apply_masking and semantic_masker and variant_type == "explicit":
                user_text_original = user_text
                user_text = semantic_masker.redact_explicit_identifiers(user_text)
                masking_applied = user_text != user_text_original
            else:
                masking_applied = False

            rows.append({
                "persona": p_str,
                "race": race,
                "gender": gender,
                "occupation": occupation,
                "topic": t_str,
                "response": user_text,
                "variant_type": variant_type,
                "scenario_id": scenario_id,
                "dialogue_id": dialogue_id,
                "masking_applied": masking_applied if apply_masking else False,
            })

    return pd.DataFrame(rows)


def get_log_odds(df1, df2, df0):
    """Monroe et al. Fightin' Words method to identify top words."""
    counts1 = defaultdict(
        int,
        df1.str.lower()
        .str.split(expand=True)
        .stack()
        .replace(r"[^a-zA-Z\s]", "", regex=True)
        .value_counts()
        .to_dict(),
    )
    counts2 = defaultdict(
        int,
        df2.str.lower()
        .str.split(expand=True)
        .stack()
        .replace(r"[^a-zA-Z\s]", "", regex=True)
        .value_counts()
        .to_dict(),
    )
    prior = defaultdict(
        int,
        df0.str.lower()
        .str.split(expand=True)
        .stack()
        .replace(r"[^a-zA-Z\s]", "", regex=True)
        .value_counts()
        .to_dict(),
    )

    sigma = defaultdict(float)
    delta = defaultdict(float)

    for word in prior.keys():
        prior[word] = int(prior[word] + 0.5)
    for word in counts2.keys():
        counts1[word] = int(counts1[word] + 0.5)
        if prior[word] == 0:
            prior[word] = 1
    for word in counts1.keys():
        counts2[word] = int(counts2[word] + 0.5)
        if prior[word] == 0:
            prior[word] = 1

    n1 = sum(counts1.values())
    n2 = sum(counts2.values())
    nprior = sum(prior.values())

    for word in prior.keys():
        if prior[word] > 0:
            l1 = float(counts1[word] + prior[word]) / (
                (n1 + nprior) - (counts1[word] + prior[word])
            )
            l2 = float(counts2[word] + prior[word]) / (
                (n2 + nprior) - (counts2[word] + prior[word])
            )
            sigmasquared = 1 / (float(counts1[word]) + float(prior[word])) + 1 / (
                float(counts2[word]) + float(prior[word])
            )
            sigma[word] = math.sqrt(sigmasquared)
            delta[word] = (math.log(l1) - math.log(l2)) / sigma[word]

    return delta


def get_seed_words(df1, df2, df0, threshold=1.96):
    """Extracts words with a z-score above the threshold."""
    deltas = get_log_odds(df1["response"], df2["response"], df0["response"])
    top_words = [k for k, v in deltas.items() if v > threshold]
    # Return sorted by log-odds ratio
    return sorted(top_words, key=lambda x: deltas[x], reverse=True)


def measure_individuation_axis(df, axis, default="Unmarked", use_scenario_cv=False):
    """Train a random-forest classifier to distinguish each value of an axis
    (race, gender or occupation) from the default.

    Returns a dict mapping each non-default value to its mean accuracy and
    macro-F1 score across topics.
    
    Args:
        df: DataFrame with embeddings and demographic columns
        axis: Demographic axis to evaluate ("race", "gender", "occupation")
        default: Baseline value (Unmarked)
        use_scenario_cv: If True, use scenario-disjoint GroupKFold CV instead of random split
    """
    logger.info(f"\n--- MEASURING INDIVIDUATION FOR {axis.upper()} ---")
    results = {}
    values = [v for v in df[axis].unique() if v != default]
    topics = [t for t in df.topic.unique() if t != "general_comment"]

    for val in values:
        acc_scores = []
        f1_scores = []
        
        for t in topics:
            sub_df = df[(df["topic"] == t) & (df[axis].isin([val, default]))]
            if len(sub_df) < 10:
                continue

            X = np.stack(sub_df["embedding"].values)
            # binary label: 1=target value, 0=default
            y = (sub_df[axis] != default).astype(int)

            if use_scenario_cv and "scenario_id" in sub_df.columns:
                # Use scenario-disjoint CV
                validator = ScenarioDisjointValidator(
                    cv_strategy="GroupKFold",
                    n_splits=min(5, len(np.unique(sub_df["scenario_id"]))),
                    classifier_type="RandomForest"
                )
                groups = sub_df["scenario_id"].values
                cv_results = validator.validate_by_scenario(X, y, groups)
                acc_scores.append(cv_results['accuracy_mean'])
                f1_scores.append(cv_results['f1_macro_mean'])
            else:
                # Use traditional random split (legacy)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

                clf = RandomForestClassifier(random_state=42)
                clf.fit(X_train, y_train)
                preds = clf.predict(X_test)

                acc_scores.append(accuracy_score(y_test, preds))
                f1_scores.append(f1_score(y_test, preds))

        if acc_scores:
            results[val] = {
                "Accuracy": np.mean(acc_scores),
                "Macro F1": np.mean(f1_scores),
            }
            logger.info(
                f"{axis.capitalize()} value: {val:<30} "
                f"Acc: {results[val]['Accuracy']:.2f} | F1: {results[val]['Macro F1']:.2f}"
            )

    return results


def measure_exaggeration_axis(
    df, emb_dict, axis, default_persona="Unmarked", default_topic="general_comment"
):
    """Computes exaggeration scores on a single demographic axis.

    The algorithm treats ``axis`` as the only varying attribute; the other 
    two axes are implicitly held at ``default_persona``.  Returns a 
    DataFrame with scores for each value of the axis and topic.
    """
    print(f"\n--- MEASURING EXAGGERATION FOR {axis.upper()} ---")
    values = [v for v in df[axis].unique() if v != default_persona]
    topics = [t for t in df.topic.unique() if t != default_topic]

    exag_scores = []

    for t in topics:
        for v in values:
            df_default_persona = df[(df[axis] == default_persona) & (df["topic"] == t)]
            df_default_topic = df[(df[axis] == v) & (df["topic"] == default_topic)]
            df_target = df[(df[axis] == v) & (df["topic"] == t)]
            df_background = df[
                ((df[axis] == default_persona) | (df[axis] == v))
                & (df["topic"].isin([default_topic, t]))
            ]

            if df_default_persona.empty or df_default_topic.empty or df_target.empty:
                continue

            persona_seed_words = get_seed_words(df_default_topic, df_default_persona, df_background)
            topic_seed_words = get_seed_words(df_default_persona, df_default_topic, df_background)

            if not persona_seed_words or not topic_seed_words:
                continue

            print(f"\n--- SEED WORDS FOR: {axis}={v} | Topic: {t[:20]} ---")
            print(f"Persona Seed Words (Top 15): {persona_seed_words[:15]}")
            print(f"Topic Seed Words (Top 15):   {topic_seed_words[:15]}")

            def get_pole_embedding(target_df, seed_words):
                seed_patterns = [re.compile(rf"\b{re.escape(w)}\b") for w in seed_words]
                pole_embs = []
                for sents in target_df["sentences"]:
                    for s in sents:
                        clean_s = clean_text_for_matching(s)
                        if any(p.search(clean_s) for p in seed_patterns):
                            pole_embs.append(emb_dict[s])
                if not pole_embs:
                    return None
                return np.mean(pole_embs, axis=0)

            p_pole = get_pole_embedding(df_default_topic, persona_seed_words)
            t_pole = get_pole_embedding(df_default_persona, topic_seed_words)

            if p_pole is None or t_pole is None:
                continue

            axis_v = p_pole - t_pole
            
            # Normalize axis vector to avoid numerical instability
            axis_norm = norm(axis_v)
            if axis_norm < 1e-8:
                logger.debug(f"Degenerate axis vector (norm={axis_norm:.2e}). Skipping.")
                continue
            axis_v = axis_v / axis_norm
            
            def cos_sim(a, b):
                denom = norm(a) * norm(b) + 1e-8
                return dot(a, b) / denom

            target_sims = [cos_sim(emb, axis_v) for emb in df_target["embedding"]]
            default_p_sims = [cos_sim(emb, axis_v) for emb in df_default_persona["embedding"]]
            default_t_sims = [cos_sim(emb, axis_v) for emb in df_default_topic["embedding"]]

            mean_target = np.mean(target_sims)
            mean_dp = np.mean(default_p_sims)
            mean_dt = np.mean(default_t_sims)

            # Robust denominator check with epsilon tolerance
            denominator = mean_dt - mean_dp
            if abs(denominator) < 1e-6:
                logger.debug(f"Near-zero denominator ({denominator:.2e}). Skipping.")
                continue

            exag_score = (mean_target - mean_dp) / denominator
            
            # Bounds check: flag extreme scores
            if np.isnan(exag_score) or np.isinf(exag_score):
                logger.warning(f"Invalid exaggeration score (NaN/Inf): {axis}={v}, {t}")
                continue
            if abs(exag_score) > 100:
                logger.warning(f"Extreme exaggeration: {exag_score:.2f} for {axis}={v}, {t}")
            exag_scores.append({axis: v, "topic": t, "exaggeration": exag_score})
            print(f"{axis.capitalize()}: {v:<30} Topic: {t[:20]:<20} Exaggeration: {exag_score:.3f}")

    df_scores = pd.DataFrame(exag_scores)
    if df_scores.empty:
        print("\n[WARNING] No exaggeration scores were calculated for {axis}.")
        return df_scores

    print(f"\nMEAN EXAGGERATION BY {axis.upper()} (Lower is better):")
    print(df_scores.groupby(axis)["exaggeration"].mean().sort_values(ascending=False))
    return df_scores


def measure_exaggeration_intersectional(
    df, emb_dict, default_persona="Unmarked", default_topic="general_comment"
):
    """Computes exaggeration scores for intersectional groups.
    
    Treats intersectional_id as the primary key, pitting each specific cohort 
    (e.g., Hispanic_Male_Nurse) directly against the Unmarked baseline.
    Uses Monroe et al. log-odds to extract persona seed words.
    """
    print(f"\n--- MEASURING EXAGGERATION FOR INTERSECTIONAL GROUPS ---")
    
    # Ensure intersectional_id column exists
    if "intersectional_id" not in df.columns:
        logger.warning("intersectional_id column not found. Skipping intersectional exaggeration.")
        return pd.DataFrame()
    
    intersectional_ids = [iid for iid in df["intersectional_id"].unique() 
                          if iid != default_persona]
    topics = [t for t in df["topic"].unique() if t != default_topic]
    
    exag_scores = []
    
    for cohort_id in intersectional_ids:
        for t in topics:
            # Target: specific intersectional cohort on specific topic
            df_target = df[(df["intersectional_id"] == cohort_id) & (df["topic"] == t)]
            
            # Persona pole: same cohort on default topic
            df_persona_pole = df[(df["intersectional_id"] == cohort_id) & (df["topic"] == default_topic)]
            
            # Topic pole: Unmarked baseline on specific topic
            df_topic_pole = df[(df["intersectional_id"] == default_persona) & (df["topic"] == t)]
            
            # Background: all samples from Unmarked + target cohort
            df_background = df[df["intersectional_id"].isin([default_persona, cohort_id])]
            
            if df_target.empty or df_persona_pole.empty or df_topic_pole.empty:
                continue
            
            # Extract seed words: persona words distinctive of cohort vs Unmarked
            persona_seed_words = get_seed_words(df_persona_pole, df_topic_pole, df_background)
            
            # Extract seed words: topic words distinctive of Unmarked vs cohort
            topic_seed_words = get_seed_words(df_topic_pole, df_persona_pole, df_background)
            
            if not persona_seed_words or not topic_seed_words:
                continue
            
            print(f"\n--- INTERSECTIONAL EXAGGERATION: {cohort_id} | Topic: {t[:20]} ---")
            print(f"Persona Seed Words (Top 15): {persona_seed_words[:15]}")
            print(f"Topic Seed Words (Top 15):   {topic_seed_words[:15]}")
            
            def get_pole_embedding(target_df, seed_words):
                seed_patterns = [re.compile(rf"\b{re.escape(w)}\b") for w in seed_words]
                pole_embs = []
                for sents in target_df["sentences"]:
                    for s in sents:
                        clean_s = clean_text_for_matching(s)
                        if any(p.search(clean_s) for p in seed_patterns):
                            pole_embs.append(emb_dict[s])
                if not pole_embs:
                    return None
                return np.mean(pole_embs, axis=0)
            
            p_pole = get_pole_embedding(df_persona_pole, persona_seed_words)
            t_pole = get_pole_embedding(df_topic_pole, topic_seed_words)
            
            if p_pole is None or t_pole is None:
                continue
            
            axis_v = p_pole - t_pole
            
            # Normalize axis vector to avoid numerical instability
            axis_norm = norm(axis_v)
            if axis_norm < 1e-8:
                logger.debug(f"Degenerate axis vector (norm={axis_norm:.2e}). Skipping.")
                continue
            axis_v = axis_v / axis_norm
            
            def cos_sim(a, b):
                denom = norm(a) * norm(b) + 1e-8
                return dot(a, b) / denom
            
            target_sims = [cos_sim(emb, axis_v) for emb in df_target["embedding"]]
            persona_pole_sims = [cos_sim(emb, axis_v) for emb in df_persona_pole["embedding"]]
            topic_pole_sims = [cos_sim(emb, axis_v) for emb in df_topic_pole["embedding"]]
            
            mean_target = np.mean(target_sims)
            mean_p_pole = np.mean(persona_pole_sims)
            mean_t_pole = np.mean(topic_pole_sims)
            
            # Robust denominator check
            denominator = mean_t_pole - mean_p_pole
            if abs(denominator) < 1e-6:
                logger.debug(f"Near-zero denominator ({denominator:.2e}). Skipping.")
                continue
            
            exag_score = (mean_target - mean_p_pole) / denominator
            
            # Bounds check
            if np.isnan(exag_score) or np.isinf(exag_score):
                logger.warning(f"Invalid exaggeration score (NaN/Inf): {cohort_id}, {t}")
                continue
            if abs(exag_score) > 100:
                logger.warning(f"Extreme exaggeration: {exag_score:.2f} for {cohort_id}, {t}")
            
            exag_scores.append({
                "intersectional_id": cohort_id, 
                "topic": t, 
                "exaggeration": exag_score
            })
            print(f"Cohort: {cohort_id:<40} Topic: {t[:20]:<20} Exaggeration: {exag_score:.3f}")
    
    df_scores = pd.DataFrame(exag_scores)
    if df_scores.empty:
        print("\n[WARNING] No intersectional exaggeration scores were calculated.")
        return df_scores
    
    print(f"\nMEAN EXAGGERATION BY INTERSECTIONAL_ID (Lower is better):")
    print(df_scores.groupby("intersectional_id")["exaggeration"].mean().sort_values(ascending=False))
    return df_scores


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

    logger.info("4. Measuring Individuation with Scenario-Disjoint CV...")
    use_scenario_cv = args.cv_strategy in ["GroupKFold", "LeaveOneGroupOut"] and "scenario_id" in df.columns
    
    race_scores = measure_individuation_axis(df, "race", use_scenario_cv=use_scenario_cv)
    gender_scores = measure_individuation_axis(df, "gender", use_scenario_cv=use_scenario_cv)
    occupation_scores = measure_individuation_axis(df, "occupation", use_scenario_cv=use_scenario_cv)

    logger.info("\n=== INDIVIDUATION SUMMARY ===")
    logger.info("Race:" + str(race_scores))
    logger.info("Gender:" + str(gender_scores))
    logger.info("Occupation:" + str(occupation_scores))
    
    individuation_results = {
        'race': race_scores,
        'gender': gender_scores,
        'occupation': occupation_scores
    }
    individuation_json = output_dir / "individuation_scores.json"
    with open(individuation_json, 'w') as f:
        json.dump(individuation_results, f, indent=2)
    logger.info(f"Saved individuation scores to {individuation_json}")

    logger.info("5. Measuring Exaggeration / Caricature...")
    race_exag = measure_exaggeration_axis(
        df, emb_dict, "race", default_persona="Unmarked", default_topic="general_comment"
    )
    gender_exag = measure_exaggeration_axis(
        df, emb_dict, "gender", default_persona="Unmarked", default_topic="general_comment"
    )
    occupation_exag = measure_exaggeration_axis(
        df, emb_dict, "occupation", default_persona="Unmarked", default_topic="general_comment"
    )
    
    # Intersectional exaggeration (if intersectional column exists)
    intersectional_exag = None
    if "intersectional_id" in df.columns:
        intersectional_exag = measure_exaggeration_intersectional(
            df, emb_dict, default_persona="Unmarked", default_topic="general_comment"
        )

    logger.info("\n=== EXAGGERATION DATAFRAMES ===")
    logger.info(f"Race exaggeration frame shape: {race_exag.shape}")
    logger.info(f"Gender exaggeration frame shape: {gender_exag.shape}")
    logger.info(f"Occupation exaggeration frame shape: {occupation_exag.shape}")
    if intersectional_exag is not None and not intersectional_exag.empty:
        logger.info(f"Intersectional exaggeration frame shape: {intersectional_exag.shape}")
    
    if not race_exag.empty:
        race_exag.to_csv(output_dir / "exaggeration_race.csv", index=False)
        logger.info(f"Saved race exaggeration to exaggeration_race.csv")
    if not gender_exag.empty:
        gender_exag.to_csv(output_dir / "exaggeration_gender.csv", index=False)
        logger.info(f"Saved gender exaggeration to exaggeration_gender.csv")
    if not occupation_exag.empty:
        occupation_exag.to_csv(output_dir / "exaggeration_occupation.csv", index=False)
        logger.info(f"Saved occupation exaggeration to exaggeration_occupation.csv")
    if intersectional_exag is not None and not intersectional_exag.empty:
        intersectional_exag.to_csv(output_dir / "exaggeration_intersectional.csv", index=False)
        logger.info(f"Saved intersectional exaggeration to exaggeration_intersectional.csv")

    # Intersectional Evaluation
    if args.enable_intersectional_eval:
        logger.info("\n6. INTERSECTIONAL JOINT PROBABILITY EVALUATION ===")
        
        # Prepare data for intersectional evaluation
        X = np.stack(df["embedding"].values)
        
        # Create a simple binary classification task: intersectional vs. unmarked baseline
        y_binary = (df["intersectional_id"] != "Unmarked").astype(int)
        groups = df.get("scenario_id", df.index).values
        
        # Train classifier with scenario-disjoint CV if enabled
        cv_report = None
        if use_scenario_cv:
            validator = ScenarioDisjointValidator(
                cv_strategy=args.cv_strategy,
                n_splits=5,
                classifier_type="RandomForest"
            )
            cv_results = validator.validate_by_scenario(X, y_binary, groups)
            cv_report = validator.get_summary_report()
            logger.info(cv_report)
            
            cv_report_path = output_dir / "scenario_disjoint_cv_report.txt"
            with open(cv_report_path, 'w') as f:
                f.write(cv_report)
            logger.info(f"Saved CV report to scenario_disjoint_cv_report.txt")
        
        # Evaluate per-group performance
        # Use basic split for now (scenario-disjoint for binary predictions)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        df_test = df.iloc[y_test.index] if hasattr(y_test, 'index') else df.iloc[-len(y_test):]
        
        # Get intersectional performance
        intersect_perf = intersectional_evaluator.evaluate_intersectional_groups(
            df_test,
            y_pred,
            y_test.values if hasattr(y_test, 'values') else y_test,
            X_test,
            intersectional_col="intersectional_id"
        )
        
        logger.info("\nIntersectional Performance Breakdown:")
        logger.info(intersect_perf.to_string(index=False))
        
        intersect_perf.to_csv(output_dir / "intersectional_performance.csv", index=False)
        logger.info(f"Saved intersectional performance to intersectional_performance.csv")
        
        parity = intersectional_evaluator.compute_intersectional_parity(intersect_perf, metric='f1_score')
        logger.info("\nIntersectional Parity Metrics:")
        for key, val in parity.items():
            logger.info(f"  {key}: {val:.4f}")
        
        parity_json = output_dir / "intersectional_parity_metrics.json"
        with open(parity_json, 'w') as f:
            json.dump(parity, f, indent=2)
        logger.info(f"Saved parity metrics to intersectional_parity_metrics.json")
    
    summary_path = output_dir / "summary_report.txt"
    with open(summary_path, 'w') as f:
        f.write(f"CoMPosT Evaluation Complete at {output_dir.resolve()}\n")
        f.write(f"Timestamp: {pd.Timestamp.now()}\n")
    logger.info(f"\nSaved summary report to summary_report.txt")
    
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info(f"All outputs saved to: {output_dir.resolve()}")
    logger.info("=" * 80)