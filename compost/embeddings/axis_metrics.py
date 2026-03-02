import json
import logging
import math
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from scenario_disjoint_cv import ScenarioDisjointValidator

logger = logging.getLogger(__name__)


def clean_text_for_matching(text):
    return re.sub(r"[^a-zA-Z\s]", "", text.lower())

def get_log_odds(df1, df2, df0):
    from collections import defaultdict

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
    deltas = get_log_odds(df1["response"], df2["response"], df0["response"])
    top_words = [k for k, v in deltas.items() if v > threshold]
    return sorted(top_words, key=lambda x: deltas[x], reverse=True)

def measure_individuation_axis(df, axis, default="Unmarked", use_scenario_cv=False):
    logger.info(f"\n--- MEASURING INDIVIDUATION FOR {axis.upper()} ---")
    results = {}
    values = [v for v in df[axis].unique() if v != default]

    for val in values:
        sub_df = df[df[axis].isin([val, default])]
        if len(sub_df) < 10:
            continue

        X = np.stack(sub_df["embedding"].values)
        y = (sub_df[axis] != default).astype(int)

        if use_scenario_cv and "scenario_id" in sub_df.columns:
            validator = ScenarioDisjointValidator(
                cv_strategy="GroupKFold",
                n_splits=min(5, len(np.unique(sub_df["scenario_id"]))),
                classifier_type="RandomForest"
            )
            groups = sub_df["scenario_id"].values
            cv_results = validator.validate_by_scenario(X, y, groups)
            results[val] = {
                "Accuracy": cv_results['accuracy_mean'],
                "Macro F1": cv_results['f1_macro_mean'],
            }
            logger.info(
                f"{axis.capitalize()} value: {val:<30} "
                f"CV Acc: {results[val]['Accuracy']:.2f} | CV F1: {results[val]['Macro F1']:.2f}"
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            clf = RandomForestClassifier(random_state=42)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)

            results[val] = {
                "Accuracy": accuracy_score(y_test, preds),
                "Macro F1": f1_score(y_test, preds),
            }
            logger.info(
                f"{axis.capitalize()} value: {val:<30} "
                f"Acc: {results[val]['Accuracy']:.2f} | F1: {results[val]['Macro F1']:.2f}"
            )

    return results

def measure_exaggeration_axis(
    df, emb_dict, axis, default_persona="Unmarked", default_topic="general_comment"
):
    print(f"\n--- MEASURING EXAGGERATION FOR {axis.upper()} (corpus-wide) ---")
    values = [v for v in df[axis].unique() if v != default_persona]

    exag_scores = []

    for v in values:
        sub_df = df[df[axis].isin([default_persona, v])]
        if sub_df.empty:
            continue

        df_default_persona = sub_df[sub_df[axis] == default_persona]
        df_default_topic = sub_df[(sub_df[axis] == v) & (sub_df["topic"] == default_topic)]
        df_target = sub_df[(sub_df[axis] == v) & (sub_df["topic"] != default_topic)]
        df_background = sub_df

        if df_default_persona.empty or df_default_topic.empty or df_target.empty:
            continue

        persona_seed_words = get_seed_words(df_default_topic, df_default_persona, df_background)
        topic_seed_words = get_seed_words(df_default_persona, df_default_topic, df_background)

        if not persona_seed_words or not topic_seed_words:
            continue

        print(f"\n--- SEED WORDS FOR: {axis}={v} ---")
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
        axis_norm = np.linalg.norm(axis_v)
        if axis_norm < 1e-8:
            logger.debug(f"Degenerate axis vector (norm={axis_norm:.2e}). Skipping.")
            continue
        axis_v = axis_v / axis_norm

        def cos_sim(a, b):
            denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
            return np.dot(a, b) / denom

        target_sims = [cos_sim(emb, axis_v) for emb in df_target["embedding"]]
        default_p_sims = [cos_sim(emb, axis_v) for emb in df_default_persona["embedding"]]
        default_t_sims = [cos_sim(emb, axis_v) for emb in df_default_topic["embedding"]]

        mean_target = np.mean(target_sims)
        mean_dp = np.mean(default_p_sims)
        mean_dt = np.mean(default_t_sims)

        denominator = mean_dt - mean_dp
        if abs(denominator) < 1e-6:
            logger.debug(f"Near-zero denominator ({denominator:.2e}). Skipping.")
            continue

        exag_score = (mean_target - mean_dp) / denominator
        if np.isnan(exag_score) or np.isinf(exag_score):
            logger.warning(f"Invalid exaggeration score (NaN/Inf): {axis}={v}")
            continue
        if abs(exag_score) > 100:
            logger.warning(f"Extreme exaggeration: {exag_score:.2f} for {axis}={v}")
        exag_scores.append({axis: v, "exaggeration": exag_score})
        print(f"{axis.capitalize()}: {v:<30} Exaggeration: {exag_score:.3f}")

    df_scores = pd.DataFrame(exag_scores)
    if df_scores.empty:
        print("\n[WARNING] No exaggeration scores were calculated for {axis}.")
        return df_scores

    print(f"\nMEAN EXAGGERATION BY {axis.upper()} (Lower is better):")
    print(df_scores.groupby(axis)["exaggeration"].mean().sort_values(ascending=False))
    return df_scores

def measure_axes(df, emb_dict, cv_strategy, output_dir):
    logger.info("4. Measuring Individuation with Scenario-Disjoint CV...")
    use_scenario_cv = cv_strategy in ["GroupKFold", "LeaveOneGroupOut"] and "scenario_id" in df.columns
    
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
    
    logger.info("\n=== EXAGGERATION DATAFRAMES ===")
    logger.info(f"Race exaggeration frame shape: {race_exag.shape}")
    logger.info(f"Gender exaggeration frame shape: {gender_exag.shape}")
    logger.info(f"Occupation exaggeration frame shape: {occupation_exag.shape}")
    
    if not race_exag.empty:
        race_exag.to_csv(output_dir / "exaggeration_race.csv", index=False)
        logger.info(f"Saved race exaggeration to exaggeration_race.csv")
    if not gender_exag.empty:
        gender_exag.to_csv(output_dir / "exaggeration_gender.csv", index=False)
        logger.info(f"Saved gender exaggeration to exaggeration_gender.csv")
    if not occupation_exag.empty:
        occupation_exag.to_csv(output_dir / "exaggeration_occupation.csv", index=False)
        logger.info(f"Saved occupation exaggeration to exaggeration_occupation.csv")