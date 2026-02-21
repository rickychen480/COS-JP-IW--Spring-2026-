"""
python compost/embeddings/compost_evaluator.py \
    --data data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/control_simulations.json \
        data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/default_topics.json \
        data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/target_simulations.json
"""

import json
import argparse
import numpy as np
import pandas as pd
import math
import re
from collections import defaultdict
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from numpy import dot
from numpy.linalg import norm


nltk.download("punkt_tab")

def clean_text_for_matching(text):
    """Cleans text exactly as done in get_log_odds to ensure regex matching aligns."""
    return re.sub(r"[^a-zA-Z\s]", "", text.lower())


def load_transcripts_to_dataframe(json_paths):
    """Parses multiple JSON files into a single Pandas DataFrame."""
    rows = []

    # Loop through all provided file paths
    for path in json_paths:
        print(f"Loading {path}...")
        with open(path, "r") as f:
            data = json.load(f)

        for d in data:
            persona_dict = d["metadata"]["persona"]
            # Define the Persona String. Treat the control group as the "Default-Persona"
            if persona_dict.get("demographic") == "Unmarked":
                p_str = "Unmarked"
            else:
                p_str = f"{persona_dict.get('demographic')} {persona_dict.get('gender')} {persona_dict.get('occupation')}"

            t_str = d["metadata"]["task_description"]

            # Flatten all "User" dialogue turns into a single string for analysis
            user_text = " ".join(
                [t["content"] for t in d["transcript"] if t["speaker"] == "User"]
            )

            rows.append({"persona": p_str, "topic": t_str, "response": user_text})

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


def measure_individuation(df, default_persona="Unmarked"):
    """Trains a Random Forest to differentiate the Persona from the Default Persona."""
    print("\n--- MEASURING INDIVIDUATION ---")
    results = {}
    personas = [p for p in df.persona.unique() if p != default_persona]
    topics = [t for t in df.topic.unique() if t != "general_comment"]

    for p in personas:
        acc_scores = []
        f1_scores = []
        for t in topics:
            # Get subset of data for this topic comparing Target Persona vs Default Persona
            sub_df = df[(df["topic"] == t) & (df["persona"].isin([p, default_persona]))]
            if len(sub_df) < 10:  # Skip if not enough samples to train/test
                continue

            X = np.stack(sub_df["embedding"].values)
            y = sub_df.persona.astype("category").cat.codes

            # 80/20 Stratified Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            clf = RandomForestClassifier(random_state=42)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='macro')
            
            acc_scores.append(acc)
            f1_scores.append(f1)

        if acc_scores:
            mean_acc = np.mean(acc_scores)
            mean_f1 = np.mean(f1_scores)
            results[p] = {"Accuracy": mean_acc, "Macro F1": mean_f1}
            print(
                f"Persona: {p:<35} Acc: {mean_acc:.2f} | F1: {mean_f1:.2f} (Target > 0.5)"
            )

    return results


def measure_exaggeration(
    df, emb_dict, default_persona="Unmarked", default_topic="general_comment"
):
    """Computes Exaggeration via Contextualized Semantic Axes."""
    print("\n--- MEASURING EXAGGERATION (CARICATURE) ---")
    personas = [p for p in df.persona.unique() if p != default_persona]
    topics = [t for t in df.topic.unique() if t != default_topic]

    exag_scores = []

    for t in topics:
        for p in personas:
            df_default_persona = df[(df["persona"] == default_persona) & (df["topic"] == t)]
            df_default_topic = df[(df["persona"] == p) & (df["topic"] == default_topic)]
            df_target = df[(df["persona"] == p) & (df["topic"] == t)]
            df_background = df[
                (df["persona"].isin([default_persona, p]))
                & (df["topic"].isin([default_topic, t]))
            ]

            if df_default_persona.empty or df_default_topic.empty or df_target.empty:
                continue

            persona_seed_words = get_seed_words(df_default_topic, df_default_persona, df_background)
            topic_seed_words = get_seed_words(df_default_persona, df_default_topic, df_background)

            if not persona_seed_words or not topic_seed_words:
                continue

            def get_pole_embedding(target_df, seed_words):
                # Enforce exact word boundaries using Regex
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

            def cos_sim(a, b):
                return dot(a, b) / (norm(a) * norm(b))

            target_sims = [cos_sim(emb, axis_v) for emb in df_target["embedding"]]
            default_p_sims = [cos_sim(emb, axis_v) for emb in df_default_persona["embedding"]]
            default_t_sims = [cos_sim(emb, axis_v) for emb in df_default_topic["embedding"]]

            mean_target = np.mean(target_sims)
            mean_dp = np.mean(default_p_sims)
            mean_dt = np.mean(default_t_sims)

            # Normalize Exaggeration Score (0 to 1)
            # (Target - DefaultPersona) / (DefaultTopic - DefaultPersona)
            if mean_dt - mean_dp == 0:
                continue

            exag_score = (mean_target - mean_dp) / (mean_dt - mean_dp)
            exag_scores.append({"persona": p, "topic": t, "exaggeration": exag_score})
            print(f"Persona: {p:<30} Topic: {t[:20]:<20} Exaggeration: {exag_score:.3f}")

    df_scores = pd.DataFrame(exag_scores)

    if df_scores.empty:
        print("\n[WARNING] No exaggeration scores were calculated.")
        return df_scores

    print("\nMEAN EXAGGERATION BY PERSONA (Lower is better):")
    print(df_scores.groupby("persona")["exaggeration"].mean().sort_values(ascending=False))
    return df_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        nargs="+",
        required=True,
        help="Paths to the transcript JSONs",
    )
    args = parser.parse_args()

    print("1. Loading Data...")
    df = load_transcripts_to_dataframe(args.data)
    print(f"Loaded {len(df)} total transcripts.")

    print("2. Tokenizing sentences...")
    df["sentences"] = df["response"].apply(sent_tokenize)

    print("3. Generating Sentence-BERT Embeddings & Caching...")
    model = SentenceTransformer("all-mpnet-base-v2")
    
    # Pre-encode and cache all unique sentences
    all_sentences = set()
    for sents in df["sentences"]:
        all_sentences.update(sents)
    all_sentences = list(all_sentences)
    
    print(f"Encoding {len(all_sentences)} unique sentences...")
    sentence_embeddings = model.encode(all_sentences, show_progress_bar=True)
    emb_dict = dict(zip(all_sentences, sentence_embeddings))

    # Calculate document-level embeddings via mean-pooling to bypass the 384 token limit safely
    def get_doc_embedding(sents):
        if not sents:
            return np.zeros(model.get_sentence_embedding_dimension())
        return np.mean([emb_dict[s] for s in sents], axis=0)

    df["embedding"] = df["sentences"].apply(get_doc_embedding)

    # 4. Run Empirical CoMPosT
    measure_individuation(df, default_persona="Unmarked")
    measure_exaggeration(
        df, emb_dict, default_persona="Unmarked", default_topic="general_comment"
    )