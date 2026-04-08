import os
import sys
import multiprocessing as mp

# Force 'spawn' start method before any other heavy libraries load.
# This prevents the "CUDA driver initialization failed" fork-context error.
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import argparse
import gc
import json
import nltk
import numpy as np
import pandas as pd
try:
    import torch
except ImportError:
    torch = None

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

nltk.download("punkt")
from nltk.tokenize import sent_tokenize

from allocational import AllocationalEvaluator
from representational import RepresentationalEvaluator
from compost.intersectional_evaluator import IntersectionalEvaluator


def get_document_embedding(text, model):
    sentences = sent_tokenize(text)
    if not sentences:
        return np.zeros(model.get_sentence_embedding_dimension())
    sentence_embeddings = model.encode(sentences)
    return np.mean(sentence_embeddings, axis=0)


def get_turn_embeddings(turn_texts, model):
    """Encodes each target turn separately for trajectory metrics."""
    if not turn_texts:
        return np.empty((0, model.get_sentence_embedding_dimension()), dtype=float)
    turn_embeddings = np.asarray(model.encode(turn_texts), dtype=float)
    if turn_embeddings.ndim == 1:
        turn_embeddings = turn_embeddings.reshape(1, -1)
    return turn_embeddings


def _encode_texts(model, texts, batch_size: int, pool=None):
    if pool is not None and hasattr(model, "encode_multi_process"):
        return model.encode_multi_process(texts, pool, chunk_size=batch_size)

    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def get_batched_document_embeddings(texts, model, batch_size: int = 128, pool=None):
    """Encodes documents as mean sentence embeddings in batched GPU/CPU passes."""
    sent_lists = [sent_tokenize(text) for text in texts]
    dim = model.get_sentence_embedding_dimension()
    out = np.zeros((len(texts), dim), dtype=float)

    flat_sents = []
    owners = []
    for i, sents in enumerate(sent_lists):
        for s in sents:
            flat_sents.append(s)
            owners.append(i)

    if not flat_sents:
        return [out[i] for i in range(len(texts))]

    sent_embs = _encode_texts(model, flat_sents, batch_size=batch_size, pool=pool)

    counts = np.zeros(len(texts), dtype=np.int32)
    for emb, doc_idx in zip(sent_embs, owners):
        out[doc_idx] += emb
        counts[doc_idx] += 1

    nonzero = counts > 0
    out[nonzero] /= counts[nonzero, None]
    return [out[i] for i in range(len(texts))]


def get_batched_turn_embeddings(turn_lists, model, batch_size: int = 128, pool=None):
    """Encodes all target turns in one batched pass and rebuilds per-dialogue arrays."""
    dim = model.get_sentence_embedding_dimension()
    flat_turns = []
    owners = []
    for i, turns in enumerate(turn_lists):
        for t in turns:
            flat_turns.append(t)
            owners.append(i)

    per_dialogue = [[] for _ in range(len(turn_lists))]
    if flat_turns:
        turn_embs = _encode_texts(model, flat_turns, batch_size=batch_size, pool=pool)
        for emb, doc_idx in zip(turn_embs, owners):
            per_dialogue[doc_idx].append(emb)

    for i in range(len(per_dialogue)):
        if per_dialogue[i]:
            per_dialogue[i] = np.asarray(per_dialogue[i], dtype=float)
            if per_dialogue[i].ndim == 1:
                per_dialogue[i] = per_dialogue[i].reshape(1, -1)
        else:
            per_dialogue[i] = np.empty((0, dim), dtype=float)
    return per_dialogue


def nanmean(values):
    """Stable mean that ignores NaNs and returns NaN if nothing is valid."""
    arr = np.array(values, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan


def paired_delta(explicit_series: pd.Series, implicit_series: pd.Series) -> float:
    """Compute mean explicit-implicit delta on aligned keys only."""
    paired = pd.concat(
        [implicit_series, explicit_series], axis=1, keys=["implicit", "explicit"]
    ).dropna()
    if paired.empty:
        return np.nan
    return float((paired["explicit"] - paired["implicit"]).mean())


def load_all_transcripts(file_paths):
    data = []
    for path in file_paths:
        with open(path, "r") as f:
            data.extend(json.load(f))
    df = pd.DataFrame(data)

    # Extract identity metadata
    df["demographic"] = df["metadata"].apply(
        lambda x: x["persona"].get("demographic", "Unmarked")
    )
    df["gender"] = df["metadata"].apply(
        lambda x: x["persona"].get("gender", "Unmarked")
    )
    df["occupation"] = df["metadata"].apply(
        lambda x: x["persona"].get("occupation", "Unmarked")
    )
    df["topic"] = df["metadata"].apply(
        lambda x: x.get("task_description", "general_comment")
    )
    df["target_logprobs"] = df["metadata"].apply(lambda x: x.get("target_logprobs", []))
    df["scenario_id"] = df["metadata"].apply(lambda x: x.get("scenario_id", "unknown"))

    return df


def main(args):
    from sentence_transformers import SentenceTransformer

    def resolve_embedding_devices():
        if args.embedding_device != "cuda":
            return None
        if torch is None or not torch.cuda.is_available():
            return None

        device_count = torch.cuda.device_count()
        if device_count <= 1:
            return None
        return [f"cuda:{i}" for i in range(device_count)]

    def run_embeddings_with_backoff(df_local, embedder_model, embedding_pool=None):
        """Retry embedding passes with smaller batches on CUDA OOM."""
        candidate_batches = [args.embedding_batch_size, 128, 64, 32, 16]
        seen = set()
        candidate_batches = [b for b in candidate_batches if not (b in seen or seen.add(b))]

        last_error = None
        for batch_size in candidate_batches:
            try:
                print(f"Encoding document embeddings in batches (batch_size={batch_size})...")
                df_local["embedding"] = get_batched_document_embeddings(
                    df_local["masked_text"].tolist(),
                    embedder_model,
                    batch_size=batch_size,
                    pool=embedding_pool,
                )

                print(f"Encoding turn embeddings in batches (batch_size={batch_size})...")
                df_local["turn_embeddings"] = get_batched_turn_embeddings(
                    df_local["masked_turn_texts"].tolist(),
                    embedder_model,
                    batch_size=batch_size,
                    pool=embedding_pool,
                )
                return
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                last_error = e
                print(
                    f"Warning: CUDA OOM during embedding with batch_size={batch_size}. Retrying with a smaller batch..."
                )
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        raise last_error

    print("Loading data...")
    target_path = os.path.join(args.dir, "target_simulations.json")
    control_path = os.path.join(args.dir, "control_simulations.json")
    default_topic_path = os.path.join(args.dir, "default_topics.json")

    df = load_all_transcripts([target_path, control_path, default_topic_path])

    print("Transcripts loaded. Initializing evaluators")
    rep_eval = RepresentationalEvaluator()
    ie = IntersectionalEvaluator()

    print("Creating intersectional IDs")
    # Create intersectional IDs
    df["intersectional_id"] = df.apply(
        lambda row: ie.create_intersectional_tuple(
            row.get("demographic", "Unmarked"),
            row.get("gender", "Unmarked"),
            row.get("occupation", "Unmarked"),
        ),
        axis=1,
    )

    print("Generating embeddings from masked data...")

    def extract_target_text(transcript):
        return " ".join(
            [turn["content"] for turn in transcript if turn["speaker"] == "Target"]
        )

    def extract_target_turn_texts(transcript):
        return [turn["content"] for turn in transcript if turn["speaker"] == "Target"]

    df["target_text"] = df["transcript"].apply(extract_target_text)
    df["target_turn_texts"] = df["transcript"].apply(extract_target_turn_texts)

    if args.masked_path:
        print(f"Loading masked data from {args.masked_path}...")
        with open(args.masked_path, "r") as f:
            masked_data = json.load(f)

        masked_df = pd.DataFrame(masked_data)
        masked_df["masked_text"] = masked_df["transcript"].apply(extract_target_text)
        masked_df["masked_turn_texts"] = masked_df["transcript"].apply(
            extract_target_turn_texts
        )

        # Merge the masked text into the main dataframe
        df = pd.merge(
            df,
            masked_df[["dialogue_id", "masked_text", "masked_turn_texts"]],
            on="dialogue_id",
            how="left",
            suffixes=("", "_masked"),
        )
        # Control simulations and other non-target simulations will have NaN in 'masked_text', so fill them
        df["masked_text"] = df["masked_text"].fillna(df["target_text"])
        missing_turns = df["masked_turn_texts"].isna()
        df.loc[missing_turns, "masked_turn_texts"] = df.loc[
            missing_turns, "target_turn_texts"
        ]
    else:
        print(
            "Warning: No masked data file provided. Using unmasked text for all embeddings."
        )
        df["masked_text"] = df["target_text"]
        df["masked_turn_texts"] = df["target_turn_texts"]

    if args.embedding_device == "auto":
        embedder_device = (
            "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        )
    else:
        embedder_device = args.embedding_device

    embedding_devices = resolve_embedding_devices()
    embedding_pool = None
    if embedding_devices:
        print("Using multi-GPU embedding pool on: " + ", ".join(embedding_devices))
        embedder_device = "cpu"

    print(f"Loading embedding model on device={embedder_device}...")
    embedder = SentenceTransformer("all-mpnet-base-v2", device=embedder_device)

    if embedding_devices:
        embedding_pool = embedder.start_multi_process_pool(target_devices=embedding_devices)

    try:
        run_embeddings_with_backoff(df, embedder, embedding_pool=embedding_pool)
    finally:
        if embedding_pool is not None:
            embedder.stop_multi_process_pool(embedding_pool)
        del embedding_pool
        del embedder
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # Load the large vLLM judge only after embeddings complete to avoid GPU memory contention.
    print("Initializing allocational evaluator (vLLM judge)...")
    alloc_eval = AllocationalEvaluator(
        model_path=args.judge_model, tensor_parallel_size=args.tensor_parallel_size
    )

    final_results = []
    target_groups = sorted(
        [g for g in df["intersectional_id"].unique() if "Unmarked_Unmarked" not in g]
    )

    # Slice the target_groups based on the chunk index and total chunks
    if args.total_chunks > 1:
        chunk_size = int(np.ceil(len(target_groups) / args.total_chunks))
        start_index = args.chunk_index * chunk_size
        end_index = start_index + chunk_size
        target_groups = target_groups[start_index:end_index]

    print(f"Processing metrics for {len(target_groups)} intersectional groups...")

    CACHE_FILE = f"llm_judge_cache_chunk_{args.chunk_index}.json"
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            judge_cache = json.load(f)
    else:
        judge_cache = {}

    def get_cached_judges_batch(d_ids, variant_types, transcripts, metadatas):
        """Fetches judgements from cache, or calls API in batches and saves so we never lose progress."""
        scores = [None] * len(d_ids)
        missing_dialogues = []
        missing_indices = []

        for i, (d_id, v, t, m) in enumerate(zip(d_ids, variant_types, transcripts, metadatas)):
            task_desc = m.get("task_description", "")
            cache_key = f"{d_id}::{v}::{task_desc}"
            
            if cache_key in judge_cache:
                cached_value = judge_cache[cache_key]
                if cached_value in (0, 1):
                    scores[i] = cached_value
                    continue
            
            missing_dialogues.append({
                "metadata": {"task_description": task_desc},
                "transcript": t,
                "cache_key": cache_key
            })
            missing_indices.append(i)

        if missing_dialogues:
            new_scores = alloc_eval.batch_evaluate_gcr(missing_dialogues)
            for idx, score, dialogue in zip(missing_indices, new_scores, missing_dialogues):
                scores[idx] = score
                # Ensure we only cache solid numerical hits, ignoring the newly caught NaNs
                if pd.notna(score) and score in (0, 1, 0.0, 1.0):
                    judge_cache[dialogue["cache_key"]] = int(score)
            
            with open(CACHE_FILE, "w") as f:
                json.dump(judge_cache, f)

        return scores

    for target_id in target_groups:
        # 1. Isolate the data for this specific intersectional group
        group_df = df[df["intersectional_id"] == target_id]

        # 2. Split into Implicit and Explicit datasets
        implicit_df = group_df[group_df["variant_type"] == "implicit"].copy()
        explicit_df = group_df[group_df["variant_type"] == "explicit"].copy()

        if implicit_df.empty or explicit_df.empty:
            print(f"Skipping {target_id}: Missing implicit or explicit data.")
            continue

        # --- A. ALLOCATIONAL METRICS ---

        # 1. Use the checkpointed judge function (Requires 'dialogue_id')
        imp_success_list = get_cached_judges_batch(
            implicit_df["dialogue_id"],
            implicit_df["variant_type"],
            implicit_df["transcript"],
            implicit_df["metadata"]
        )

        exp_success_list = get_cached_judges_batch(
            explicit_df["dialogue_id"],
            explicit_df["variant_type"],
            explicit_df["transcript"],
            explicit_df["metadata"]
        )

        implicit_df = implicit_df.copy()
        explicit_df = explicit_df.copy()
        implicit_df["success"] = imp_success_list
        explicit_df["success"] = exp_success_list

        implicit_gcr = nanmean(imp_success_list)
        explicit_gcr = nanmean(exp_success_list)
        d_gcr = alloc_eval.calculate_d_gcr(implicit_gcr, explicit_gcr)

        # Scenario-paired delta avoids composition bias from unequal scenario mixes.
        implicit_scenario_success = implicit_df.groupby("scenario_id", dropna=False)[
            "success"
        ].mean()
        explicit_scenario_success = explicit_df.groupby("scenario_id", dropna=False)[
            "success"
        ].mean()
        d_gcr_paired = paired_delta(
            explicit_scenario_success, implicit_scenario_success
        )

        # 2. ATC is computed over successful runs only.
        imp_atcs_raw = [
            alloc_eval.calculate_atc(t, is_successful=succ)
            for t, succ in zip(implicit_df["transcript"], imp_success_list)
        ]
        implicit_atc = nanmean([x for x in imp_atcs_raw if x is not None])

        exp_atcs_raw = [
            alloc_eval.calculate_atc(t, is_successful=succ)
            for t, succ in zip(explicit_df["transcript"], exp_success_list)
        ]
        explicit_atc = nanmean([x for x in exp_atcs_raw if x is not None])

        d_atc = nanmean(explicit_atc) - nanmean(implicit_atc)

        # Rejection diagnostics (count of refusal-like turns per dialogue).
        implicit_rejection_counts = [
            alloc_eval.calculate_rejection_rate(t) for t in implicit_df["transcript"]
        ]
        explicit_rejection_counts = [
            alloc_eval.calculate_rejection_rate(t) for t in explicit_df["transcript"]
        ]
        d_rejection_count = nanmean(explicit_rejection_counts) - nanmean(
            implicit_rejection_counts
        )

        # --- REPRESENTATIONAL METRICS (CONFIDENCE) ---
        # Compute confidence per dialogue first to avoid over-weighting longer outputs.
        implicit_dialogue_conf = [
            rep_eval.calculate_confidence(
                logprobs if isinstance(logprobs, list) else []
            )
            for logprobs in implicit_df["target_logprobs"]
        ]
        explicit_dialogue_conf = [
            rep_eval.calculate_confidence(
                logprobs if isinstance(logprobs, list) else []
            )
            for logprobs in explicit_df["target_logprobs"]
        ]

        d_ccd = rep_eval.calculate_d_ccd(implicit_dialogue_conf, explicit_dialogue_conf)

        implicit_df["dialogue_confidence"] = implicit_dialogue_conf
        explicit_df["dialogue_confidence"] = explicit_dialogue_conf
        implicit_scenario_conf = implicit_df.groupby("scenario_id", dropna=False)[
            "dialogue_confidence"
        ].mean()
        explicit_scenario_conf = explicit_df.groupby("scenario_id", dropna=False)[
            "dialogue_confidence"
        ].mean()
        d_ccd_paired = paired_delta(explicit_scenario_conf, implicit_scenario_conf)

        # --- SEMANTIC STEERING (CoMPosT INTEGRATION) ---
        # Dynamically define the counterfactual to extract the exact CoMPosT axis
        occupation = target_id.split("_")[-1]
        control_id = f"Unmarked_Unmarked_{occupation}"

        implicit_steerings = []
        explicit_steerings = []
        implicit_trajectories = []
        explicit_trajectories = []

        if control_id in df["intersectional_id"].values:
            # Get unique scenarios (topics) for this group, excluding the general_comment baseline
            unique_scenarios = group_df[group_df["topic"] != "general_comment"][
                "scenario_id"
            ].unique()

            for scenario_id in unique_scenarios:
                try:
                    # Use one shared axis per scenario for both variants.
                    axis_v, topic_pole_sim, persona_pole_sim = (
                        ie.get_fightin_words_poles(
                            df,
                            target_id,
                            control_id,
                            variant_type="implicit",
                            target_topic_id=scenario_id,
                            default_topic="general_comment",
                        )
                    )

                    if np.isnan(topic_pole_sim) or np.isnan(persona_pole_sim):
                        continue

                    implicit_scenario_df = implicit_df[
                        (implicit_df["scenario_id"] == scenario_id)
                        & (implicit_df["topic"] != "general_comment")
                    ]
                    explicit_scenario_df = explicit_df[
                        (explicit_df["scenario_id"] == scenario_id)
                        & (explicit_df["topic"] != "general_comment")
                    ]

                    if implicit_scenario_df.empty or explicit_scenario_df.empty:
                        continue

                    imp_target_embs = np.vstack(
                        implicit_scenario_df["embedding"].values
                    )
                    exp_target_embs = np.vstack(
                        explicit_scenario_df["embedding"].values
                    )
                    steer_dict = rep_eval.calculate_semantic_steering(
                        implicit_target_embeddings=imp_target_embs,
                        explicit_target_embeddings=exp_target_embs,
                        axis_v=axis_v,
                        topic_pole_sim=topic_pole_sim,
                        persona_pole_sim=persona_pole_sim,
                    )
                    trajectory_dict = rep_eval.calculate_semantic_steering_trajectory(
                        implicit_turn_embeddings=implicit_scenario_df[
                            "turn_embeddings"
                        ].tolist(),
                        explicit_turn_embeddings=explicit_scenario_df[
                            "turn_embeddings"
                        ].tolist(),
                        axis_v=axis_v,
                        topic_pole_sim=topic_pole_sim,
                        persona_pole_sim=persona_pole_sim,
                    )

                    if np.isfinite(steer_dict["implicit_steering"]):
                        implicit_steerings.append(steer_dict["implicit_steering"])
                    if np.isfinite(steer_dict["explicit_steering"]):
                        explicit_steerings.append(steer_dict["explicit_steering"])

                    imp_traj = np.asarray(
                        trajectory_dict["implicit_trajectory"], dtype=float
                    )
                    exp_traj = np.asarray(
                        trajectory_dict["explicit_trajectory"], dtype=float
                    )
                    if imp_traj.size and np.isfinite(imp_traj).any():
                        implicit_trajectories.append(imp_traj)
                    if exp_traj.size and np.isfinite(exp_traj).any():
                        explicit_trajectories.append(exp_traj)

                except (ValueError, np.linalg.LinAlgError):
                    # Monroe log-odds failed or insufficient data for this scenario
                    continue

        # Average scores across all valid scenarios to get final steering metrics
        final_imp_steer = (
            np.nanmean(implicit_steerings) if implicit_steerings else np.nan
        )
        final_exp_steer = (
            np.nanmean(explicit_steerings) if explicit_steerings else np.nan
        )
        final_delta_steer = (
            final_exp_steer - final_imp_steer
            if not (np.isnan(final_imp_steer) or np.isnan(final_exp_steer))
            else np.nan
        )

        def aggregate_trajectory(trajectory_list):
            if not trajectory_list:
                return []
            max_len = max(len(t) for t in trajectory_list)
            out = []
            for i in range(max_len):
                vals = [t[i] for t in trajectory_list if i < len(t) and np.isfinite(t[i])]
                out.append(float(np.mean(vals)) if vals else np.nan)
            return out

        final_imp_traj = aggregate_trajectory(implicit_trajectories)
        final_exp_traj = aggregate_trajectory(explicit_trajectories)

        max_len_delta = max(len(final_imp_traj), len(final_exp_traj))
        final_delta_traj = []
        for i in range(max_len_delta):
            imp_val = final_imp_traj[i] if i < len(final_imp_traj) else np.nan
            exp_val = final_exp_traj[i] if i < len(final_exp_traj) else np.nan
            if np.isfinite(imp_val) and np.isfinite(exp_val):
                final_delta_traj.append(float(exp_val - imp_val))
            else:
                final_delta_traj.append(np.nan)

        steering_scores = {
            "implicit_steering": final_imp_steer,
            "explicit_steering": final_exp_steer,
            "delta_steering": final_delta_steer,
        }

        # 3. Compile the group's results
        final_results.append(
            {
                "group_label": target_id,
                "implicit_GCR": implicit_gcr,
                "explicit_GCR": explicit_gcr,
                "d_GCR": d_gcr,
                "d_GCR_paired_by_scenario": d_gcr_paired,
                "implicit_ATC": implicit_atc,
                "explicit_ATC": explicit_atc,
                "d_ATC": d_atc,
                "d_CCD": d_ccd,
                "d_CCD_paired_by_scenario": d_ccd_paired,
                "implicit_rejection_count": nanmean(implicit_rejection_counts),
                "explicit_rejection_count": nanmean(explicit_rejection_counts),
                "d_rejection_count": d_rejection_count,
                "implicit_steering_n_valid": len(implicit_steerings),
                "explicit_steering_n_valid": len(explicit_steerings),
                "implicit_Steering": steering_scores["implicit_steering"],
                "explicit_Steering": steering_scores["explicit_steering"],
                "delta_Steering": steering_scores["delta_steering"],
                "implicit_Steering_Trajectory": json.dumps(final_imp_traj),
                "explicit_Steering_Trajectory": json.dumps(final_exp_traj),
                "delta_Steering_Trajectory": json.dumps(final_delta_traj),
                "implicit_steering_traj_n_valid": len(implicit_trajectories),
                "explicit_steering_traj_n_valid": len(explicit_trajectories),
            }
        )

    # 4. Save and Report
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    results_df = pd.DataFrame(final_results)
    results_df.to_csv(args.out, index=False)
    print("\nEvaluation Complete. Results saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate dynamic intersectional bias."
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Directory containing the transcript JSONs",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="dynamic_bias_results.csv",
        help="Output CSV filename",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        required=True,
        help="HuggingFace ID or local path for the LLM judge",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=2,
        help="Number of GPUs to use for vLLM",
    )
    parser.add_argument(
        "--chunk_index", type=int, default=0, help="Index of the current chunk"
    )
    parser.add_argument(
        "--total_chunks", type=int, default=1, help="Total number of chunks"
    )
    parser.add_argument(
        "--masked_path",
        type=str,
        default=None,
        help="Path to the masked simulations file",
    )
    parser.add_argument(
        "--embedding_batch_size",
        type=int,
        default=128,
        help="Batch size for sentence-transformer encoding",
    )
    parser.add_argument(
        "--embedding_device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for sentence-transformer encoding",
    )
    args = parser.parse_args()

    target_path = os.path.join(args.dir, "target_simulations.json")
    control_path = os.path.join(args.dir, "control_simulations.json")
    default_topic_path = os.path.join(args.dir, "default_topics.json")

    print(f"Evaluating transcripts in: {args.dir}")
    main(args)
