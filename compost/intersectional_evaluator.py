"""
Intersectional Joint Probability Evaluation

Measures individuation and caricaturization of intersectional persona simulations 
using round-robin pairwise comparisons or "Unmarked" baselines and high-dimensional
clustering metrics (Mahalanobis Distance) or Fightin' Words semantic axes.
"""

import numpy as np
import pandas as pd
import math
import re
import itertools
import sys
import os
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from sklearn.covariance import LedoitWolf

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from compost.scenario_disjoint_cv import ScenarioDisjointValidator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text_for_matching(text):
    return re.sub(r"[^a-zA-Z\s]", "", text.lower())

def get_log_odds(df1, df2, df0):
    """
    Compute Monroe et al. 'Fightin' Words' log-odds ratio with prior.
    
    Identifies words that statistically distinguish df1 from df2 using 
    Laplace smoothing and a prior distribution from df0.
    
    Args:
        df1: Series of texts (e.g., S_{p,_,c} - persona with default topic)
        df2: Series of texts (e.g., S_{_,t,c} - default persona with topic)
        df0: Series of texts (background corpus for prior estimation)
    
    Returns:
        Dictionary mapping words to z-scores: delta[word] = (log(l1) - log(l2)) / sigma
        Positive z-score: word is more prevalent in df1
        Negative z-score: word is more prevalent in df2
        |z-score| > 1.96 indicates statistical significance at p < 0.05
    """
    def _word_counts(text_series: pd.Series) -> Dict[str, float]:
        if text_series is None or text_series.empty:
            return {}

        tokens = (
            text_series.astype(str)
            .str.lower()
            .str.replace(r"[^a-zA-Z\s]", " ", regex=True)
            .str.split()
            .explode()
        )
        tokens = tokens[tokens.notna() & tokens.str.fullmatch(r"[a-zA-Z]{2,}")]
        if tokens.empty:
            return {}
        return tokens.value_counts().astype(float).to_dict()

    # Get word->frequency maps for df1/df2 and prior corpus.
    counts1_raw = _word_counts(df1)
    counts2_raw = _word_counts(df2)
    prior_raw = _word_counts(df0)

    vocab = set(counts1_raw) | set(counts2_raw) | set(prior_raw)
    if not vocab:
        return {}

    # Strict floating-point additive smoothing
    prior = defaultdict(float, {w: prior_raw.get(w, 0.0) + 0.5 for w in vocab})
    counts1 = defaultdict(float, {w: counts1_raw.get(w, 0.0) + 0.5 for w in vocab})
    counts2 = defaultdict(float, {w: counts2_raw.get(w, 0.0) + 0.5 for w in vocab})

    delta = defaultdict(float)

    n1 = float(sum(counts1.values()))
    n2 = float(sum(counts2.values()))
    nprior = float(sum(prior.values()))

    # Compute smoothed odds, variance, and z-score for each word.
    for word in vocab:
        c1, c2, p = counts1[word], counts2[word], prior[word]
        num1, num2 = c1 + p, c2 + p
        den1, den2 = (n1 + nprior) - num1, (n2 + nprior) - num2
        if den1 <= 0 or den2 <= 0: continue

        l1, l2 = num1 / den1, num2 / den2
        sigmasquared = (1.0 / (c1 + p)) + (1.0 / (c2 + p))
        if sigmasquared > 0:
            delta[word] = (math.log(l1) - math.log(l2)) / math.sqrt(sigmasquared)

    return delta

def get_seed_words(df1, df2, df0, threshold=1.96):
    """
    Extract seed words from Fightin' Words analysis using z-score threshold.
    
    Args:
        df1: Series of texts (e.g., S_{p,_,c})
        df2: Series of texts (e.g., S_{_,t,c})
        df0: Series of texts (background corpus)
        threshold: Z-score threshold for significance (default 1.96 for p < 0.05)
    
    Returns:
        List of seed words sorted by z-score (descending)
        Only includes words where |z-score| > threshold
    """
    deltas = get_log_odds(df1["response"], df2["response"], df0["response"])
    top_words = [
        k for k, v in deltas.items()
        if v > threshold and isinstance(k, str) and re.fullmatch(r"[a-zA-Z]{2,}", k)
    ]
    return sorted(top_words, key=lambda x: deltas[x], reverse=True)


class IntersectionalEvaluator:
    """
    Evaluates bias through the lens of intersectional identities.
    """
    
    def __init__(self, min_group_size: int = 10):
        self.min_group_size = min_group_size
        self.intersectional_results = {}
        self.skipped_groups = []
    
    def create_intersectional_tuple(
        self,
        demographic: str,
        gender: str,
        occupation: str
    ) -> str:
        """
        Create intersectional identity tuple from individual attributes.
        
        Args:
            demographic: Demographic group (e.g., "Hispanic", "Black", "Asian")
            gender: Gender identity (e.g., "Male", "Female")
            occupation: Occupation class (e.g., "Nurse", "CEO", "Cleaner")
            
        Returns:
            Intersectional tuple string (e.g., "Hispanic_Male_Nurse")
        """
        # always preserve every component, even if it is "Unmarked";
        # this makes the control label explicit (e.g. Unmarked_Unmarked_Nurse)
        parts = [demographic or "Unmarked",
                 gender or "Unmarked",
                 occupation or "Unmarked"]
        return "_".join(parts)
    
    def add_intersectional_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add intersectional identity column to DataFrame.
        
        Args:
            df: DataFrame with 'demographic', 'gender', 'occupation' columns
            
        Returns:
            DataFrame with new 'intersectional_id' column
        """
        df = df.copy()
        df['intersectional_id'] = df.apply(
            lambda row: self.create_intersectional_tuple(
                row.get('race', row.get('demographic', '')),
                row.get('gender', ''),
                row.get('occupation', '')
            ),
            axis=1
        )
        return df

    def _get_valid_pairs(
        self, 
        unique_groups: np.ndarray, 
        directed: bool = False, 
        baseline: str = "unmarked",
    ) -> List[Tuple[str, str]]:
        """Pairs each intersectional target persona with its control baseline. """

        if baseline == "unmarked":
            return self._get_valid_unmarked_pairs(unique_groups, directed)
        elif baseline == "pairwise":
            return self._get_valid_pairwise_pairs(unique_groups, directed)
        else:
            logger.warning(f"Baseline {baseline} does not exist. Using 'Unmarked' baseline.")
            return self._get_valid_unmarked_pairs(unique_groups, directed)

    def _get_valid_unmarked_pairs(self, unique_groups: np.ndarray, directed: bool = False) -> List[Tuple[str, str]]:
        """
        Pairs each intersectional target persona with its exact 
        'Unmarked_Unmarked_[Occupation]' baseline for CoMPosT evaluation.
        """
        pairs = []
        unique_groups_set = set(unique_groups)
        
        for g in unique_groups:
            parts = g.split('_')
            
            # Ensure proper format
            if len(parts) != 3: 
                continue
                
            race, gender, occupation = parts
            
            # Skip if this IS the control group
            if race == "Unmarked" and gender == "Unmarked":
                continue
                
            # Define the pure control for this specific occupation
            control_group = f"Unmarked_Unmarked_{occupation}"
            
            # Only add the pair if the control group actually exists in the data
            if control_group in unique_groups_set:
                if directed:
                    pairs.append((g, control_group))
                else:
                    # For undirected (Individuation), sort to avoid duplicates if 
                    # called iteratively, though (target, control) is generally fine
                    pairs.append((g, control_group))
                
        return pairs

    def _get_valid_pairwise_pairs(self, unique_groups: np.ndarray, directed: bool = False) -> List[Tuple[str, str]]:
        """
        Generates pairwise comparisons that isolate exactly ONE axis of variation.
        (e.g., Hispanic_Male_Nurse vs White_Male_Nurse).
        """
        pairs = []
        # Use combinations for undirected (Individuation), permutations for directed (Exaggeration)
        iterator = itertools.permutations(unique_groups, 2) if directed else itertools.combinations(unique_groups, 2)
        
        for g1, g2 in iterator:
            p1 = g1.split('_')
            p2 = g2.split('_')
            
            # Ensure proper format
            if len(p1) != 3 or len(p2) != 3: 
                continue
                
            r1, gen1, o1 = p1
            r2, gen2, o2 = p2
            
            # Must share the same occupation to form a valid counterfactual
            if o1 != o2: 
                continue
            
            # Must differ by exactly ONE demographic axis to isolate the variation
            diff_race = r1 != r2
            diff_gen = gen1 != gen2
            
            if (diff_race and not diff_gen) or (not diff_race and diff_gen):
                pairs.append((g1, g2))
                
        return pairs

    def measure_individuation(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        intersectional_col: str = 'intersectional_id',
        classifier_type: str = 'XGBoost',
        min_group_size: Optional[int] = None,
        evaluation_mode: str = 'cv',
        test_size: float = 0.2,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Measure individuation: Is S_{p,t,c} differentiable from S_{_,t,c}?
        
        Default mode uses scenario-disjoint cross-validation for more stable estimates.
        Optional mode uses scenario-disjoint grouped holdout (80/20 split).
        Near 50% indicates a "cardboard cutout" - no meaningful differentiation.

        Args:
            df: DataFrame containing intersectional_id, scenario_id, etc.
            embeddings: numpy array of Sentence-BERT embeddings (aligned with df)
            intersectional_col: column name for intersectional identity groups
            classifier_type: 'XGBoost', 'LinearSVC', 'RandomForest', 'GradientBoosting', or 'LogisticRegression'
            min_group_size: minimum samples required per group (default 10)
            evaluation_mode: 'cv' (default) or 'grouped_holdout'
            test_size: fraction of data for test set when evaluation_mode='grouped_holdout'
            n_splits: maximum number of folds for GroupKFold (default 5)
            random_state: random seed for reproducibility

        Returns:
            DataFrame with per-pair individuation metrics:
            - intersectional_id: target persona (e.g., "Hispanic_Male_Nurse")
            - control_id: control baseline (e.g., "Unmarked_Unmarked_Nurse")
            - accuracy: test set accuracy on binary classification task
            - f1_score: test set F1-macro score
            - n_target, n_control: sample counts per group
        """
        if evaluation_mode not in {"cv", "grouped_holdout"}:
            raise ValueError("evaluation_mode must be 'cv' or 'grouped_holdout'")

        threshold = min_group_size if min_group_size is not None else self.min_group_size
        results = []
        self.skipped_groups = []

        if "scenario_id" not in df.columns:
            raise ValueError(
                "measure_individuation requires a 'scenario_id' column for scenario-disjoint evaluation."
            )

        unique_groups = df[intersectional_col].unique()
        
        # Get UNDIRECTED pairwise comparisons
        pairs = self._get_valid_pairs(unique_groups, directed=False)
        logger.info(f"Measuring individuation for {len(pairs)} persona-control pairs (min_group_size={threshold})")

        for target, control_label in pairs:
            mask = df[intersectional_col].isin([target, control_label])
            sub_df = df[mask]
            n_t = (sub_df[intersectional_col] == target).sum()
            n_c = (sub_df[intersectional_col] == control_label).sum()
            
            if n_t < threshold or n_c < threshold:
                self.skipped_groups.append({
                    'target': target,
                    'control': control_label,
                    'reason': f"Below threshold ({n_t},{n_c} < {threshold})"
                })
                continue

            X = embeddings[mask]
            # Binary label: 1 = target persona, 0 = control
            y = (sub_df[intersectional_col] == target).astype(int).values
            # Always use scenario_id groups to enforce scenario-disjoint splitting.
            groups = sub_df["scenario_id"].astype(str).values

            num_unique_scenarios = len(np.unique(groups))
            if num_unique_scenarios < 2 and evaluation_mode == "cv":
                self.skipped_groups.append({
                    'target': target,
                    'control': control_label,
                    'reason': f"Not enough unique scenarios for CV (found {num_unique_scenarios})"
                })
                logger.warning(f"Skipping '{target}': Only {num_unique_scenarios} unique scenario(s) found, need at least 2 for GroupKFold.")
                continue

            if evaluation_mode == "cv":
                validator = ScenarioDisjointValidator(
                    cv_strategy="StratifiedGroupKFold",
                    n_splits=min(n_splits, len(np.unique(groups))),
                    classifier_type=classifier_type,
                )
                split_results = validator.validate_cv(
                    X=X,
                    y=y,
                    groups=groups,
                )
            else:
                validator = ScenarioDisjointValidator(
                    classifier_type=classifier_type,
                )
                split_results = validator.validate_grouped_holdout(
                    X=X,
                    y=y,
                    groups=groups,
                    test_size=test_size,
                    random_state=random_state,
                )

            results.append({
                'intersectional_id': target,
                'control_id': control_label,
                'n_target': n_t,
                'n_control': n_c,
                'accuracy': split_results['accuracy_mean'],
                'f1_score': split_results['f1_macro_mean'],
            })
            
            logger.info(f"Pair ({target}, {control_label}): f1={split_results['f1_macro_mean']:.4f}")

        df_results = pd.DataFrame(results)

        if self.skipped_groups:
            logger.warning(f"Skipped {len(self.skipped_groups)} pairs due to insufficient data: {self.skipped_groups}")

        logger.info(f"Computed individuation for {len(df_results)} pairs out of {len(pairs)} possible pairs")
        return df_results

    def measure_exaggeration(
        self,
        df: pd.DataFrame,
        emb_dict: Dict[str, np.ndarray],
        metric: str = "fighting_words"
    ) -> pd.DataFrame:
        
        if metric == "mahalanobis":
            return self.measure_exaggeration_mahalanobis(df)
        elif metric == "fighting_words":
            return self.measure_exaggeration_fighting_words(df, emb_dict)
        else:
            logger.warning(f"Metric strategy {metric} not recognized! Using Fightin' Words")
            return self.measure_exaggeration_fighting_words(df, emb_dict)

    def measure_exaggeration_fighting_words(
        self,
        df: pd.DataFrame,
        emb_dict: Dict[str, np.ndarray],
        default_topic: str = "general_comment",
    ) -> pd.DataFrame:
        """
        Measure exaggeration (caricature) using the Fightin' Words method and semantic axes.
        
        Methodology (CoMPosT specification):
        1. For each persona-topic pair, construct two semantic poles:
           - Persona pole P_p: Mean embedding of S_{p,_,c} (persona with default topic)
           - Topic pole P_t: Mean embedding of S_{_,t,c} (default persona with topic t)
        
        2. Identify seed words using Fightin' Words (Monroe log-odds):
           - Compare S_{p,_,c} vs S_{_,t,c} to find statistically significant words
           - Use words with z-score > 1.96 as seed words
           - Separate sets: W_p (persona-distinctive) and W_t (topic-distinctive)
        
        3. Construct semantic axis: V_{p,t} = Mean(P_p) - Mean(P_t)
        
        4. Evaluate target S_{p,t,c}:
           - Compute: cos(S_{p,t,c}, V_{p,t}) = mean of cosine similarities
           - Normalize to [0, 1]:
             - 0 = target is similar to topic pole (no personalization)
             - 1 = target is similar to persona pole (high caricature)
        
        Returns:
            DataFrame with columns: intersectional_id, control_id, exaggeration
        """
        exag_scores = []
        unique_groups = df["intersectional_id"].unique()
        pairs = self._get_valid_pairs(unique_groups, directed=True)

        def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
            denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
            return float(np.dot(a, b) / denom)

        for cohort_id, control in pairs:
            pair_df = df[df["intersectional_id"].isin([cohort_id, control])]
            
            # Get the persona pole (Sp,_,c) - This remains constant for the persona
            df_persona_pole = pair_df[(pair_df["intersectional_id"] == cohort_id) & (pair_df["topic"] == default_topic)]
            
            # We evaluate caricature PER TOPIC
            topics = pair_df[pair_df["topic"] != default_topic]["topic"].unique()
            for t in topics:
                # Topic pole (S_,t,c)
                df_topic_pole = pair_df[(pair_df["intersectional_id"] == control) & (pair_df["topic"] == t)]
                
                # Target (Sp,t,c)
                df_target = pair_df[(pair_df["intersectional_id"] == cohort_id) & (pair_df["topic"] == t)]

                if df_persona_pole.empty or df_topic_pole.empty or df_target.empty:
                    continue

                # Prior distribution: persona on default, control on topic, and target
                df_background = pd.concat([df_persona_pole, df_topic_pole])

                persona_seed_words = get_seed_words(df_persona_pole, df_topic_pole, df_background)
                topic_seed_words = get_seed_words(df_topic_pole, df_persona_pole, df_background)

                # Print the top 20 seed words
                logger.info(f"Target Cohort ({cohort_id}) Seed Words: {persona_seed_words[:20]}")
                logger.info(f"Control Cohort ({control}) Seed Words: {topic_seed_words[:20]}")

                if not persona_seed_words or not topic_seed_words:
                    continue

                def get_pole_embedding(target_df, seed_words):
                    patterns = [re.compile(rf"\b{re.escape(w)}\b") for w in seed_words]
                    embs = []
                    for sents in target_df["sentences"]:
                        for s in sents:
                            clean_s = re.sub(r"[^a-zA-Z\s]", "", s.lower())
                            if any(p.search(clean_s) for p in patterns):
                                embs.append(emb_dict[s])
                    if not embs:
                        return None
                    return np.mean(embs, axis=0)

                p_pole = get_pole_embedding(df_persona_pole, persona_seed_words)
                t_pole = get_pole_embedding(df_topic_pole, topic_seed_words)
                
                if p_pole is None or t_pole is None:
                    continue

                axis_v = p_pole - t_pole
                axis_norm = np.linalg.norm(axis_v)
                if axis_norm < 1e-8:
                    continue
                axis_v = axis_v / axis_norm

                target_sims = [cos_sim(emb_dict[s], axis_v) for s in df_target["sentences"].explode()]
                mean_target = np.mean(target_sims)
                
                default_p_sims = [cos_sim(emb_dict[s], axis_v) for s in df_persona_pole["sentences"].explode()]
                mean_dp = np.mean(default_p_sims)

                mean_t_pole = np.mean([cos_sim(emb_dict[s], axis_v) for s in df_topic_pole["sentences"].explode()])


                denominator = (mean_dp - mean_t_pole)
                if abs(denominator) < 1e-8:
                    # Poles are too close, use simple normalization: (cos_sim + 1) / 2
                    exag_score = (mean_target + 1.0) / 2.0
                else:
                    # Relative position: how close to persona pole vs topic pole
                    exag_score = (mean_target - mean_t_pole) / denominator
                    # Clip to [0, 1] as safety measure
                    exag_score = max(0.0, min(1.0, exag_score))
                
                exag_scores.append({
                    "intersectional_id": cohort_id, 
                    "control_id": control, 
                    "topic": t,               
                    "exaggeration": exag_score
                })

        res_df = pd.DataFrame(exag_scores)
        
        # Aggregate per-topic scores back down to one row per demographic pair
        if not res_df.empty:
            res_df = res_df.groupby(["intersectional_id", "control_id"])["exaggeration"].mean().reset_index()
            
        return res_df

    def measure_exaggeration_mahalanobis(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Computes the relative semantic distance (caricature/exaggeration) by calculating 
        the Mahalanobis Distance between the target and control intersectional clusters 
        in the high-dimensional embedding space.
        """
        exag_scores = []
        unique_groups = df["intersectional_id"].unique()

        # DIRECTED pairs: A vs B maps covariance to B (the control shape)
        pairs = self._get_valid_pairs(unique_groups, directed=True)

        for cohort_id, control in pairs:
            target_mask = df["intersectional_id"] == cohort_id
            control_mask = df["intersectional_id"] == control
            
            target_embs = np.stack(df[target_mask]["embedding"].values)
            control_embs = np.stack(df[control_mask]["embedding"].values)

            if len(target_embs) < 5 or len(control_embs) < 5:
                continue

            # Centroids of the clusters
            mu_t = np.mean(target_embs, axis=0)
            mu_c = np.mean(control_embs, axis=0)

            # Use Ledoit-Wolf Shrinkage to estimate a well-conditioned, noise-resistant covariance matrix
            # and handles the num_dims > num_samples problem mathematically
            lw = LedoitWolf()
            cov_c = lw.fit(control_embs).covariance_
            
            # Because Ledoit-Wolf guarantees a positive definite matrix, we can safely use standard inverse
            inv_cov_c = np.linalg.inv(cov_c)

            # Compute Mahalanobis Distance
            delta = mu_t - mu_c
            m_dist = np.sqrt(np.dot(np.dot(delta, inv_cov_c), delta))

            exag_scores.append({
                "intersectional_id": cohort_id, 
                "control_id": control, 
                "exaggeration": float(m_dist)
            })

        return pd.DataFrame(exag_scores)

    def compare_implicit_vs_explicit(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        intersectional_col: str = 'intersectional_id'
    ) -> pd.DataFrame:
        """
        Compare metrics between implicit and explicit variants for each
        intersectional cohort. Returns a merged dataframe with deltas.
        """
        imp_mask = df['variant_type'] == 'implicit'
        exp_mask = df['variant_type'] == 'explicit'
        imp_perf = self.measure_individuation(
            df[imp_mask], embeddings[imp_mask], intersectional_col=intersectional_col
        )
        exp_perf = self.measure_individuation(
            df[exp_mask], embeddings[exp_mask], intersectional_col=intersectional_col
        )
        
        # Merge on both target and control to accurately map the pairwise deltas
        merged = imp_perf.merge(exp_perf, on=['intersectional_id', 'control_id'], suffixes=('_imp','_exp'))
        
        for metric in ['accuracy','f1_score']:
            merged[f'{metric}_delta'] = merged[f'{metric}_exp'] - merged[f'{metric}_imp']
            
        return merged

    def compute_intersectional_parity(
        self,
        performance_df: pd.DataFrame,
        metric: str = 'exaggeration'
    ) -> Dict[str, float]:
        """
        Calculate statistical parity metrics across intersectional groups.
        
        Args:
            performance_df: DataFrame from evaluate_intersectional_groups
            metric: Performance metric to evaluate ('accuracy', 'f1_score', etc.)
            
        Returns:
            Dictionary of parity metrics (disparity ratio, gap, std deviation)
        """
        if metric not in performance_df.columns:
            raise ValueError(f"Metric '{metric}' not found in results dataframe")
        
        values = performance_df[metric].dropna()
        if len(values) == 0:
            return {}
        
        min_val = values.min()
        max_val = values.max()
        mean_val = values.mean()
        std_val = values.std()
        
        # Disparate impact ratio (Min / Max). 
        # For exaggeration, a low ratio means one group is caricatured drastically more than another.
        di_ratio = min_val / (max_val + 1e-6)
        
        return {
            f'{metric}_min': min_val,
            f'{metric}_max': max_val,
            f'{metric}_mean': mean_val,
            f'{metric}_std': std_val,
            f'{metric}_gap': max_val - min_val,
            f'{metric}_disparate_impact_ratio': di_ratio,
            f'{metric}_inequality_measure': std_val / (mean_val + 1e-6)  # Coefficient of variation
        }

    def generate_intersectional_report(
        self,
        performance_df: pd.DataFrame,
        parity_metrics: Dict[str, float],
        metric_name: str = 'exaggeration'
    ) -> str:
        report = """\n================ PAIRWISE INTERSECTIONAL BIAS REPORT ================\n\nRELATIVE SEMANTIC DISTANCES ACROSS GROUPS:\n"""
        
        if not performance_df.empty:
            # Sort to bring the most highly exaggerated pair relationships to the top
            sorted_df = performance_df.sort_values(by=metric_name, ascending=False)
            report += sorted_df.to_string(index=False)
        
        report += """\n\nSTATISTICAL PARITY ANALYSIS:\n"""
        
        for key, val in parity_metrics.items():
            if isinstance(val, float):
                report += f"  {key}: {val:.4f}\n"
        
        report += "\nINTERPRETATION:\n"
        
        di_key = f'{metric_name}_disparate_impact_ratio'
        if di_key in parity_metrics:
            di = parity_metrics[di_key]
            if di < 0.80:
                report += f"  ⚠️  Strong Disparate Caricature detected (ratio: {di:.3f}).\n"
                report += f"      The model exaggerates certain demographic swaps significantly more than others.\n"
            elif di < 0.90:
                report += f"  ⚠️  Moderate Disparate Caricature (ratio: {di:.3f}).\n"
            else:
                report += f"  ✓ Acceptable parity across groups (ratio: {di:.3f}). Exaggeration is applied relatively equally.\n"
        
        report += "=" * 73
        return report

    def get_fightin_words_poles(
        self,
        df: pd.DataFrame,
        target_id: str,
        control_id: str,
        variant_type: str = 'implicit',
        target_topic_id: Optional[str] = None,
        default_topic: str = "general_comment",
    ) -> Tuple[np.ndarray, float, float]:
        """
        Compute CoMPosT statistical poles for a given intersectional pair, optionally
        restricted to a single topic or scenario (target_topic_id), using data from a
        single variant type.

        This path is designed to be called separately for implicit and explicit
        variants to avoid variant contamination.

        Args:
            df: Full dataframe with all transcripts.
            target_id: Target intersectional identity (e.g., "Hispanic_Male_Nurse").
            control_id: Control intersectional identity (e.g., "Unmarked_Unmarked_Nurse").
            variant_type: Either 'implicit' or 'explicit' to use only that variant's data.
            target_topic_id: Specific scenario ID to restrict poles to. If None, uses all scenarios.
            default_topic: Label for the neutral baseline topic.

        Returns:
            axis_v: Normalized vector pointing from topic pole to persona pole.
            topic_pole_sim: Mean cosine similarity of control examples to axis_v.
            persona_pole_sim: Mean cosine similarity of target examples to axis_v.
        """
        # Filter to the requested variant; keep default-topic rows as baseline anchors.
        if 'variant_type' in df.columns:
            df_variant = df[
                (df['variant_type'] == variant_type) |
                (df['topic'] == default_topic)
            ].copy()
        else:
            df_variant = df.copy()

        # Restrict to just the two identity groups
        sub_df = df_variant[df_variant["intersectional_id"].isin([target_id, control_id])].copy()

        # Restrict to a specific scenario or topic if requested.
        if target_topic_id is not None:
            if "scenario_id" in sub_df.columns:
                sub_df = sub_df[
                    (sub_df["scenario_id"].astype(str) == str(target_topic_id)) |
                    (sub_df["topic"].astype(str) == str(target_topic_id)) |
                    (sub_df["topic"] == default_topic)
                ].copy()
            else:
                sub_df = sub_df[
                    (sub_df["topic"].astype(str) == str(target_topic_id)) |
                    (sub_df["topic"] == default_topic)
                ].copy()

        if "masked_text" in sub_df.columns:
            sub_df["response"] = sub_df["masked_text"].fillna("").astype(str)
        elif "target_text" in sub_df.columns:
            sub_df["response"] = sub_df["target_text"].fillna("").astype(str)
        elif "response" in sub_df.columns:
            sub_df["response"] = sub_df["response"].fillna("").astype(str)
        else:
            raise ValueError(
                "Dataframe must contain one of: 'masked_text', 'target_text', or 'response'."
            )

        # Both poles are built from default-topic rows to keep identity-only contrast.
        df_persona_pole = sub_df[
            (sub_df["intersectional_id"] == target_id) &
            (sub_df["topic"] == default_topic)
        ]
        df_topic_pole = sub_df[
            (sub_df["intersectional_id"] == control_id) &
            (sub_df["topic"] == default_topic)
        ]

        if not sub_df.empty:
            emb_dim = len(sub_df.iloc[0]['embedding'])
        else:
            emb_dim = 768

        if df_persona_pole.empty or df_topic_pole.empty:
            # Insufficient data for this scenario/variant combination
            return np.zeros(emb_dim), np.nan, np.nan

        # Use in-scope pair data as the Monroe prior without target leakage from held-out rows.
        persona_seed_words = get_seed_words(df_persona_pole, df_topic_pole, sub_df)
        topic_seed_words = get_seed_words(df_topic_pole, df_persona_pole, sub_df)

        # Print the top 20 seed words
        logger.info(f"Target Cohort ({target_id}, scenario={target_topic_id}) Seed Words: {persona_seed_words[:20]}")
        logger.info(f"Control Cohort ({control_id}) Seed Words: {topic_seed_words[:20]}")

        if not persona_seed_words or not topic_seed_words:
            # Monroe log-odds failed to find significant words
            return np.zeros(emb_dim), np.nan, np.nan

        def contains_any(text: str, words: list) -> bool:
            clean = re.sub(r"[^a-zA-Z\s]", "", text.lower())
            return any(re.search(rf"\b{re.escape(w)}\b", clean) for w in words)

        p_vecs = [emb for txt, emb in zip(df_persona_pole['response'], df_persona_pole['embedding'])
                  if contains_any(txt, persona_seed_words)]
        t_vecs = [emb for txt, emb in zip(df_topic_pole['response'], df_topic_pole['embedding'])
                  if contains_any(txt, topic_seed_words)]

        if p_vecs and t_vecs:
            p_pole = np.mean(p_vecs, axis=0)
            t_pole = np.mean(t_vecs, axis=0)
            axis_v = p_pole - t_pole
            norm = np.linalg.norm(axis_v)
            axis_v = axis_v / norm if norm > 1e-8 else np.zeros_like(axis_v)
        else:
            return np.zeros(emb_dim), np.nan, np.nan

        def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
            denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
            return float(np.dot(a, b) / denom)

        topic_pole_sim = np.mean([cos_sim(emb, axis_v) for emb in df_topic_pole['embedding']]) if not df_topic_pole.empty else np.nan
        persona_pole_sim = np.mean([cos_sim(emb, axis_v) for emb in df_persona_pole['embedding']]) if not df_persona_pole.empty else np.nan

        return axis_v, topic_pole_sim, persona_pole_sim