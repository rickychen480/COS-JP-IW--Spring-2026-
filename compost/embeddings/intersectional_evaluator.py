"""
Intersectional Joint Probability Evaluation Module
Treats multidimensional identities as indivisible constructs to capture
their unique linguistic realities. Evaluates intersectional tuples (e.g.,
Hispanic_Male_Nurse) against an "Unmarked" baseline using log-odds calculations.
"""

import numpy as np
import pandas as pd
import math
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scenario_disjoint_cv import ScenarioDisjointValidator
from axis_metrics import get_seed_words
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntersectionalEvaluator:
    """
    Evaluates bias through the lens of intersectional identities rather than
    isolated demographic dimensions.
    """
    
    def __init__(self, baseline_label: str = "Unmarked", min_group_size: int = 10):
        """
        Initialize intersectional evaluator.
        
        Args:
            baseline_label: The "unmarked" baseline identity to compare against
            min_group_size: Minimum number of samples required per group for evaluation.
                           Groups below this threshold will be skipped with a warning.
                           Default: 10 (provides reasonable statistical power)
        """
        self.baseline_label = baseline_label
        self.min_group_size = min_group_size
        self.intersectional_results = {}
        self.log_odds_cache = {}
        self.skipped_groups = []  # Track which groups were excluded
    
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
        parts = [demographic or self.baseline_label,
                 gender or self.baseline_label,
                 occupation or self.baseline_label]
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
                row.get('demographic', ''),
                row.get('gender', ''),
                row.get('occupation', '')
            ),
            axis=1
        )
        return df

    def measure_individuation(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        intersectional_col: str = 'intersectional_id',
        cv_strategy: str = 'GroupKFold',
        n_splits: int = 5,
        classifier_type: str = 'RandomForest',
        min_group_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Evaluate classification performance for each intersectional cohort paired
        with its occupational counterfactual. Training is done with
        scenario-disjoint cross-validation on the combined examples.

        Args:
            df: DataFrame containing at least the intersectional column and
                a "scenario_id" column.
            embeddings: numpy array of document embeddings aligned with df.
            intersectional_col: name of the column holding the cohort label.
            cv_strategy: cross-validation strategy to hand to the validator.
            n_splits: maximum number of folds for GroupKFold.
            classifier_type: type of sklearn classifier to instantiate.
            min_group_size: override default minimum group size per cohort/control.

        Returns:
            DataFrame with per-cohort metrics (accuracy, f1, sample counts, etc.)
        """
        threshold = min_group_size if min_group_size is not None else self.min_group_size
        results = []
        self.skipped_groups = []

        unique_groups = df[intersectional_col].unique()
        logger.info(f"Evaluating {len(unique_groups)} intersectional groups with pairing (min_group_size={threshold})")

        def occupation_of(label: str) -> str:
            return label.split("_")[-1] if isinstance(label, str) else ""

        # build mapping from each non‑baseline group to its counterfactual label
        for target in unique_groups:
            if target == self.baseline_label or target.startswith(self.baseline_label + "_" + self.baseline_label):
                continue
            control_label = f"{self.baseline_label}_{self.baseline_label}_{occupation_of(target)}"
            if control_label not in unique_groups:
                logger.warning(f"Control '{control_label}' not found for target '{target}'. Skipping.")
                continue

            mask = df[intersectional_col].isin([target, control_label])
            sub_df = df[mask]
            n_t = (sub_df[intersectional_col] == target).sum()
            n_c = (sub_df[intersectional_col] == control_label).sum()
            if n_t < threshold or n_c < threshold:
                self.skipped_groups.append({
                    'group': target,
                    'n_target': n_t,
                    'n_control': n_c,
                    'reason': f"Below threshold ({n_t},{n_c} < {threshold})"
                })
                continue

            X = embeddings[mask]
            y = (sub_df[intersectional_col] == target).astype(int).values
            groups = sub_df.get("scenario_id", sub_df.index).values

            # Safety check
            num_unique_scenarios = len(np.unique(groups))
            if num_unique_scenarios < 2 and cv_strategy == 'GroupKFold':
                self.skipped_groups.append({
                    'group': target,
                    'n_target': n_t,
                    'n_control': n_c,
                    'reason': f"Not enough unique scenarios for CV (found {num_unique_scenarios})"
                })
                logger.warning(f"Skipping '{target}': Only {num_unique_scenarios} unique scenario(s) found, need at least 2 for GroupKFold.")
                continue

            validator = ScenarioDisjointValidator(
                cv_strategy=cv_strategy,
                n_splits=min(n_splits, len(np.unique(groups))),
                classifier_type=classifier_type,
            )
            cv_results = validator.validate_by_scenario(X, y, groups)

            results.append({
                'intersectional_id': target,
                'control_id': control_label,
                'n_target': n_t,
                'n_control': n_c,
                'accuracy': cv_results['accuracy_mean'],
                'f1_score': cv_results['f1_macro_mean'],
            })

        df_results = pd.DataFrame(results)

        if self.skipped_groups:
            logger.warning(f"Skipped {len(self.skipped_groups)} groups due to insufficient data or missing control")
            for s in self.skipped_groups:
                logger.warning(f"  - {s}")

        logger.info(f"Computed {len(df_results)} paired evaluations out of {len(unique_groups)} groups")
        return df_results

    def measure_exaggeration(
        self,
        df: pd.DataFrame,
        emb_dict: Dict[str, np.ndarray],
        default_persona: str = "Unmarked",
        default_topic: str = "general_comment",
    ) -> pd.DataFrame:
        """
        Compute exaggeration scores for each intersectional cohort paired with its
        occupational counterfactual.  Seed words are extracted using Monroe
        log-odds between persona pole and topic pole, and exaggeration is the
        scalar projection described in the specification.
        """
        exag_scores = []
        unique_groups = df["intersectional_id"].unique()

        def occupation_of(label: str) -> str:
            return label.split("_")[-1] if isinstance(label, str) else ""

        for cohort_id in unique_groups:
            if cohort_id == default_persona or cohort_id.startswith(default_persona + "_" + default_persona):
                continue
            control = f"{default_persona}_{default_persona}_{occupation_of(cohort_id)}"
            if control not in unique_groups:
                continue

            sub_df = df[df["intersectional_id"].isin([cohort_id, control])]
            df_persona_pole = sub_df[(sub_df["intersectional_id"] == cohort_id) & (sub_df["topic"] == default_topic)]
            df_topic_pole = sub_df[sub_df["intersectional_id"] == control]
            df_target = sub_df[(sub_df["intersectional_id"] == cohort_id) & (sub_df["topic"] != default_topic)]
            df_background = sub_df

            if df_persona_pole.empty or df_topic_pole.empty or df_target.empty:
                continue

            persona_seed_words = get_seed_words(df_persona_pole, df_topic_pole, df_background)
            topic_seed_words = get_seed_words(df_topic_pole, df_persona_pole, df_background)

            if not persona_seed_words or not topic_seed_words:
                continue

            # compute poles
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

            def cos_sim(a, b):
                denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
                return np.dot(a, b) / denom

            target_sims = [cos_sim(emb_dict[s], axis_v) for s in df_target["sentences"].explode()]
            mean_target = np.mean(target_sims)
            default_p_sims = [cos_sim(emb_dict[s], axis_v) for s in df_persona_pole["sentences"].explode()]
            mean_dp = np.mean(default_p_sims)

            mean_t_pole = np.mean([cos_sim(emb_dict[s], axis_v) for s in df_topic_pole["sentences"].explode()])

            # add small epsilon to denominator to guard against near-zero values
            denominator = (mean_t_pole - mean_dp) + 1e-6
            exag_score = (mean_target - mean_dp) / denominator
            exag_scores.append({"intersectional_id": cohort_id, "exaggeration": exag_score})

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
        merged = imp_perf.merge(exp_perf, on='intersectional_id', suffixes=('_imp','_exp'))
        for metric in ['accuracy','f1_score']:
            merged[f'{metric}_delta'] = merged[f'{metric}_exp'] - merged[f'{metric}_imp']
        return merged

    def compute_intersectional_parity(
        self,
        performance_df: pd.DataFrame,
        metric: str = 'exaggeration_score'
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
        metric_name: str = 'exaggeration_score'
    ) -> str:
        """
        Generate a detailed report on intersectional evaluation results.
        
        Args:
            performance_df: DataFrame from evaluate_intersectional_groups
            parity_metrics: Dictionary from compute_intersectional_parity
            
        Returns:
            Formatted report string
        """
        report = """
================ INTERSECTIONAL BIAS EVALUATION REPORT ================

PERFORMANCE ACROSS INTERSECTIONAL GROUPS:
"""
        
        if not performance_df.empty:
            # Sort by the metric descending so the most exaggerated groups are at the top
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
                report += f"      The model exaggerates certain intersectional groups significantly more than others.\n"
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
        restricted to a single scenario (target_topic_id), using a restricted background
        corpus of a specific variant type (implicit or explicit).

        This method is designed to be called separately for implicit and explicit variants
        to avoid variant contamination. When target_topic_id is specified, only that
        scenario's data (plus the general_comment baseline) is used, ensuring the
        Monroe log-odds algorithm operates on semantically coherent task vocabularies.

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
        # Filter to the specified variant type first
        df_variant = df[df['variant_type'] == variant_type].copy()

        # Restrict to just the two identity groups
        sub_df = df_variant[df_variant["intersectional_id"].isin([target_id, control_id])].copy()

        # Further restrict to a specific scenario if provided (per-topic analysis)
        if target_topic_id is not None:
            sub_df = sub_df[
                (sub_df["scenario_id"] == target_topic_id) | (sub_df["topic"] == default_topic)
            ].copy()

        if "masked_text" in sub_df.columns:
            sub_df["response"] = sub_df["masked_text"]
        elif "target_text" in sub_df.columns:
            sub_df["response"] = sub_df["target_text"]
        else:
            raise ValueError("Dataframe must contain 'masked_text' or 'target_text'.")

        df_persona_pole = sub_df[(sub_df["intersectional_id"] == target_id) & (sub_df["topic"] == default_topic)]
        df_topic_pole = sub_df[(sub_df["intersectional_id"] == control_id) & (sub_df["topic"] == default_topic)]

        if df_persona_pole.empty or df_topic_pole.empty:
            # Insufficient data for this scenario/variant combination
            return np.zeros(768), 0.0, 0.0

        persona_seed_words = get_seed_words(df_persona_pole, df_topic_pole, sub_df)
        topic_seed_words = get_seed_words(df_topic_pole, df_persona_pole, sub_df)

        if not persona_seed_words or not topic_seed_words:
            # Monroe log-odds failed to find significant words
            return np.zeros(768), 0.0, 0.0

        def contains_any(text: str, words: list) -> bool:
            clean = re.sub(r"[^a-zA-Z\s]", "", text.lower())
            return any(re.search(rf"\b{re.escape(w)}\b", clean) for w in words)

        p_vecs = [emb for txt, emb in zip(df_persona_pole.get('masked_text', pd.Series()), df_persona_pole['embedding'])
                  if contains_any(txt, persona_seed_words)]
        t_vecs = [emb for txt, emb in zip(df_topic_pole.get('masked_text', pd.Series()), df_topic_pole['embedding'])
                  if contains_any(txt, topic_seed_words)]

        if p_vecs and t_vecs:
            p_pole = np.mean(p_vecs, axis=0)
            t_pole = np.mean(t_vecs, axis=0)
            axis_v = p_pole - t_pole
            norm = np.linalg.norm(axis_v)
            axis_v = axis_v / norm if norm > 1e-8 else np.zeros_like(axis_v)
        else:
            return np.zeros(768), 0.0, 0.0

        def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
            denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
            return float(np.dot(a, b) / denom)

        topic_pole_sim = np.mean([cos_sim(emb, axis_v) for emb in df_persona_pole['embedding']]) if not df_persona_pole.empty else 0.0
        persona_pole_sim = np.mean([cos_sim(emb, axis_v) for emb in df_topic_pole['embedding']]) if not df_topic_pole.empty else 0.0

        return axis_v, topic_pole_sim, persona_pole_sim