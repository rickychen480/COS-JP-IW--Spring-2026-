"""
Intersectional Joint Probability Evaluation Module
Treats multidimensional identities as indivisible constructs to capture
their unique linguistic realities. Evaluates intersectional tuples using 
round-robin pairwise comparisons and high-dimensional clustering metrics
(Mahalanobis Distance) to calculate relative semantic distance.
"""

import numpy as np
import pandas as pd
import itertools
import sys
import os
from typing import Dict, List, Tuple, Optional
from sklearn.covariance import LedoitWolf

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from compost.scenario_disjoint_cv import ScenarioDisjointValidator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntersectionalEvaluator:
    """
    Evaluates bias through the lens of intersectional identities. 
    Uses Round-Robin pairwise comparisons and High-Dimensional distances.
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

    def _get_valid_pairs(self, unique_groups: np.ndarray, directed: bool = False) -> List[Tuple[str, str]]:
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
        
        # Get UNDIRECTED pairwise comparisons
        pairs = self._get_valid_pairs(unique_groups, directed=False)
        logger.info(f"Evaluating {len(pairs)} pairwise intersectional combinations (min_group_size={threshold})")

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
            y = (sub_df[intersectional_col] == target).astype(int).values
            groups = sub_df.get("scenario_id", sub_df.index).values

            num_unique_scenarios = len(np.unique(groups))
            if num_unique_scenarios < 2 and cv_strategy == 'GroupKFold':
                self.skipped_groups.append({
                    'target': target,
                    'control': control_label,
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
            logger.warning(f"Skipped {len(self.skipped_groups)} pairwise comparisons due to insufficient data")

        logger.info(f"Computed {len(df_results)} pairwise evaluations out of {len(pairs)} possible pairs")
        return df_results

    def measure_exaggeration(
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