"""
Intersectional Joint Probability Evaluation Module
Treats multidimensional identities as indivisible constructs to capture
their unique linguistic realities. Evaluates intersectional tuples (e.g.,
Hispanic_Male_Nurse) against an "Unmarked" baseline using log-odds calculations.
"""

import numpy as np
import pandas as pd
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntersectionalEvaluator:
    """
    Evaluates bias through the lens of intersectional identities rather than
    isolated demographic dimensions.
    """
    
    def __init__(self, baseline_label: str = "Unmarked"):
        """
        Initialize intersectional evaluator.
        
        Args:
            baseline_label: The "unmarked" baseline identity to compare against
        """
        self.baseline_label = baseline_label
        self.intersectional_results = {}
        self.log_odds_cache = {}
    
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
        parts = [demographic, gender, occupation]
        # Filter out "Unmarked" elements for cleaner labeling
        parts = [p for p in parts if p and p != "Unmarked"]
        
        if not parts:
            return self.baseline_label
        
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
    
    def calculate_intersectional_log_odds(
        self,
        target_embeddings: np.ndarray,
        baseline_embeddings: np.ndarray,
        target_labels: List[str],
        baseline_labels: List[str],
        background_embeddings: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate log-odds between a specific intersectional group and unmarked baseline.
        Uses Monroe et al.'s Fightin' Words method adapted for embeddings.
        
        Args:
            target_embeddings: Embeddings for target intersectional group
            baseline_embeddings: Embeddings for unmarked baseline
            target_labels: Text labels/sentences for target group
            baseline_labels: Text labels/sentences for baseline
            background_embeddings: Optional background for prior
            
        Returns:
            Dictionary mapping embedding dimensions to log-odds ratios
        """
        if len(target_embeddings) == 0 or len(baseline_embeddings) == 0:
            return {}
        
        # Compute embedding space statistics
        target_mean = np.mean(target_embeddings, axis=0)
        baseline_mean = np.mean(baseline_embeddings, axis=0)
        target_var = np.var(target_embeddings, axis=0)
        baseline_var = np.var(baseline_embeddings, axis=0)
        
        # Per-dimension log-odds (z-score style)
        log_odds = {}
        n_dims = target_embeddings.shape[1]
        
        for dim in range(n_dims):
            t_mean, t_var = target_mean[dim], target_var[dim] + 1e-6  # Add small epsilon
            b_mean, b_var = baseline_mean[dim], baseline_var[dim] + 1e-6
            
            # Effect size: Cohen's d
            pooled_std = np.sqrt((t_var + b_var) / 2.0)
            cohens_d = (t_mean - b_mean) / (pooled_std + 1e-6)
            
            log_odds[dim] = cohens_d
        
        return log_odds
    def get_most_distinctive_dimensions(
        self,
        log_odds: Dict[int, float],
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Extract the most distinctive embedding dimensions for an intersectional group.
        
        Args:
            log_odds: Dictionary of dimension -> log-odds
            top_k: Number of top dimensions to return
            
        Returns:
            List of (dimension_index, log_odds_value) tuples, sorted by distinctiveness
        """
        sorted_dims = sorted(log_odds.items(), key=lambda x: abs(x[1]), reverse=True)
        return sorted_dims[:top_k]
    
    def evaluate_intersectional_groups(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        embeddings: np.ndarray,
        intersectional_col: str = 'intersectional_id'
    ) -> pd.DataFrame:
        """
        Evaluate classification performance per intersectional group.
        
        Args:
            df: DataFrame with intersectional identity and other attributes
            predictions: Model predictions
            ground_truth: True labels
            embeddings: Feature embeddings
            intersectional_col: Column name for intersectional identity
            
        Returns:
            DataFrame with per-group performance metrics
        """
        results = []
        
        unique_groups = df[intersectional_col].unique()
        
        for group in unique_groups:
            mask = df[intersectional_col] == group
            
            if mask.sum() < 2:  # Skip groups with too few samples
                continue
            
            group_preds = predictions[mask]
            group_truth = ground_truth[mask]
            group_embs = embeddings[mask]
            
            # Compute metrics
            accuracy = accuracy_score(group_truth, group_preds)
            precision = precision_score(group_truth, group_preds, average='weighted', zero_division=0)
            recall = recall_score(group_truth, group_preds, average='weighted', zero_division=0)
            f1 = f1_score(group_truth, group_preds, average='weighted', zero_division=0)
            
            # Compute group embedding statistics
            embedding_mean = np.mean(group_embs, axis=0)
            embedding_std = np.std(group_embs, axis=0)
            embedding_dist_to_origin = np.linalg.norm(embedding_mean)
            
            results.append({
                'intersectional_id': group,
                'n_samples': mask.sum(),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'embedding_mean_norm': embedding_dist_to_origin,
                'embedding_std_mean': np.mean(embedding_std)
            })
        
        df_results = pd.DataFrame(results)
        
        # Log group disparities
        if not df_results.empty:
            acc_range = df_results['accuracy'].max() - df_results['accuracy'].min()
            if acc_range > 0.1:
                logger.warning(f"Large accuracy disparity across groups: {acc_range:.3f}")
                worst_group = df_results.loc[df_results['accuracy'].idxmin()]
                logger.warning(f"  Worst: {worst_group['intersectional_id']} "
                              f"({worst_group['accuracy']:.3f})")
        
        return df_results
    
    def compute_intersectional_parity(
        self,
        performance_df: pd.DataFrame,
        metric: str = 'f1_score'
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
            raise ValueError(f"Metric '{metric}' not found in results")
        
        values = performance_df[metric].dropna()
        
        if len(values) == 0:
            return {}
        
        min_val = values.min()
        max_val = values.max()
        mean_val = values.mean()
        std_val = values.std()
        
        # Disparate impact ratio (80% rule)
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
    
    def compare_vs_baseline(
        self,
        target_embeddings: np.ndarray,
        baseline_embeddings: np.ndarray,
        target_predictions: np.ndarray,
        baseline_predictions: np.ndarray,
        target_truth: np.ndarray,
        baseline_truth: np.ndarray,
        intersectional_label: str
    ) -> Dict[str, float]:
        """
        Directly compare metrics for a specific intersectional group vs. unmarked baseline.
        
        Args:
            target_embeddings: Embeddings for target intersectional group
            baseline_embeddings: Embeddings for unmarked baseline
            target_predictions: Predictions for target group
            baseline_predictions: Predictions for baseline
            target_truth: Ground truth for target group
            baseline_truth: Ground truth for baseline
            intersectional_label: Label for the target group (e.g., "Hispanic_Male_Nurse")
            
        Returns:
            Dictionary with comparative metrics
        """
        results = {
            'intersectional_id': intersectional_label,
            'target_n': len(target_predictions),
            'baseline_n': len(baseline_predictions)
        }
        
        # Performance comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        for metric_name in metrics:
            if metric_name == 'accuracy':
                target_val = accuracy_score(target_truth, target_predictions)
                baseline_val = accuracy_score(baseline_truth, baseline_predictions)
            else:
                metric_fn = {'precision': precision_score, 'recall': recall_score, 'f1': f1_score}[metric_name]
                target_val = metric_fn(target_truth, target_predictions, average='weighted', zero_division=0)
                baseline_val = metric_fn(baseline_truth, baseline_predictions, average='weighted', zero_division=0)
            
            results[f'target_{metric_name}'] = target_val
            results[f'baseline_{metric_name}'] = baseline_val
            results[f'{metric_name}_gap'] = baseline_val - target_val  # Negative = underprivileged group
        
        # Embedding space analysis
        target_norm = np.linalg.norm(np.mean(target_embeddings, axis=0))
        baseline_norm = np.linalg.norm(np.mean(baseline_embeddings, axis=0))
        results['target_embedding_norm'] = target_norm
        results['baseline_embedding_norm'] = baseline_norm
        results['embedding_distance_ratio'] = target_norm / (baseline_norm + 1e-6)
        
        return results
    
    def generate_intersectional_report(
        self,
        performance_df: pd.DataFrame,
        parity_metrics: Dict[str, float]
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
            report += performance_df.to_string(index=False)
        
        report += """

STATISTICAL PARITY ANALYSIS:
"""
        for key, val in parity_metrics.items():
            if isinstance(val, float):
                report += f"  {key}: {val:.4f}\n"
        
        report += "\nINTERPRETATION:\n"
        
        if 'accuracy_disparate_impact_ratio' in parity_metrics:
            di = parity_metrics['accuracy_disparate_impact_ratio']
            if di < 0.80:
                report += f"  ⚠️  Strong disparate impact detected (ratio: {di:.3f})\n"
            elif di < 0.90:
                report += f"  ⚠️  Moderate disparate impact (ratio: {di:.3f})\n"
            else:
                report += f"  ✓ Acceptable parity (80% rule satisfied, ratio: {di:.3f})\n"
        
        report += "\n" + "=" * 80
        return report
