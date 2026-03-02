"""
Scenario-Disjoint Cross-Validation Module
Implements GroupKFold and LeaveOneGroupOut cross-validation to prevent data leakage.

The key insight: train the classifier on personas navigating Scenarios A, B, C
and test exclusively on unseen Scenario D. If accuracy drops to 50% (random chance),
the model was memorizing prompts rather than learning stylistic patterns.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold, 
    LeaveOneGroupOut, 
    cross_val_score, 
    cross_val_predict,
    cross_validate
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from typing import Dict, List, Tuple, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScenarioDisjointValidator:
    """
    Implements scenario-disjoint cross-validation to eliminate data leakage.
    Groups data by scenario_id so each fold trains on unseen scenarios.
    """
    
    def __init__(self, 
                 cv_strategy: str = "GroupKFold",
                 n_splits: int = 5,
                 classifier_type: str = "RandomForest"):
        """
        Initialize scenario-disjoint validator.
        
        Args:
            cv_strategy: "GroupKFold" or "LeaveOneGroupOut"
            n_splits: Number of splits for GroupKFold (ignored for LOGO)
            classifier_type: "RandomForest", "GradientBoosting", or "LogisticRegression"
        """
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.classifier_type = classifier_type
        
        # Initialize the CV splitter
        if cv_strategy == "GroupKFold":
            self.cv = GroupKFold(n_splits=n_splits)
        elif cv_strategy == "LeaveOneGroupOut":
            self.cv = LeaveOneGroupOut()
        else:
            raise ValueError(f"Unknown CV strategy: {cv_strategy}")
        
        # Initialize classifier
        self.classifier = self._get_classifier()
        
        # Store results
        self.cv_results = {}
        self.fold_results = []
    
    def _get_classifier(self) -> Any:
        """Get classifier instance based on type."""
        if self.classifier_type == "RandomForest":
            return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif self.classifier_type == "GradientBoosting":
            return GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif self.classifier_type == "LogisticRegression":
            return LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
    
    def validate_by_scenario(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        scenario_ids: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Perform scenario-disjoint cross-validation.
        
        Args:
            X: Feature matrix (embeddings)
            y: Target labels
            groups: Group IDs (scenario_id) for GroupKFold
            scenario_ids: Optional scenario IDs for per-scenario reporting
            
        Returns:
            Dictionary with cross-validation results including accuracy, F1, and per-fold metrics
        """
        # Validate inputs
        if len(X) != len(y) or len(X) != len(groups):
            raise ValueError("X, y, and groups must have the same length")
        
        logger.info(f"Starting {self.cv_strategy} cross-validation with {self.classifier_type}")
        logger.info(f"Total samples: {len(X)}, Unique groups: {len(np.unique(groups))}")
        
        # Run cross-validation with multiple metrics
        scoring = {
            'accuracy': 'accuracy',
            'f1_macro': 'f1_macro',
            'f1_weighted': 'f1_weighted'
        }
        
        cv_results = cross_validate(
            self.classifier,
            X, y, 
            groups=groups,
            cv=self.cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        # Get predictions for per-fold analysis
        y_pred = cross_val_predict(self.classifier, X, y, groups=groups, cv=self.cv)
        
        # Store results
        self.cv_results = {
            'accuracy_mean': np.mean(cv_results['test_accuracy']),
            'accuracy_std': np.std(cv_results['test_accuracy']),
            'f1_macro_mean': np.mean(cv_results['test_f1_macro']),
            'f1_macro_std': np.std(cv_results['test_f1_macro']),
            'f1_weighted_mean': np.mean(cv_results['test_f1_weighted']),
            'f1_weighted_std': np.std(cv_results['test_f1_weighted']),
            'train_accuracy_mean': np.mean(cv_results['train_accuracy']),
            'raw_cv_results': cv_results
        }
        
        # Detailed per-fold results
        n_folds = len(cv_results['test_accuracy'])
        self.fold_results = []
        
        for fold_idx in range(n_folds):
            fold_info = {
                'fold': fold_idx,
                'train_accuracy': cv_results['train_accuracy'][fold_idx],
                'test_accuracy': cv_results['test_accuracy'][fold_idx],
                'test_f1_macro': cv_results['test_f1_macro'][fold_idx],
                'test_f1_weighted': cv_results['test_f1_weighted'][fold_idx]
            }
            self.fold_results.append(fold_info)
            logger.info(f"Fold {fold_idx}: train_acc={fold_info['train_accuracy']:.3f}, "
                       f"test_acc={fold_info['test_accuracy']:.3f}, "
                       f"f1_macro={fold_info['test_f1_macro']:.3f}")
        
        # Check for overfitting
        train_acc = self.cv_results['train_accuracy_mean']
        test_acc = self.cv_results['accuracy_mean']
        if train_acc - test_acc > 0.15:
            logger.warning(f"Large train-test gap detected: {train_acc:.3f} vs {test_acc:.3f}")
            logger.warning("This may indicate overfitting. Consider regularization.")
        
        # Alert if performance is near random chance
        n_classes = len(np.unique(y))
        random_baseline = 1.0 / n_classes
        if test_acc < random_baseline + 0.05:
            logger.warning(f"Test accuracy ({test_acc:.3f}) near random baseline ({random_baseline:.3f})")
            logger.warning("Model may be unable to distinguish patterns beyond memorization.")
        
        return self.cv_results
    
    def per_scenario_performance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        scenario_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Evaluate per-scenario held-out performance (if using GroupKFold).
        Returns accuracy and F1 for each unique scenario used as a test set.
        
        Args:
            X: Feature matrix
            y: Target labels
            groups: Scenario group IDs
            scenario_df: Optional DataFrame with scenario metadata
            
        Returns:
            DataFrame with per-scenario performance metrics
        """
        results = []
        unique_scenarios = np.unique(groups)
        
        for test_scenario in unique_scenarios:
            # Create train-test split based on scenario
            test_mask = groups == test_scenario
            train_mask = ~test_mask
            
            if np.sum(train_mask) == 0 or np.sum(test_mask) == 0:
                continue
            
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            
            # Train classifier on other scenarios
            clf = self._get_classifier()
            clf.fit(X_train, y_train)
            
            # Evaluate on held-out scenario
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
            f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            results.append({
                'scenario_id': test_scenario,
                'test_samples': np.sum(test_mask),
                'train_samples': np.sum(train_mask),
                'accuracy': acc,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted
            })
            
            logger.info(f"Scenario {test_scenario}: "
                       f"test_acc={acc:.3f}, f1_macro={f1_macro:.3f}")
        
        return pd.DataFrame(results)
    
    def get_summary_report(self) -> str:
        """Generate a human-readable summary of cross-validation results."""
        if not self.cv_results:
            return "No cross-validation results available. Run validate_by_scenario first."
        
        report = f"""
================== SCENARIO-DISJOINT CROSS-VALIDATION REPORT ==================
CV Strategy: {self.cv_strategy}
Classifier: {self.classifier_type}
Number of Folds: {len(self.fold_results)}

OVERALL RESULTS:
  Test Accuracy:    {self.cv_results['accuracy_mean']:.4f} ± {self.cv_results['accuracy_std']:.4f}
  Test F1 (Macro):  {self.cv_results['f1_macro_mean']:.4f} ± {self.cv_results['f1_macro_std']:.4f}
  Test F1 (Weighted): {self.cv_results['f1_weighted_mean']:.4f} ± {self.cv_results['f1_weighted_std']:.4f}
  Train Accuracy:   {self.cv_results['train_accuracy_mean']:.4f}

PER-FOLD BREAKDOWN:
"""
        for fold in self.fold_results:
            report += f"  Fold {fold['fold']}: test_acc={fold['test_accuracy']:.4f}, f1_macro={fold['test_f1_macro']:.4f}\n"
        
        report += "\n" + "=" * 80
        return report
    
    def plot_fold_performance(self):
        """Generate plot data for fold-by-fold performance visualization."""
        if not self.fold_results:
            logger.warning("No fold results to plot")
            return None
        
        folds = [f['fold'] for f in self.fold_results]
        test_accs = [f['test_accuracy'] for f in self.fold_results]
        
        return {
            'folds': folds,
            'test_accuracies': test_accs,
            'mean_accuracy': self.cv_results['accuracy_mean'],
            'std_accuracy': self.cv_results['accuracy_std']
        }
