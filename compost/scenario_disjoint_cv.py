"""
Scenario-Disjoint Cross-Validation Module
Implements both:
1. Scenario-disjoint cross-validation (GroupKFold, LeaveOneGroupOut) to prevent data leakage
2. Scenario-disjoint grouped 80/20 split for individuation measurement

For individuation (CoMPosT bias audit), uses a scenario-disjoint 80/20 split to measure if S_{p,t,c}
is differentiable from S_{_,t,c} based on test set accuracy alone.

For cross-validation, trains the classifier on personas navigating Scenarios A, B, C
and tests exclusively on unseen Scenario D. If accuracy drops to 50% (random chance),
the model was memorizing prompts rather than learning stylistic patterns.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold, 
    GroupShuffleSplit,
    StratifiedGroupKFold,
    LeaveOneGroupOut,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from typing import Dict, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScenarioDisjointValidator:
    """
    Implements scenario-disjoint cross-validation to eliminate data leakage.
    Groups data by scenario_id so each fold trains on unseen scenarios.
    """
    
    def __init__(self, 
                 cv_strategy: str = "StratifiedGroupKFold",
                 n_splits: int = 5,
                 classifier_type: str = "XGBoost"):
        """
        Initialize scenario-disjoint validator.
        
        Args:
            cv_strategy: "GroupKFold" or "LeaveOneGroupOut"
            n_splits: Number of splits for GroupKFold (ignored for LOGO)
            classifier_type: "XGBoost", "LinearSVC", "RandomForest", "GradientBoosting", or "LogisticRegression"
        """
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.classifier_type = classifier_type
        
        # Initialize the CV splitter
        if cv_strategy == "StratifiedGroupKFold":
            self.cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        elif cv_strategy == "GroupKFold":
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

    def _fit_classifier(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Any:
        """
        Fit classifier on train split.
        For XGBoost, compute split-specific scale_pos_weight to handle class imbalance.
        """
        labels, counts = np.unique(y_train, return_counts=True)
        if len(labels) < 2:
            clf = self._get_classifier()
            clf.fit(X_train, y_train)
            return clf

        scale_pos_weight = None
        if self.classifier_type == "XGBoost":
            n_neg = int(np.sum(y_train == 0))
            n_pos = int(np.sum(y_train == 1))
            if n_pos > 0 and n_neg > 0:
                scale_pos_weight = float(n_neg) / float(n_pos)
                logger.info("XGBoost scale_pos_weight for split: %.3f", scale_pos_weight)

        clf = self._get_classifier(scale_pos_weight=scale_pos_weight)
        clf.fit(X_train, y_train)

        return clf
    
    def _get_classifier(self, scale_pos_weight: Optional[float] = None) -> Any:
        """Get classifier instance based on type."""
        if self.classifier_type == "XGBoost":
            return XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight if scale_pos_weight is not None else 1.0,
                random_state=42,
                n_jobs=-1,
                tree_method="hist",
            )

        elif self.classifier_type == "RandomForest":
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=5,
                min_samples_leaf=10,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        
        elif self.classifier_type == "GradientBoosting":
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                min_samples_leaf=4,
                max_features="sqrt",
                random_state=42
            )
        elif self.classifier_type == "LogisticRegression":
            return LogisticRegression(
                C=0.5, 
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=42,
                n_jobs=-1,
            )
        elif self.classifier_type == "LinearSVC":
            return LinearSVC(
                C=0.5,
                class_weight="balanced",
                dual=False, # dual=False is preferred when n_samples > n_features is not guaranteed
                max_iter=2000,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
    
    def validate_grouped_holdout(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Run one scenario-disjoint grouped holdout split (default 80/20)."""
        if len(X) != len(y) or len(X) != len(groups):
            raise ValueError("X, y, and groups must have the same length")

        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        if len(unique_groups) < 2:
            raise ValueError(
                "Cannot perform group-disjoint holdout with fewer than 2 unique groups."
            )

        logger.info(
            f"Starting grouped holdout ({1.0 - test_size:.0%}/{test_size:.0%}) with "
            f"{self.classifier_type}; samples={len(X)}, unique_groups={len(unique_groups)}"
        )

        split_found = False
        for offset in range(20):
            gss = GroupShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=random_state + offset,
            )
            train_idx, test_idx = next(gss.split(X, y, groups=groups))
            y_train = y[train_idx]
            y_test = y[test_idx]
            if len(np.unique(y_train)) > 1 and len(np.unique(y_test)) > 1:
                split_found = True
                break

        if not split_found:
            raise ValueError(
                "Unable to find a group-disjoint split with both classes in train and test. "
                "Try more data, different group balance, or scenario-disjoint CV."
            )

        X_train, X_test = X[train_idx], X[test_idx]
        clf = self._fit_classifier(X_train, y_train)
        y_pred = clf.predict(X_test)

        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        test_f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        train_accuracy = accuracy_score(y_train, clf.predict(X_train))

        self.cv_results = {
            'accuracy_mean': test_accuracy,
            'accuracy_std': 0.0,
            'f1_macro_mean': test_f1_macro,
            'f1_macro_std': 0.0,
            'f1_weighted_mean': test_f1_weighted,
            'f1_weighted_std': 0.0,
            'train_accuracy_mean': train_accuracy,
            'raw_cv_results': None,
            'evaluation_mode': 'grouped_holdout',
            'n_train': len(X_train),
            'n_test': len(X_test),
        }
        self.fold_results = [
            {
                'fold': 0,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'test_f1_macro': test_f1_macro,
                'test_f1_weighted': test_f1_weighted,
            }
        ]
        return self.cv_results

    def validate_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Run scenario-disjoint cross-validation (GroupKFold or LeaveOneGroupOut).
        
        Args:
            X: Feature matrix (embeddings)
            y: Target labels
            groups: Group IDs (scenario_id) for GroupKFold
            
        Returns:
            Dictionary with evaluation results including accuracy and F1
        """
        if len(X) != len(y) or len(X) != len(groups):
            raise ValueError("X, y, and groups must have the same length")

        logger.info(f"Starting {self.cv_strategy} cross-validation with {self.classifier_type}")
        logger.info(f"Total samples: {len(X)}, Unique groups: {len(np.unique(groups))}")
        
        if self.cv_strategy in {"GroupKFold", "StratifiedGroupKFold"}:
            unique_groups = np.unique(groups)
            n_groups = len(unique_groups)
            if n_groups < 2:
                raise ValueError(
                    "Cannot perform grouped CV with fewer than 2 unique groups."
                )
            if n_groups < self.n_splits:
                logger.warning(
                    f"Reducing n_splits from {self.n_splits} to {n_groups} "
                    "because only {n_groups} unique groups were provided."
                )
            actual_splits = min(self.n_splits, n_groups)
            if self.cv_strategy == "StratifiedGroupKFold":
                cv = StratifiedGroupKFold(n_splits=actual_splits, shuffle=True, random_state=42)
            else:
                cv = GroupKFold(n_splits=actual_splits)
        else:
            cv = self.cv

        # Manual CV loop keeps scenario-disjoint behavior and applies split-specific fitting.
        test_accuracy_scores = []
        test_f1_macro_scores = []
        test_f1_weighted_scores = []
        train_accuracy_scores = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                logger.warning(
                    "Skipping fold %d because train/test contains a single class.",
                    fold_idx,
                )
                continue

            clf = self._fit_classifier(X_train, y_train)

            y_pred_test = clf.predict(X_test)
            y_pred_train = clf.predict(X_train)

            test_acc = accuracy_score(y_test, y_pred_test)
            test_f1_macro = f1_score(y_test, y_pred_test, average='macro', zero_division=0)
            test_f1_weighted = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
            train_acc = accuracy_score(y_train, y_pred_train)

            test_accuracy_scores.append(test_acc)
            test_f1_macro_scores.append(test_f1_macro)
            test_f1_weighted_scores.append(test_f1_weighted)
            train_accuracy_scores.append(train_acc)

            fold_info = {
                'fold': fold_idx,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'test_f1_macro': test_f1_macro,
                'test_f1_weighted': test_f1_weighted,
            }
            self.fold_results.append(fold_info)
            logger.info(
                "Fold %d: train_acc=%.3f, test_acc=%.3f, f1_macro=%.3f",
                fold_idx,
                train_acc,
                test_acc,
                test_f1_macro,
            )

        if not test_accuracy_scores:
            raise ValueError(
                "No valid CV folds were produced. Check class balance and group splits."
            )

        cv_results = {
            'test_accuracy': np.array(test_accuracy_scores),
            'test_f1_macro': np.array(test_f1_macro_scores),
            'test_f1_weighted': np.array(test_f1_weighted_scores),
            'train_accuracy': np.array(train_accuracy_scores),
        }
        
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


    ##### DEBUG FUNCTIONS #####
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

            if hasattr(clf, 'best_params_'):
                print("Best parameters found: ", clf.best_params_)
                print("Best cross-validation score: ", clf.best_score_)
            
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
            return "No cross-validation results available. Run validation first."
        
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
