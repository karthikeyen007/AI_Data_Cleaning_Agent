"""
Metrics Engine Service

Comprehensive evaluation metrics for ML models:
- Classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Regression metrics (RMSE, MAE, R², MAPE)
- Confusion matrix and classification reports
- Metric visualization data
- Threshold optimization
"""

import logging
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from sklearn.metrics import (
    # Classification
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
    log_loss,
    
    # Regression
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score
)

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


@dataclass
class ClassificationMetrics:
    """Container for classification metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float]
    log_loss_value: Optional[float]
    confusion_matrix: List[List[int]]
    classification_report: Dict[str, Any]
    class_labels: List[str]
    support: Dict[str, int]


@dataclass
class RegressionMetrics:
    """Container for regression metrics."""
    rmse: float
    mae: float
    r2: float
    mape: Optional[float]
    explained_variance: float
    residuals_stats: Dict[str, float]


@dataclass
class ThresholdAnalysis:
    """Analysis of classification thresholds."""
    thresholds: List[float]
    precisions: List[float]
    recalls: List[float]
    f1_scores: List[float]
    optimal_threshold: float
    optimal_f1: float


class MetricsEngine:
    """
    Comprehensive metrics computation engine.
    
    Provides standardized evaluation for both classification
    and regression problems with detailed reporting.
    """
    
    def __init__(self):
        """Initialize the metrics engine."""
        self._history: List[Dict[str, Any]] = []
        logger.info("MetricsEngine initialized")
    
    def compute_classification_metrics(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        y_proba: Optional[np.ndarray] = None,
        labels: Optional[List[str]] = None,
        average: str = "weighted"
    ) -> ClassificationMetrics:
        """
        Compute comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            labels: Class labels
            average: Averaging method for multiclass
            
        Returns:
            ClassificationMetrics with all computed metrics
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Infer labels if not provided
        if labels is None:
            labels = sorted(list(set(y_true) | set(y_pred)))
        
        n_classes = len(labels)
        is_binary = n_classes == 2
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=average, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # ROC-AUC (requires probabilities)
        roc_auc = None
        log_loss_val = None
        
        if y_proba is not None:
            try:
                if is_binary:
                    # Binary classification
                    if y_proba.ndim > 1:
                        proba = y_proba[:, 1]
                    else:
                        proba = y_proba
                    roc_auc = roc_auc_score(y_true, proba)
                else:
                    # Multiclass
                    roc_auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average=average)
                
                log_loss_val = log_loss(y_true, y_proba)
            except Exception as e:
                logger.warning(f"Could not compute ROC-AUC: {e}")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # Support per class
        support = {}
        for i, label in enumerate(labels):
            support[str(label)] = int((y_true == label).sum())
        
        metrics = ClassificationMetrics(
            accuracy=round(accuracy, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            roc_auc=round(roc_auc, 4) if roc_auc else None,
            log_loss_value=round(log_loss_val, 4) if log_loss_val else None,
            confusion_matrix=cm.tolist(),
            classification_report=report,
            class_labels=[str(l) for l in labels],
            support=support
        )
        
        # Store in history
        self._history.append({
            "type": "classification",
            "metrics": metrics.__dict__.copy()
        })
        
        logger.info(f"Classification metrics: Accuracy={accuracy:.4f}, F1={f1:.4f}")
        
        return metrics
    
    def compute_regression_metrics(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series]
    ) -> RegressionMetrics:
        """
        Compute comprehensive regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            RegressionMetrics with all computed metrics
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Core metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        explained_var = explained_variance_score(y_true, y_pred)
        
        # MAPE (handle zeros)
        mape = None
        try:
            if not np.any(y_true == 0):
                mape = mean_absolute_percentage_error(y_true, y_pred)
        except Exception:
            pass
        
        # Residuals analysis
        residuals = y_true - y_pred
        residuals_stats = {
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals)),
            "min": float(np.min(residuals)),
            "max": float(np.max(residuals)),
            "median": float(np.median(residuals)),
            "skewness": float(pd.Series(residuals).skew()),
        }
        
        metrics = RegressionMetrics(
            rmse=round(rmse, 4),
            mae=round(mae, 4),
            r2=round(r2, 4),
            mape=round(mape, 4) if mape else None,
            explained_variance=round(explained_var, 4),
            residuals_stats=residuals_stats
        )
        
        # Store in history
        self._history.append({
            "type": "regression",
            "metrics": metrics.__dict__.copy()
        })
        
        logger.info(f"Regression metrics: RMSE={rmse:.4f}, R²={r2:.4f}")
        
        return metrics
    
    def compute_threshold_analysis(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_proba: np.ndarray,
        n_thresholds: int = 100
    ) -> ThresholdAnalysis:
        """
        Analyze different classification thresholds.
        
        Args:
            y_true: True binary labels
            y_proba: Probabilities for positive class
            n_thresholds: Number of thresholds to evaluate
            
        Returns:
            ThresholdAnalysis with optimal threshold
        """
        y_true = np.array(y_true)
        
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1]
        
        # Get precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
        
        # Compute F1 for each threshold
        f1_scores = []
        for p, r in zip(precisions[:-1], recalls[:-1]):
            if p + r > 0:
                f1_scores.append(2 * p * r / (p + r))
            else:
                f1_scores.append(0)
        
        # Find optimal threshold
        best_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[best_idx]
        optimal_f1 = f1_scores[best_idx]
        
        # Sample thresholds for output
        step = max(1, len(thresholds) // n_thresholds)
        
        return ThresholdAnalysis(
            thresholds=thresholds[::step].tolist(),
            precisions=precisions[:-1:step].tolist(),
            recalls=recalls[:-1:step].tolist(),
            f1_scores=f1_scores[::step],
            optimal_threshold=round(float(optimal_threshold), 4),
            optimal_f1=round(float(optimal_f1), 4)
        )
    
    def get_roc_curve_data(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_proba: np.ndarray
    ) -> Dict[str, List[float]]:
        """
        Get ROC curve data for visualization.
        
        Args:
            y_true: True binary labels
            y_proba: Probabilities for positive class
            
        Returns:
            Dictionary with fpr, tpr, and thresholds
        """
        y_true = np.array(y_true)
        
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        
        return {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "auc": round(roc_auc_score(y_true, y_proba), 4)
        }
    
    def compare_models(
        self,
        results: List[Dict[str, Any]],
        metric_key: str = "f1"
    ) -> List[Dict[str, Any]]:
        """
        Compare multiple model results.
        
        Args:
            results: List of dicts with 'name' and 'metrics' keys
            metric_key: Metric to use for ranking
            
        Returns:
            Sorted list of model comparisons
        """
        comparisons = []
        
        for result in results:
            name = result.get("name", "Unknown")
            metrics = result.get("metrics", {})
            
            score = 0
            if hasattr(metrics, metric_key):
                score = getattr(metrics, metric_key)
            elif isinstance(metrics, dict):
                score = metrics.get(metric_key, 0)
            
            comparisons.append({
                "name": name,
                "score": score,
                "metrics": metrics.__dict__ if hasattr(metrics, "__dict__") else metrics
            })
        
        # Sort by score descending
        comparisons.sort(key=lambda x: x["score"], reverse=True)
        
        # Add ranks
        for i, comp in enumerate(comparisons):
            comp["rank"] = i + 1
        
        return comparisons
    
    def get_metrics_summary(
        self,
        classification_metrics: Optional[ClassificationMetrics] = None,
        regression_metrics: Optional[RegressionMetrics] = None
    ) -> Dict[str, Any]:
        """
        Get a summary of metrics for display.
        
        Args:
            classification_metrics: Classification metrics
            regression_metrics: Regression metrics
            
        Returns:
            Dictionary summary for UI display
        """
        summary = {"type": None, "primary_metric": None, "all_metrics": {}}
        
        if classification_metrics:
            summary["type"] = "classification"
            summary["primary_metric"] = {
                "name": "F1 Score",
                "value": classification_metrics.f1
            }
            summary["all_metrics"] = {
                "Accuracy": classification_metrics.accuracy,
                "Precision": classification_metrics.precision,
                "Recall": classification_metrics.recall,
                "F1 Score": classification_metrics.f1,
                "ROC-AUC": classification_metrics.roc_auc
            }
            
        elif regression_metrics:
            summary["type"] = "regression"
            summary["primary_metric"] = {
                "name": "R² Score",
                "value": regression_metrics.r2
            }
            summary["all_metrics"] = {
                "RMSE": regression_metrics.rmse,
                "MAE": regression_metrics.mae,
                "R² Score": regression_metrics.r2,
                "MAPE": regression_metrics.mape,
                "Explained Variance": regression_metrics.explained_variance
            }
        
        return summary
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get metrics computation history."""
        return self._history
    
    def clear_history(self) -> None:
        """Clear metrics history."""
        self._history = []


# Convenience functions
def quick_classification_eval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Quick classification evaluation.
    
    Returns:
        Dictionary of key metrics
    """
    engine = MetricsEngine()
    metrics = engine.compute_classification_metrics(y_true, y_pred, y_proba)
    
    return {
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "roc_auc": metrics.roc_auc
    }


def quick_regression_eval(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Quick regression evaluation.
    
    Returns:
        Dictionary of key metrics
    """
    engine = MetricsEngine()
    metrics = engine.compute_regression_metrics(y_true, y_pred)
    
    return {
        "rmse": metrics.rmse,
        "mae": metrics.mae,
        "r2": metrics.r2,
        "mape": metrics.mape
    }
