"""
Explainability Service - Model Interpretation & Insights

Production-grade ML explainability:
- SHAP values (global and local)
- Feature importance extraction
- Coefficient interpretation (linear models)
- Partial dependence plots data
- Decision path analysis
"""

import logging
from enum import Enum
from typing import Optional, Dict, Any, List, Union, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Optional SHAP import
SHAP_AVAILABLE = False
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    logger.warning("SHAP not installed. Install with: pip install shap")


class ExplainerType(Enum):
    """Types of explainers."""
    SHAP_TREE = "shap_tree"
    SHAP_KERNEL = "shap_kernel"
    SHAP_LINEAR = "shap_linear"
    PERMUTATION = "permutation"
    BUILTIN = "builtin"


@dataclass
class FeatureImportance:
    """Feature importance result."""
    feature: str
    importance: float
    std: Optional[float] = None
    rank: int = 0


@dataclass 
class LocalExplanation:
    """Explanation for a single prediction."""
    prediction: Any
    probability: Optional[float] = None
    base_value: float = 0.0
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    top_positive: List[Tuple[str, float]] = field(default_factory=list)
    top_negative: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class GlobalExplanation:
    """Global model explanation."""
    explainer_type: str
    feature_importances: List[FeatureImportance]
    shap_values: Optional[np.ndarray] = None
    expected_value: Optional[float] = None
    interaction_effects: Optional[Dict[str, Dict[str, float]]] = None


class ExplainabilityService:
    """
    Production-grade ML model explainability.
    
    Provides:
    - Feature importance (multiple methods)
    - SHAP explanations (when available)
    - Local predictions explanations
    - Coefficient interpretation
    """
    
    def __init__(self, model: BaseEstimator = None):
        """
        Initialize explainability service.
        
        Args:
            model: Trained scikit-learn compatible model
        """
        self.model = model
        self._shap_explainer = None
        self._expected_value = None
        
        logger.info(f"ExplainabilityService initialized (SHAP available: {SHAP_AVAILABLE})")
    
    def set_model(self, model: BaseEstimator) -> None:
        """Set or update the model to explain."""
        self.model = model
        self._shap_explainer = None
        self._expected_value = None
    
    def get_feature_importance(
        self,
        X: pd.DataFrame = None,
        y: pd.Series = None,
        method: str = "auto"
    ) -> List[FeatureImportance]:
        """
        Extract feature importance using the best available method.
        
        Args:
            X: Feature data (required for some methods)
            y: Target data (required for permutation importance)
            method: 'auto', 'builtin', 'permutation', 'shap'
            
        Returns:
            List of FeatureImportance objects sorted by importance
        """
        if self.model is None:
            raise ValueError("No model set. Call set_model first.")
        
        if method == "auto":
            method = self._select_importance_method()
        
        importances = []
        
        if method == "builtin":
            importances = self._get_builtin_importance(X)
        elif method == "permutation" and X is not None and y is not None:
            importances = self._get_permutation_importance(X, y)
        elif method == "shap" and SHAP_AVAILABLE and X is not None:
            importances = self._get_shap_importance(X)
        elif method == "coefficients":
            importances = self._get_coefficient_importance(X)
        else:
            # Fallback to builtin
            importances = self._get_builtin_importance(X)
        
        # Sort and add ranks
        importances.sort(key=lambda x: abs(x.importance), reverse=True)
        for i, imp in enumerate(importances):
            imp.rank = i + 1
        
        return importances
    
    def _select_importance_method(self) -> str:
        """Select best importance method based on model type."""
        model_type = type(self.model).__name__
        
        # Tree-based models
        if hasattr(self.model, 'feature_importances_'):
            return "builtin"
        
        # Linear models
        if hasattr(self.model, 'coef_'):
            return "coefficients"
        
        # Fallback
        return "permutation"
    
    def _get_builtin_importance(self, X: pd.DataFrame = None) -> List[FeatureImportance]:
        """Get importance from model's feature_importances_ attribute."""
        if not hasattr(self.model, 'feature_importances_'):
            return []
        
        importances = self.model.feature_importances_
        
        if X is not None:
            feature_names = X.columns.tolist()
        else:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        return [
            FeatureImportance(feature=name, importance=float(imp))
            for name, imp in zip(feature_names, importances)
        ]
    
    def _get_coefficient_importance(self, X: pd.DataFrame = None) -> List[FeatureImportance]:
        """Get importance from model coefficients."""
        if not hasattr(self.model, 'coef_'):
            return []
        
        coefs = self.model.coef_
        
        # Handle multi-output
        if coefs.ndim > 1:
            coefs = np.abs(coefs).mean(axis=0)
        
        if X is not None:
            feature_names = X.columns.tolist()
        else:
            feature_names = [f"feature_{i}" for i in range(len(coefs))]
        
        return [
            FeatureImportance(feature=name, importance=float(abs(coef)))
            for name, coef in zip(feature_names, coefs)
        ]
    
    def _get_permutation_importance(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> List[FeatureImportance]:
        """Get importance using permutation method."""
        result = permutation_importance(
            self.model, X, y, 
            n_repeats=10, 
            random_state=42,
            n_jobs=-1
        )
        
        return [
            FeatureImportance(
                feature=name,
                importance=float(imp),
                std=float(std)
            )
            for name, imp, std in zip(
                X.columns, 
                result.importances_mean,
                result.importances_std
            )
        ]
    
    def _get_shap_importance(self, X: pd.DataFrame) -> List[FeatureImportance]:
        """Get importance from SHAP values."""
        if not SHAP_AVAILABLE:
            return []
        
        shap_values = self.get_shap_values(X)
        
        if shap_values is None:
            return []
        
        # Compute mean absolute SHAP values
        if shap_values.ndim == 3:
            # Multi-output: average across outputs
            mean_shap = np.abs(shap_values).mean(axis=(0, 2))
        else:
            mean_shap = np.abs(shap_values).mean(axis=0)
        
        return [
            FeatureImportance(feature=name, importance=float(imp))
            for name, imp in zip(X.columns, mean_shap)
        ]
    
    def get_shap_values(
        self, 
        X: pd.DataFrame,
        check_additivity: bool = False
    ) -> Optional[np.ndarray]:
        """
        Compute SHAP values for the data.
        
        Args:
            X: Feature data
            check_additivity: Whether to check SHAP additivity
            
        Returns:
            SHAP values array or None if unavailable
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available")
            return None
        
        try:
            # Create explainer if needed
            if self._shap_explainer is None:
                self._shap_explainer = self._create_shap_explainer(X)
            
            if self._shap_explainer is None:
                return None
            
            # Compute SHAP values
            shap_values = self._shap_explainer.shap_values(X, check_additivity=check_additivity)
            
            return shap_values
            
        except Exception as e:
            logger.warning(f"SHAP computation failed: {str(e)}")
            return None
    
    def _create_shap_explainer(self, X: pd.DataFrame):
        """Create appropriate SHAP explainer for model type."""
        model_type = type(self.model).__name__
        
        try:
            # Tree-based models
            if model_type in ['RandomForestClassifier', 'RandomForestRegressor',
                              'XGBClassifier', 'XGBRegressor', 
                              'GradientBoostingClassifier', 'GradientBoostingRegressor']:
                explainer = shap.TreeExplainer(self.model)
                self._expected_value = explainer.expected_value
                return explainer
            
            # Linear models
            elif model_type in ['LinearRegression', 'Ridge', 'Lasso',
                                'LogisticRegression', 'ElasticNet']:
                # Use a background sample
                background = shap.sample(X, min(100, len(X)))
                explainer = shap.LinearExplainer(self.model, background)
                self._expected_value = explainer.expected_value
                return explainer
            
            # Fallback to KernelExplainer (slower but works with any model)
            else:
                background = shap.sample(X, min(50, len(X)))
                
                def predict_fn(x):
                    if hasattr(self.model, 'predict_proba'):
                        return self.model.predict_proba(x)
                    return self.model.predict(x)
                
                explainer = shap.KernelExplainer(predict_fn, background)
                self._expected_value = explainer.expected_value
                return explainer
                
        except Exception as e:
            logger.error(f"Failed to create SHAP explainer: {str(e)}")
            return None
    
    def explain_prediction(
        self,
        X_single: pd.DataFrame,
        top_n: int = 5
    ) -> LocalExplanation:
        """
        Explain a single prediction.
        
        Args:
            X_single: Single row DataFrame
            top_n: Number of top features to highlight
            
        Returns:
            LocalExplanation with feature contributions
        """
        if self.model is None:
            raise ValueError("No model set")
        
        if len(X_single) != 1:
            X_single = X_single.head(1)
        
        # Get prediction
        prediction = self.model.predict(X_single)[0]
        
        probability = None
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X_single)[0]
            probability = float(max(proba))
        
        # Get feature contributions
        feature_contributions = {}
        base_value = 0.0
        
        if SHAP_AVAILABLE:
            shap_values = self.get_shap_values(X_single)
            
            if shap_values is not None:
                # Handle multi-class
                if isinstance(shap_values, list):
                    # Take the class with highest probability
                    if probability is not None:
                        class_idx = np.argmax(self.model.predict_proba(X_single)[0])
                        values = shap_values[class_idx][0]
                    else:
                        values = shap_values[0][0]
                else:
                    values = shap_values[0] if shap_values.ndim > 1 else shap_values
                
                for name, val in zip(X_single.columns, values):
                    feature_contributions[name] = float(val)
                
                if self._expected_value is not None:
                    if isinstance(self._expected_value, np.ndarray):
                        base_value = float(self._expected_value.mean())
                    else:
                        base_value = float(self._expected_value)
        else:
            # Fallback: use feature importance as proxy
            importances = self.get_feature_importance(X_single)
            for imp in importances:
                feature_contributions[imp.feature] = imp.importance
        
        # Sort by contribution
        sorted_contribs = sorted(
            feature_contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        top_positive = [(k, v) for k, v in sorted_contribs if v > 0][:top_n]
        top_negative = [(k, v) for k, v in sorted_contribs if v < 0][-top_n:]
        
        return LocalExplanation(
            prediction=prediction,
            probability=probability,
            base_value=base_value,
            feature_contributions=feature_contributions,
            top_positive=top_positive,
            top_negative=top_negative
        )
    
    def get_global_explanation(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        n_samples: int = 100
    ) -> GlobalExplanation:
        """
        Generate global model explanation.
        
        Args:
            X: Feature data
            y: Target data (optional, for permutation importance)
            n_samples: Number of samples for SHAP
            
        Returns:
            GlobalExplanation with feature importances
        """
        # Sample if too large
        if len(X) > n_samples:
            sample_idx = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X.iloc[sample_idx]
            if y is not None:
                y_sample = y.iloc[sample_idx]
            else:
                y_sample = None
        else:
            X_sample = X
            y_sample = y
        
        # Get feature importances
        importances = self.get_feature_importance(X_sample, y_sample)
        
        # Get SHAP values if available
        shap_values = None
        expected_value = None
        
        if SHAP_AVAILABLE:
            shap_values = self.get_shap_values(X_sample)
            expected_value = self._expected_value
        
        explainer_type = self._select_importance_method()
        if SHAP_AVAILABLE and shap_values is not None:
            explainer_type = "shap"
        
        return GlobalExplanation(
            explainer_type=explainer_type,
            feature_importances=importances,
            shap_values=shap_values,
            expected_value=expected_value
        )
    
    def get_coefficient_report(self) -> Optional[Dict[str, Any]]:
        """
        Get coefficient interpretation for linear models.
        
        Returns:
            Dict with coefficient analysis or None
        """
        if not hasattr(self.model, 'coef_'):
            return None
        
        coefs = self.model.coef_
        
        if coefs.ndim > 1:
            # Multi-class
            report = {
                "n_classes": coefs.shape[0],
                "n_features": coefs.shape[1],
                "coefficients_by_class": coefs.tolist()
            }
        else:
            report = {
                "coefficients": coefs.tolist(),
                "n_features": len(coefs)
            }
        
        if hasattr(self.model, 'intercept_'):
            intercept = self.model.intercept_
            if isinstance(intercept, np.ndarray):
                report["intercept"] = intercept.tolist()
            else:
                report["intercept"] = float(intercept)
        
        return report
    
    def to_dict(self, X: pd.DataFrame, y: pd.Series = None) -> Dict[str, Any]:
        """
        Export complete explanation as dictionary.
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            Dict with all explanations
        """
        result = {
            "model_type": type(self.model).__name__,
            "shap_available": SHAP_AVAILABLE
        }
        
        # Feature importance
        importances = self.get_feature_importance(X, y)
        result["feature_importances"] = [
            {
                "feature": imp.feature,
                "importance": imp.importance,
                "rank": imp.rank,
                "std": imp.std
            }
            for imp in importances
        ]
        
        # Coefficient report for linear models
        coef_report = self.get_coefficient_report()
        if coef_report:
            result["coefficient_report"] = coef_report
        
        return result
    
    def save_explanation(
        self, 
        path: Union[str, Path],
        X: pd.DataFrame,
        y: pd.Series = None
    ) -> None:
        """Save explanation to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        explanation = self.to_dict(X, y)
        
        with open(path, "w") as f:
            json.dump(explanation, f, indent=2, default=str)
        
        logger.info(f"Explanation saved to {path}")


def explain_model(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series = None
) -> Dict[str, Any]:
    """
    Quick function to explain a model.
    
    Args:
        model: Trained model
        X: Feature data
        y: Target data (optional)
        
    Returns:
        Dict with explanation
    """
    service = ExplainabilityService(model)
    return service.to_dict(X, y)
