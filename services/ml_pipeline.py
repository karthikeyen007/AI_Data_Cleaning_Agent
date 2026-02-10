"""
ML Pipeline Service

Production-grade machine learning training pipeline:
- Multiple algorithm support for classification and regression
- Automatic preprocessing integration
- Cross-validation and evaluation
- Model serialization and export
- Training progress tracking
"""

import logging
import time
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json

# Sklearn imports
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR

# XGBoost
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    
logger = logging.getLogger(__name__)


class AlgorithmType(Enum):
    """Supported ML algorithms."""
    # Regression
    LINEAR_REGRESSION = "linear_regression"
    RIDGE = "ridge"
    LASSO = "lasso"
    RANDOM_FOREST_REG = "random_forest_reg"
    XGBOOST_REG = "xgboost_reg"
    
    # Classification
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST_CLF = "random_forest_clf"
    SVM_CLF = "svm_clf"
    XGBOOST_CLF = "xgboost_clf"


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    algorithm: AlgorithmType
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    cv_folds: int = 5
    random_state: int = 42
    n_jobs: int = -1
    scoring: Optional[str] = None  # None = use default for problem type
    early_stopping: bool = True
    verbose: int = 1


@dataclass
class TrainingResult:
    """Result of a training run."""
    algorithm: str
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    training_time_seconds: float
    hyperparameters: Dict[str, Any]
    feature_importances: Optional[Dict[str, float]]
    model: Any  # Fitted model
    success: bool
    error_message: Optional[str] = None


@dataclass
class ModelArtifact:
    """Complete model artifact for deployment."""
    model: Any
    preprocessing_pipeline: Any
    feature_names: List[str]
    target_name: str
    algorithm: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    created_at: str
    version: str


class MLPipeline:
    """
    Production-grade ML training pipeline.
    
    Provides:
    - Multi-algorithm training
    - Cross-validation
    - Feature importance extraction
    - Model serialization
    - Training callbacks
    """
    
    # Algorithm factory
    ALGORITHM_REGISTRY = {
        # Regression
        AlgorithmType.LINEAR_REGRESSION: LinearRegression,
        AlgorithmType.RIDGE: Ridge,
        AlgorithmType.LASSO: Lasso,
        AlgorithmType.RANDOM_FOREST_REG: RandomForestRegressor,
        
        # Classification
        AlgorithmType.LOGISTIC_REGRESSION: LogisticRegression,
        AlgorithmType.RANDOM_FOREST_CLF: RandomForestClassifier,
        AlgorithmType.SVM_CLF: SVC,
    }
    
    # Default scoring metrics
    DEFAULT_SCORING = {
        "classification": "accuracy",
        "regression": "r2"
    }
    
    # Default hyperparameters
    DEFAULT_HYPERPARAMETERS = {
        AlgorithmType.LINEAR_REGRESSION: {},
        AlgorithmType.RIDGE: {"alpha": 1.0},
        AlgorithmType.LASSO: {"alpha": 1.0, "max_iter": 10000},
        AlgorithmType.RANDOM_FOREST_REG: {"n_estimators": 100, "max_depth": None, "n_jobs": -1},
        AlgorithmType.LOGISTIC_REGRESSION: {"C": 1.0, "max_iter": 1000, "n_jobs": -1},
        AlgorithmType.RANDOM_FOREST_CLF: {"n_estimators": 100, "max_depth": None, "n_jobs": -1},
        AlgorithmType.SVM_CLF: {"C": 1.0, "kernel": "rbf", "probability": True},
    }
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the ML pipeline.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self._trained_models: Dict[str, TrainingResult] = {}
        self._best_model: Optional[TrainingResult] = None
        self._callbacks: List[Callable] = []
        
        # Register XGBoost if available
        if XGBOOST_AVAILABLE:
            self.ALGORITHM_REGISTRY[AlgorithmType.XGBOOST_REG] = XGBRegressor
            self.ALGORITHM_REGISTRY[AlgorithmType.XGBOOST_CLF] = XGBClassifier
            self.DEFAULT_HYPERPARAMETERS[AlgorithmType.XGBOOST_REG] = {
                "n_estimators": 100, "learning_rate": 0.1, "max_depth": 6, "n_jobs": -1
            }
            self.DEFAULT_HYPERPARAMETERS[AlgorithmType.XGBOOST_CLF] = {
                "n_estimators": 100, "learning_rate": 0.1, "max_depth": 6, 
                "n_jobs": -1, "use_label_encoder": False, "eval_metric": "logloss"
            }
        
        logger.info("MLPipeline initialized")
    
    def add_callback(self, callback: Callable[[str, Any], None]) -> None:
        """
        Add a callback for training progress.
        
        Args:
            callback: Function taking (event_name, data)
        """
        self._callbacks.append(callback)
    
    def _emit(self, event: str, data: Any) -> None:
        """Emit an event to all callbacks."""
        for callback in self._callbacks:
            try:
                callback(event, data)
            except Exception as e:
                logger.warning(f"Callback error: {e}")
    
    def _create_model(
        self, 
        algorithm: AlgorithmType,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> BaseEstimator:
        """
        Create a model instance.
        
        Args:
            algorithm: Algorithm type
            hyperparameters: Custom hyperparameters
            
        Returns:
            Configured model instance
        """
        if algorithm not in self.ALGORITHM_REGISTRY:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        model_class = self.ALGORITHM_REGISTRY[algorithm]
        params = self.DEFAULT_HYPERPARAMETERS.get(algorithm, {}).copy()
        
        if hyperparameters:
            params.update(hyperparameters)
        
        # Add random state if supported
        if "random_state" in model_class().get_params():
            params["random_state"] = self.random_state
        
        return model_class(**params)
    
    def _extract_feature_importances(
        self, 
        model: BaseEstimator, 
        feature_names: List[str]
    ) -> Optional[Dict[str, float]]:
        """
        Extract feature importances from a fitted model.
        
        Args:
            model: Fitted model
            feature_names: List of feature names
            
        Returns:
            Dictionary of feature importances or None
        """
        importances = None
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            if len(coef.shape) > 1:
                importances = np.abs(coef).mean(axis=0)
            else:
                importances = np.abs(coef)
        
        if importances is not None:
            return dict(zip(feature_names, importances.tolist()))
        
        return None
    
    def train(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        config: TrainingConfig
    ) -> TrainingResult:
        """
        Train a single model with cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            config: Training configuration
            
        Returns:
            TrainingResult with metrics and fitted model
        """
        algorithm_name = config.algorithm.value
        logger.info(f"Training {algorithm_name}...")
        
        self._emit("training_start", {"algorithm": algorithm_name})
        
        start_time = time.time()
        
        try:
            # Create model
            model = self._create_model(config.algorithm, config.hyperparameters)
            
            # Determine scoring
            scoring = config.scoring
            if scoring is None:
                # Infer from algorithm type
                if config.algorithm.value.endswith("_reg") or config.algorithm in [
                    AlgorithmType.LINEAR_REGRESSION, AlgorithmType.RIDGE, AlgorithmType.LASSO
                ]:
                    scoring = self.DEFAULT_SCORING["regression"]
                else:
                    scoring = self.DEFAULT_SCORING["classification"]
            
            # Cross-validation
            cv_results = cross_validate(
                model, X, y,
                cv=config.cv_folds,
                scoring=scoring,
                return_train_score=False,
                n_jobs=config.n_jobs
            )
            
            cv_scores = cv_results["test_score"].tolist()
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            
            # Fit on full data
            model.fit(X, y)
            
            # Extract feature importances
            feature_importances = self._extract_feature_importances(model, X.columns.tolist())
            
            training_time = time.time() - start_time
            
            result = TrainingResult(
                algorithm=algorithm_name,
                cv_scores=cv_scores,
                cv_mean=cv_mean,
                cv_std=cv_std,
                training_time_seconds=round(training_time, 2),
                hyperparameters=config.hyperparameters or self.DEFAULT_HYPERPARAMETERS.get(config.algorithm, {}),
                feature_importances=feature_importances,
                model=model,
                success=True
            )
            
            # Store result
            self._trained_models[algorithm_name] = result
            
            # Update best model
            if self._best_model is None or cv_mean > self._best_model.cv_mean:
                self._best_model = result
            
            logger.info(f"{algorithm_name}: CV Score = {cv_mean:.4f} (+/- {cv_std:.4f})")
            
            self._emit("training_complete", {
                "algorithm": algorithm_name,
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "training_time": training_time
            })
            
            return result
            
        except Exception as e:
            training_time = time.time() - start_time
            logger.error(f"Training failed for {algorithm_name}: {str(e)}")
            
            result = TrainingResult(
                algorithm=algorithm_name,
                cv_scores=[],
                cv_mean=0.0,
                cv_std=0.0,
                training_time_seconds=round(training_time, 2),
                hyperparameters=config.hyperparameters or {},
                feature_importances=None,
                model=None,
                success=False,
                error_message=str(e)
            )
            
            self._emit("training_error", {"algorithm": algorithm_name, "error": str(e)})
            
            return result
    
    def train_multiple(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        algorithms: List[AlgorithmType],
        cv_folds: int = 5,
        scoring: Optional[str] = None
    ) -> List[TrainingResult]:
        """
        Train multiple algorithms and compare.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            algorithms: List of algorithms to train
            cv_folds: Cross-validation folds
            scoring: Scoring metric
            
        Returns:
            List of TrainingResults sorted by performance
        """
        results = []
        
        for algorithm in algorithms:
            config = TrainingConfig(
                algorithm=algorithm,
                cv_folds=cv_folds,
                scoring=scoring,
                random_state=self.random_state
            )
            
            result = self.train(X, y, config)
            results.append(result)
        
        # Sort by CV mean (descending)
        results.sort(key=lambda x: x.cv_mean, reverse=True)
        
        return results
    
    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """
        Get ranked leaderboard of all trained models.
        
        Returns:
            List of model summaries sorted by performance
        """
        leaderboard = []
        
        for name, result in self._trained_models.items():
            leaderboard.append({
                "rank": 0,  # Will be set below
                "algorithm": result.algorithm,
                "cv_mean": round(result.cv_mean, 4),
                "cv_std": round(result.cv_std, 4),
                "training_time": result.training_time_seconds,
                "success": result.success,
                "error": result.error_message
            })
        
        # Sort and assign ranks
        leaderboard.sort(key=lambda x: x["cv_mean"], reverse=True)
        for i, entry in enumerate(leaderboard):
            entry["rank"] = i + 1
        
        return leaderboard
    
    def get_best_model(self) -> Optional[TrainingResult]:
        """Get the best performing model."""
        return self._best_model
    
    def predict(
        self, 
        X: pd.DataFrame, 
        model: Optional[BaseEstimator] = None
    ) -> np.ndarray:
        """
        Make predictions with a model.
        
        Args:
            X: Feature DataFrame
            model: Model to use (default: best model)
            
        Returns:
            Predictions array
        """
        if model is None:
            if self._best_model is None:
                raise ValueError("No trained model available")
            model = self._best_model.model
        
        return model.predict(X)
    
    def predict_proba(
        self, 
        X: pd.DataFrame, 
        model: Optional[BaseEstimator] = None
    ) -> np.ndarray:
        """
        Get prediction probabilities (classification only).
        
        Args:
            X: Feature DataFrame
            model: Model to use (default: best model)
            
        Returns:
            Probability array
        """
        if model is None:
            if self._best_model is None:
                raise ValueError("No trained model available")
            model = self._best_model.model
        
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        else:
            raise ValueError("Model does not support predict_proba")
    
    def save_model(
        self, 
        path: Union[str, Path],
        model_result: Optional[TrainingResult] = None,
        preprocessing_pipeline: Any = None
    ) -> str:
        """
        Save a model to disk.
        
        Args:
            path: Directory to save model
            model_result: Model to save (default: best model)
            preprocessing_pipeline: Optional preprocessing pipeline to include
            
        Returns:
            Path where model was saved
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if model_result is None:
            model_result = self._best_model
        
        if model_result is None:
            raise ValueError("No model to save")
        
        # Save model
        joblib.dump(model_result.model, path / "model.pkl")
        
        # Save preprocessing if provided
        if preprocessing_pipeline:
            joblib.dump(preprocessing_pipeline, path / "preprocessing.pkl")
        
        # Save metadata
        metadata = {
            "algorithm": model_result.algorithm,
            "cv_mean": model_result.cv_mean,
            "cv_std": model_result.cv_std,
            "hyperparameters": model_result.hyperparameters,
            "training_time": model_result.training_time_seconds,
            "feature_importances": model_result.feature_importances,
            "created_at": pd.Timestamp.now().isoformat()
        }
        
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {path}")
        
        return str(path)
    
    @classmethod
    def load_model(cls, path: Union[str, Path]) -> Tuple[BaseEstimator, Dict[str, Any], Any]:
        """
        Load a model from disk.
        
        Args:
            path: Directory containing model files
            
        Returns:
            Tuple of (model, metadata, preprocessing_pipeline)
        """
        path = Path(path)
        
        model = joblib.load(path / "model.pkl")
        
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        preprocessing = None
        if (path / "preprocessing.pkl").exists():
            preprocessing = joblib.load(path / "preprocessing.pkl")
        
        logger.info(f"Model loaded from {path}")
        
        return model, metadata, preprocessing


# Convenience functions
def get_classification_algorithms() -> List[AlgorithmType]:
    """Get list of classification algorithms."""
    algos = [
        AlgorithmType.LOGISTIC_REGRESSION,
        AlgorithmType.RANDOM_FOREST_CLF,
        AlgorithmType.SVM_CLF,
    ]
    if XGBOOST_AVAILABLE:
        algos.append(AlgorithmType.XGBOOST_CLF)
    return algos


def get_regression_algorithms() -> List[AlgorithmType]:
    """Get list of regression algorithms."""
    algos = [
        AlgorithmType.LINEAR_REGRESSION,
        AlgorithmType.RIDGE,
        AlgorithmType.LASSO,
        AlgorithmType.RANDOM_FOREST_REG,
    ]
    if XGBOOST_AVAILABLE:
        algos.append(AlgorithmType.XGBOOST_REG)
    return algos
