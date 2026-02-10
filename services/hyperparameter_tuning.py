"""
Hyperparameter Tuning Service

Production-grade hyperparameter optimization:
- GridSearchCV integration
- RandomSearchCV integration
- Optuna integration for Bayesian optimization
- Automatic search space definition
- Best parameter persistence
"""

import logging
import time
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from pathlib import Path
import json

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator

# Optuna
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from .ml_pipeline import AlgorithmType, MLPipeline

logger = logging.getLogger(__name__)


class TuningMethod(Enum):
    """Hyperparameter tuning methods."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    OPTUNA = "optuna"


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning."""
    method: TuningMethod = TuningMethod.RANDOM_SEARCH
    n_trials: int = 100  # For random search and Optuna
    cv_folds: int = 5
    scoring: Optional[str] = None
    n_jobs: int = -1
    timeout: Optional[int] = None  # Seconds
    random_state: int = 42
    verbose: int = 1


@dataclass
class TuningResult:
    """Result of hyperparameter tuning."""
    algorithm: str
    method: str
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict[str, Any]]
    n_trials_completed: int
    total_time_seconds: float
    cv_results: Optional[Dict[str, Any]] = None


class HyperparameterTuner:
    """
    Production-grade hyperparameter optimization.
    
    Supports multiple tuning strategies with automatic
    search space generation and result persistence.
    """
    
    # Default search spaces for each algorithm
    SEARCH_SPACES = {
        # Regression
        AlgorithmType.RIDGE: {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        },
        AlgorithmType.LASSO: {
            "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
        },
        AlgorithmType.RANDOM_FOREST_REG: {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [None, 10, 20, 30, 50],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None]
        },
        AlgorithmType.XGBOOST_REG: {
            "n_estimators": [50, 100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7, 10],
            "min_child_weight": [1, 3, 5],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0]
        },
        
        # Classification
        AlgorithmType.LOGISTIC_REGRESSION: {
            "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "saga"]
        },
        AlgorithmType.RANDOM_FOREST_CLF: {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [None, 10, 20, 30, 50],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None]
        },
        AlgorithmType.SVM_CLF: {
            "C": [0.1, 1.0, 10.0, 100.0],
            "kernel": ["rbf", "poly", "sigmoid"],
            "gamma": ["scale", "auto"]
        },
        AlgorithmType.XGBOOST_CLF: {
            "n_estimators": [50, 100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7, 10],
            "min_child_weight": [1, 3, 5],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0]
        }
    }
    
    # Optuna-specific search spaces (continuous ranges)
    OPTUNA_SPACES = {
        AlgorithmType.RIDGE: {
            "alpha": ("log_uniform", 1e-4, 100.0)
        },
        AlgorithmType.LASSO: {
            "alpha": ("log_uniform", 1e-5, 10.0)
        },
        AlgorithmType.RANDOM_FOREST_REG: {
            "n_estimators": ("int", 50, 500),
            "max_depth": ("int_or_none", 5, 100),
            "min_samples_split": ("int", 2, 20),
            "min_samples_leaf": ("int", 1, 10),
        },
        AlgorithmType.XGBOOST_REG: {
            "n_estimators": ("int", 50, 500),
            "learning_rate": ("log_uniform", 0.001, 0.3),
            "max_depth": ("int", 3, 15),
            "min_child_weight": ("int", 1, 10),
            "subsample": ("uniform", 0.5, 1.0),
            "colsample_bytree": ("uniform", 0.5, 1.0),
        },
        AlgorithmType.LOGISTIC_REGRESSION: {
            "C": ("log_uniform", 1e-4, 100.0),
        },
        AlgorithmType.RANDOM_FOREST_CLF: {
            "n_estimators": ("int", 50, 500),
            "max_depth": ("int_or_none", 5, 100),
            "min_samples_split": ("int", 2, 20),
            "min_samples_leaf": ("int", 1, 10),
        },
        AlgorithmType.SVM_CLF: {
            "C": ("log_uniform", 0.01, 100.0),
            "gamma": ("categorical", ["scale", "auto"]),
        },
        AlgorithmType.XGBOOST_CLF: {
            "n_estimators": ("int", 50, 500),
            "learning_rate": ("log_uniform", 0.001, 0.3),
            "max_depth": ("int", 3, 15),
            "min_child_weight": ("int", 1, 10),
            "subsample": ("uniform", 0.5, 1.0),
            "colsample_bytree": ("uniform", 0.5, 1.0),
        }
    }
    
    def __init__(self, ml_pipeline: Optional[MLPipeline] = None):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            ml_pipeline: Optional MLPipeline instance
        """
        self.ml_pipeline = ml_pipeline or MLPipeline()
        self._tuning_history: List[TuningResult] = []
        self._best_result: Optional[TuningResult] = None
        
        logger.info("HyperparameterTuner initialized")
        if OPTUNA_AVAILABLE:
            logger.info("Optuna available for Bayesian optimization")
    
    def _get_search_space(self, algorithm: AlgorithmType) -> Dict[str, Any]:
        """Get the default search space for an algorithm."""
        return self.SEARCH_SPACES.get(algorithm, {})
    
    def _tune_grid_search(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Dict[str, List],
        config: TuningConfig
    ) -> TuningResult:
        """
        Tune using GridSearchCV.
        
        Args:
            model: Model to tune
            X: Features
            y: Target
            param_grid: Parameter grid
            config: Tuning configuration
            
        Returns:
            TuningResult
        """
        start_time = time.time()
        
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=config.cv_folds,
            scoring=config.scoring,
            n_jobs=config.n_jobs,
            verbose=config.verbose,
            return_train_score=False
        )
        
        grid_search.fit(X, y)
        
        # Extract results
        all_results = []
        for i in range(len(grid_search.cv_results_["params"])):
            all_results.append({
                "params": grid_search.cv_results_["params"][i],
                "mean_score": grid_search.cv_results_["mean_test_score"][i],
                "std_score": grid_search.cv_results_["std_test_score"][i]
            })
        
        total_time = time.time() - start_time
        
        return TuningResult(
            algorithm=type(model).__name__,
            method="grid_search",
            best_params=grid_search.best_params_,
            best_score=grid_search.best_score_,
            all_results=all_results,
            n_trials_completed=len(all_results),
            total_time_seconds=round(total_time, 2),
            cv_results=dict(grid_search.cv_results_)
        )
    
    def _tune_random_search(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        param_distributions: Dict[str, Any],
        config: TuningConfig
    ) -> TuningResult:
        """
        Tune using RandomizedSearchCV.
        
        Args:
            model: Model to tune
            X: Features
            y: Target
            param_distributions: Parameter distributions
            config: Tuning configuration
            
        Returns:
            TuningResult
        """
        start_time = time.time()
        
        random_search = RandomizedSearchCV(
            model,
            param_distributions,
            n_iter=config.n_trials,
            cv=config.cv_folds,
            scoring=config.scoring,
            n_jobs=config.n_jobs,
            verbose=config.verbose,
            random_state=config.random_state,
            return_train_score=False
        )
        
        random_search.fit(X, y)
        
        # Extract results
        all_results = []
        for i in range(len(random_search.cv_results_["params"])):
            all_results.append({
                "params": random_search.cv_results_["params"][i],
                "mean_score": random_search.cv_results_["mean_test_score"][i],
                "std_score": random_search.cv_results_["std_test_score"][i]
            })
        
        total_time = time.time() - start_time
        
        return TuningResult(
            algorithm=type(model).__name__,
            method="random_search",
            best_params=random_search.best_params_,
            best_score=random_search.best_score_,
            all_results=all_results,
            n_trials_completed=len(all_results),
            total_time_seconds=round(total_time, 2),
            cv_results=dict(random_search.cv_results_)
        )
    
    def _tune_optuna(
        self,
        model_class: type,
        X: pd.DataFrame,
        y: pd.Series,
        algorithm: AlgorithmType,
        config: TuningConfig
    ) -> TuningResult:
        """
        Tune using Optuna Bayesian optimization.
        
        Args:
            model_class: Model class to instantiate
            X: Features
            y: Target
            algorithm: Algorithm type
            config: Tuning configuration
            
        Returns:
            TuningResult
        """
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna is not installed. Run: pip install optuna")
        
        start_time = time.time()
        
        search_space = self.OPTUNA_SPACES.get(algorithm, {})
        all_results = []
        
        def objective(trial: optuna.Trial) -> float:
            params = {}
            
            for param_name, spec in search_space.items():
                spec_type = spec[0]
                
                if spec_type == "int":
                    params[param_name] = trial.suggest_int(param_name, spec[1], spec[2])
                elif spec_type == "int_or_none":
                    use_none = trial.suggest_categorical(f"{param_name}_none", [True, False])
                    if use_none:
                        params[param_name] = None
                    else:
                        params[param_name] = trial.suggest_int(param_name, spec[1], spec[2])
                elif spec_type == "uniform":
                    params[param_name] = trial.suggest_float(param_name, spec[1], spec[2])
                elif spec_type == "log_uniform":
                    params[param_name] = trial.suggest_float(param_name, spec[1], spec[2], log=True)
                elif spec_type == "categorical":
                    params[param_name] = trial.suggest_categorical(param_name, spec[1])
            
            # Create and evaluate model
            model = model_class(**params, random_state=config.random_state)
            
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(
                model, X, y,
                cv=config.cv_folds,
                scoring=config.scoring,
                n_jobs=1  # Optuna handles parallelization
            )
            
            mean_score = np.mean(scores)
            
            all_results.append({
                "params": params.copy(),
                "mean_score": mean_score,
                "std_score": np.std(scores)
            })
            
            return mean_score
        
        # Create study
        sampler = TPESampler(seed=config.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        
        # Optimize
        study.optimize(
            objective,
            n_trials=config.n_trials,
            timeout=config.timeout,
            show_progress_bar=config.verbose > 0
        )
        
        total_time = time.time() - start_time
        
        return TuningResult(
            algorithm=model_class.__name__,
            method="optuna",
            best_params=study.best_params,
            best_score=study.best_value,
            all_results=all_results,
            n_trials_completed=len(study.trials),
            total_time_seconds=round(total_time, 2)
        )
    
    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        algorithm: AlgorithmType,
        config: Optional[TuningConfig] = None,
        custom_search_space: Optional[Dict[str, Any]] = None
    ) -> TuningResult:
        """
        Tune hyperparameters for an algorithm.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            algorithm: Algorithm to tune
            config: Tuning configuration
            custom_search_space: Custom search space (overrides defaults)
            
        Returns:
            TuningResult with best parameters
        """
        config = config or TuningConfig()
        
        logger.info(f"Starting hyperparameter tuning for {algorithm.value}")
        logger.info(f"Method: {config.method.value}, Trials: {config.n_trials}")
        
        # Get model class
        model_class = self.ml_pipeline.ALGORITHM_REGISTRY.get(algorithm)
        if model_class is None:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Get search space
        search_space = custom_search_space or self._get_search_space(algorithm)
        
        if not search_space:
            logger.warning(f"No search space defined for {algorithm.value}")
            # Return with default params
            return TuningResult(
                algorithm=algorithm.value,
                method=config.method.value,
                best_params={},
                best_score=0.0,
                all_results=[],
                n_trials_completed=0,
                total_time_seconds=0.0
            )
        
        # Create base model
        model = model_class(random_state=config.random_state)
        
        # Run tuning
        if config.method == TuningMethod.GRID_SEARCH:
            result = self._tune_grid_search(model, X, y, search_space, config)
        elif config.method == TuningMethod.RANDOM_SEARCH:
            result = self._tune_random_search(model, X, y, search_space, config)
        elif config.method == TuningMethod.OPTUNA:
            result = self._tune_optuna(model_class, X, y, algorithm, config)
        else:
            raise ValueError(f"Unknown tuning method: {config.method}")
        
        # Store result
        self._tuning_history.append(result)
        
        if self._best_result is None or result.best_score > self._best_result.best_score:
            self._best_result = result
        
        logger.info(f"Tuning complete. Best score: {result.best_score:.4f}")
        logger.info(f"Best params: {result.best_params}")
        
        return result
    
    def tune_and_train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        algorithm: AlgorithmType,
        tuning_config: Optional[TuningConfig] = None
    ) -> Tuple[TuningResult, Any]:
        """
        Tune hyperparameters and train final model.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            algorithm: Algorithm to tune and train
            tuning_config: Tuning configuration
            
        Returns:
            Tuple of (TuningResult, fitted_model)
        """
        # Tune
        tuning_result = self.tune(X, y, algorithm, tuning_config)
        
        # Train with best params
        model_class = self.ml_pipeline.ALGORITHM_REGISTRY[algorithm]
        best_model = model_class(**tuning_result.best_params, random_state=42)
        best_model.fit(X, y)
        
        return tuning_result, best_model
    
    def get_tuning_history(self) -> List[TuningResult]:
        """Get all tuning results."""
        return self._tuning_history
    
    def get_best_result(self) -> Optional[TuningResult]:
        """Get the best tuning result."""
        return self._best_result
    
    def save_results(self, path: Union[str, Path]) -> None:
        """
        Save tuning results to disk.
        
        Args:
            path: Directory to save results
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        results = []
        for result in self._tuning_history:
            results.append({
                "algorithm": result.algorithm,
                "method": result.method,
                "best_params": result.best_params,
                "best_score": result.best_score,
                "n_trials": result.n_trials_completed,
                "time_seconds": result.total_time_seconds
            })
        
        with open(path / "tuning_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Tuning results saved to {path}")
    
    @classmethod
    def load_results(cls, path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load tuning results from disk.
        
        Args:
            path: Directory containing results
            
        Returns:
            List of result dictionaries
        """
        path = Path(path)
        
        with open(path / "tuning_results.json", "r") as f:
            results = json.load(f)
        
        return results
