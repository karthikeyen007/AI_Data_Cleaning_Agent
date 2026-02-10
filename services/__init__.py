"""
AI Data Cleaning + AutoML Platform - Services Package

Production-grade service layer providing:
- Source-aware AI routing for data cleaning
- Feature engineering and preprocessing pipelines
- Problem type detection (classification/regression)
- ML training pipeline with multiple algorithms
- Hyperparameter optimization (Grid, Random, Optuna)
- Model comparison and leaderboard generation
- Model versioning and export management
- Retraining and rollback capabilities

Production Features (v2.1):
- Data validation and drift detection
- Async training with job tracking
- Model explainability (SHAP)
- Cost governance and observability
- Unified pipeline export
"""

from .ai_router import AIRouter, DataSourceType
from .preprocessing import PreprocessingPipeline
from .feature_engineering import FeatureEngineer
from .problem_detection import ProblemDetector
from .ml_pipeline import MLPipeline
from .hyperparameter_tuning import HyperparameterTuner
from .metrics_engine import MetricsEngine
from .model_manager import ModelManager

# Production readiness modules
from .data_validation import DataValidator, DataDriftDetector, ValidationResult
from .async_training import AsyncTrainingService, JobTracker, JobStatus, get_async_service
from .explainability import ExplainabilityService, explain_model
from .observability import (
    CostGovernor, 
    TokenEstimator, 
    OperationLogger,
    OperationType,
    get_cost_governor,
    get_operation_logger,
    timed_operation
)
from .unified_pipeline import UnifiedPipeline, create_unified_pipeline

__all__ = [
    # Core services
    "AIRouter",
    "DataSourceType",
    "PreprocessingPipeline",
    "FeatureEngineer",
    "ProblemDetector",
    "MLPipeline",
    "HyperparameterTuner",
    "MetricsEngine",
    "ModelManager",
    
    # Data validation
    "DataValidator",
    "DataDriftDetector",
    "ValidationResult",
    
    # Async training
    "AsyncTrainingService",
    "JobTracker",
    "JobStatus",
    "get_async_service",
    
    # Explainability
    "ExplainabilityService",
    "explain_model",
    
    # Observability & Cost
    "CostGovernor",
    "TokenEstimator",
    "OperationLogger",
    "OperationType",
    "get_cost_governor",
    "get_operation_logger",
    "timed_operation",
    
    # Unified pipeline
    "UnifiedPipeline",
    "create_unified_pipeline",
]

__version__ = "2.1.0"
