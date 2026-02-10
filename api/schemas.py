"""
API Schemas - Request and Response Models

Defines Pydantic models for all API endpoints:
- Data upload and cleaning
- ML training and evaluation
- Model management
- Hyperparameter tuning
"""

from enum import Enum
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field
from datetime import datetime


# ============================================================================
# Enums
# ============================================================================

class DataSourceType(str, Enum):
    """Data source types."""
    DATABASE = "database"
    UPLOAD = "upload"
    API = "api"


class ProblemType(str, Enum):
    """ML problem types."""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"


class AlgorithmType(str, Enum):
    """Supported algorithms."""
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


class TuningMethod(str, Enum):
    """Hyperparameter tuning methods."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    OPTUNA = "optuna"


class MissingValueStrategy(str, Enum):
    """Missing value handling strategies."""
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "most_frequent"
    CONSTANT = "constant"
    DROP = "drop"


class ScalingStrategy(str, Enum):
    """Scaling strategies."""
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    NONE = "none"


class EncodingStrategy(str, Enum):
    """Encoding strategies."""
    LABEL = "label"
    ONEHOT = "onehot"
    ORDINAL = "ordinal"
    NONE = "none"


# ============================================================================
# Data Upload & Cleaning
# ============================================================================

class DatabaseQueryRequest(BaseModel):
    """Request for database query."""
    db_url: str = Field(..., description="Database connection URL")
    query: str = Field(..., description="SQL query to execute")


class APIDataRequest(BaseModel):
    """Request for API data fetch."""
    api_url: str = Field(..., description="REST API URL")
    headers: Optional[Dict[str, str]] = Field(default=None, description="Request headers")
    method: str = Field(default="GET", description="HTTP method")


class CleaningConfig(BaseModel):
    """Configuration for data cleaning."""
    source_type: DataSourceType = Field(..., description="Type of data source")
    batch_size: int = Field(default=20, ge=1, le=100, description="Rows per batch")
    max_retries: int = Field(default=3, ge=1, le=10, description="Max retry attempts")


class CleaningResponse(BaseModel):
    """Response from data cleaning."""
    model_config = {"protected_namespaces": ()}  # Allow 'model_' prefix
    
    success: bool
    cleaned_data: Optional[List[Dict[str, Any]]] = None
    raw_shape: Optional[List[int]] = None
    cleaned_shape: Optional[List[int]] = None
    model_used: Optional[str] = None
    processing_time_ms: Optional[float] = None
    message: str
    errors: List[str] = Field(default_factory=list)


class DataProfileResponse(BaseModel):
    """Dataset profile response."""
    n_rows: int
    n_cols: int
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    missing_values: Dict[str, int]
    column_profiles: List[Dict[str, Any]]


# ============================================================================
# Target Selection & Problem Detection
# ============================================================================

class SelectTargetRequest(BaseModel):
    """Request to select target column."""
    target_column: str = Field(..., description="Name of target column")
    force_problem_type: Optional[ProblemType] = Field(
        default=None, description="Override automatic detection"
    )


class TargetSuggestion(BaseModel):
    """A suggested target column."""
    column: str
    score: int
    unique_values: int
    missing_pct: float
    dtype: str
    reasons: List[str]


class ProblemDetectionResponse(BaseModel):
    """Response from problem detection."""
    problem_type: ProblemType
    target_column: str
    unique_values: int
    is_trainable: bool
    recommended_algorithms: List[Dict[str, Any]]
    data_issues: List[str]
    warnings: List[str]
    dataset_stats: Dict[str, Any]


# ============================================================================
# Preprocessing
# ============================================================================

class PreprocessingConfig(BaseModel):
    """Configuration for preprocessing."""
    missing_strategy: MissingValueStrategy = Field(
        default=MissingValueStrategy.MEDIAN
    )
    scaling_strategy: ScalingStrategy = Field(
        default=ScalingStrategy.STANDARD
    )
    encoding_strategy: EncodingStrategy = Field(
        default=EncodingStrategy.ONEHOT
    )
    drop_constant_columns: bool = Field(default=True)
    handle_outliers: bool = Field(default=False)
    outlier_threshold: float = Field(default=3.0)


class PreprocessingResponse(BaseModel):
    """Response from preprocessing."""
    success: bool
    original_shape: List[int]
    processed_shape: List[int]
    numeric_columns: List[str]
    categorical_columns: List[str]
    dropped_columns: List[str]
    feature_names: List[str]
    message: str


# ============================================================================
# Training
# ============================================================================

class TrainModelRequest(BaseModel):
    """Request to train a model."""
    algorithm: AlgorithmType = Field(..., description="Algorithm to use")
    hyperparameters: Optional[Dict[str, Any]] = Field(
        default=None, description="Custom hyperparameters"
    )
    cv_folds: int = Field(default=5, ge=2, le=20, description="CV folds")
    test_size: float = Field(default=0.2, ge=0.1, le=0.5, description="Test set ratio")


class TrainMultipleRequest(BaseModel):
    """Request to train multiple models."""
    algorithms: List[AlgorithmType] = Field(..., description="Algorithms to train")
    cv_folds: int = Field(default=5, ge=2, le=20)
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)


class TrainingResult(BaseModel):
    """Result of model training."""
    algorithm: str
    cv_mean: float
    cv_std: float
    cv_scores: List[float]
    training_time_seconds: float
    feature_importances: Optional[Dict[str, float]] = None
    success: bool
    error_message: Optional[str] = None


class TrainModelResponse(BaseModel):
    """Response from model training."""
    success: bool
    result: Optional[TrainingResult] = None
    message: str


class LeaderboardEntry(BaseModel):
    """Entry in model leaderboard."""
    rank: int
    algorithm: str
    cv_mean: float
    cv_std: float
    training_time: float
    success: bool


class CompareModelsResponse(BaseModel):
    """Response from model comparison."""
    success: bool
    leaderboard: List[LeaderboardEntry]
    best_model: str
    message: str


# ============================================================================
# Hyperparameter Tuning
# ============================================================================

class TuneModelRequest(BaseModel):
    """Request to tune hyperparameters."""
    algorithm: AlgorithmType = Field(..., description="Algorithm to tune")
    method: TuningMethod = Field(default=TuningMethod.RANDOM_SEARCH)
    n_trials: int = Field(default=50, ge=10, le=500)
    cv_folds: int = Field(default=5, ge=2, le=10)
    timeout_seconds: Optional[int] = Field(default=None)
    custom_search_space: Optional[Dict[str, List[Any]]] = Field(default=None)


class TuningResponse(BaseModel):
    """Response from hyperparameter tuning."""
    success: bool
    algorithm: str
    method: str
    best_params: Dict[str, Any]
    best_score: float
    n_trials_completed: int
    total_time_seconds: float
    message: str


# ============================================================================
# Metrics & Evaluation
# ============================================================================

class ClassificationMetricsResponse(BaseModel):
    """Classification metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float] = None
    confusion_matrix: List[List[int]]
    class_labels: List[str]


class RegressionMetricsResponse(BaseModel):
    """Regression metrics."""
    rmse: float
    mae: float
    r2: float
    mape: Optional[float] = None
    explained_variance: float


class EvaluationResponse(BaseModel):
    """Model evaluation response."""
    success: bool
    problem_type: str
    metrics: Union[ClassificationMetricsResponse, RegressionMetricsResponse]
    message: str


# ============================================================================
# Model Management
# ============================================================================

class SaveModelRequest(BaseModel):
    """Request to save a model."""
    project_id: str = Field(..., description="Project identifier")
    notes: str = Field(default="", description="Version notes")


class ModelVersionInfo(BaseModel):
    """Information about a model version."""
    version_id: str
    algorithm: str
    created_at: str
    status: str
    metrics: Dict[str, float]
    notes: str


class ExportModelRequest(BaseModel):
    """Request to export a model."""
    project_id: str = Field(..., description="Project identifier")
    version_id: Optional[str] = Field(default=None, description="Version to export")


class ExportModelResponse(BaseModel):
    """Response from model export."""
    success: bool
    export_path: Optional[str] = None
    files: List[str] = Field(default_factory=list)
    message: str


class RollbackRequest(BaseModel):
    """Request to rollback to a version."""
    project_id: str
    version_id: str


# ============================================================================
# Retraining
# ============================================================================

class RetrainModelRequest(BaseModel):
    """Request to retrain a model."""
    project_id: str = Field(..., description="Project identifier")
    version_id: Optional[str] = Field(
        default=None, description="Version to base retraining on"
    )
    tune_hyperparameters: bool = Field(
        default=False, description="Whether to re-tune hyperparameters"
    )
    tuning_config: Optional[TuneModelRequest] = Field(default=None)
    notes: str = Field(default="Retrained model")


class RetrainResponse(BaseModel):
    """Response from model retraining."""
    success: bool
    new_version_id: Optional[str] = None
    previous_version_id: Optional[str] = None
    metrics_comparison: Optional[Dict[str, Dict[str, float]]] = None
    message: str


# ============================================================================
# Prediction
# ============================================================================

class PredictRequest(BaseModel):
    """Request for predictions."""
    project_id: str
    version_id: Optional[str] = None
    data: List[Dict[str, Any]] = Field(..., description="Data to predict on")


class PredictResponse(BaseModel):
    """Response from predictions."""
    success: bool
    predictions: List[Any]
    probabilities: Optional[List[List[float]]] = None
    message: str


# ============================================================================
# Health & Status
# ============================================================================

class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    version: str
    services: Dict[str, str]
    timestamp: str


class SessionState(BaseModel):
    """Current session state."""
    has_data: bool
    data_shape: Optional[List[int]] = None
    data_source: Optional[str] = None
    target_selected: bool
    target_column: Optional[str] = None
    problem_type: Optional[str] = None
    preprocessing_done: bool
    models_trained: int
    best_model: Optional[str] = None


# ============================================================================
# Async Training Jobs
# ============================================================================

class JobStatus(str, Enum):
    """Status of async training jobs."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class SubmitTrainingRequest(BaseModel):
    """Request to submit async training job."""
    algorithm: str = Field(..., description="Algorithm to train")
    hyperparameters: Optional[Dict[str, Any]] = None
    cv_folds: int = Field(default=5, ge=2, le=20)
    timeout_seconds: int = Field(default=3600, ge=60, le=86400)


class SubmitComparisonRequest(BaseModel):
    """Request to submit async model comparison."""
    algorithms: List[str] = Field(..., min_length=1)
    cv_folds: int = Field(default=5, ge=2, le=20)
    timeout_seconds: int = Field(default=7200)


class JobSubmitResponse(BaseModel):
    """Response from job submission."""
    job_id: str
    status: JobStatus
    message: str


class JobStatusResponse(BaseModel):
    """Response with job status."""
    job_id: str
    status: JobStatus
    algorithm: str
    progress: float = Field(ge=0.0, le=1.0)
    current_step: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    logs: List[Dict[str, Any]] = Field(default_factory=list)


class JobListResponse(BaseModel):
    """List of jobs."""
    jobs: List[JobStatusResponse]
    total: int


# ============================================================================
# Data Validation
# ============================================================================

class ValidationIssue(BaseModel):
    """A validation issue."""
    type: str
    severity: str
    column: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


class ValidationResponse(BaseModel):
    """Response from data validation."""
    is_valid: bool
    schema_match: bool
    errors_count: int
    warnings_count: int
    issues: List[ValidationIssue]
    validated_at: str


class DriftReport(BaseModel):
    """Data drift detection report."""
    column: str
    has_drift: bool
    mean_drift_zscore: float
    std_ratio: float
    reference_mean: float
    current_mean: float


class DriftResponse(BaseModel):
    """Response from drift detection."""
    drift_detected: bool
    drifted_columns: List[str]
    reports: List[DriftReport]


# ============================================================================
# Explainability
# ============================================================================

class FeatureImportanceItem(BaseModel):
    """Single feature importance entry."""
    feature: str
    importance: float
    rank: int
    std: Optional[float] = None


class LocalExplanationResponse(BaseModel):
    """Explanation for a single prediction."""
    prediction: Any
    probability: Optional[float] = None
    base_value: float
    feature_contributions: Dict[str, float]
    top_positive_features: List[Dict[str, float]]
    top_negative_features: List[Dict[str, float]]


class GlobalExplanationResponse(BaseModel):
    """Global model explanation."""
    model_type: str
    explainer_type: str
    feature_importances: List[FeatureImportanceItem]
    coefficient_report: Optional[Dict[str, Any]] = None
    shap_available: bool


class ExplainRequest(BaseModel):
    """Request for prediction explanation."""
    project_id: str
    version_id: Optional[str] = None
    data: List[Dict[str, Any]] = Field(..., description="Data to explain")
    top_n_features: int = Field(default=5, ge=1, le=20)


# ============================================================================
# Cost Governance
# ============================================================================

class CostEstimateResponse(BaseModel):
    """Cost estimate for an operation."""
    operation: str
    estimated_tokens: int
    estimated_cost_usd: float
    model_used: str
    within_budget: bool
    warnings: List[str] = Field(default_factory=list)


class DailySpendResponse(BaseModel):
    """Daily spending report."""
    date: str
    total_spent_usd: float
    remaining_budget_usd: float
    daily_limit_usd: float


class DatasetLimitsResponse(BaseModel):
    """Dataset limits check."""
    rows: int
    columns: int
    max_rows: int
    max_columns: int
    within_limits: bool
    violations: List[str] = Field(default_factory=list)


# ============================================================================
# Unified Pipeline Export
# ============================================================================

class ExportUnifiedPipelineRequest(BaseModel):
    """Request to export unified pipeline."""
    project_id: str
    version_id: Optional[str] = None
    output_filename: str = Field(default="pipeline.pkl")


class UnifiedPipelineExportResponse(BaseModel):
    """Response from unified pipeline export."""
    success: bool
    export_path: str
    pipeline_info: Dict[str, Any]
    message: str
