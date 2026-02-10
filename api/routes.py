"""
FastAPI Routes - Production-Grade AutoML API

Implements all API endpoints for:
- Data ingestion and cleaning
- ML training and evaluation
- Model management and export
- Hyperparameter tuning
- Retraining workflows
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from io import StringIO, BytesIO

import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from .schemas import (
    # Enums
    DataSourceType, ProblemType, AlgorithmType, TuningMethod,
    MissingValueStrategy, ScalingStrategy, EncodingStrategy,
    
    # Requests
    DatabaseQueryRequest, APIDataRequest, CleaningConfig,
    SelectTargetRequest, PreprocessingConfig, TrainModelRequest,
    TrainMultipleRequest, TuneModelRequest, SaveModelRequest,
    ExportModelRequest, RollbackRequest, RetrainModelRequest,
    PredictRequest,
    
    # Responses
    CleaningResponse, DataProfileResponse, ProblemDetectionResponse,
    TargetSuggestion, PreprocessingResponse, TrainModelResponse,
    TrainingResult, CompareModelsResponse, LeaderboardEntry,
    TuningResponse, EvaluationResponse, ClassificationMetricsResponse,
    RegressionMetricsResponse, ModelVersionInfo, ExportModelResponse,
    RetrainResponse, PredictResponse, HealthResponse, SessionState
)

logger = logging.getLogger(__name__)

# ============================================================================
# Router Setup
# ============================================================================

router = APIRouter(prefix="/api/v2", tags=["AutoML"])
mlops_router = APIRouter(prefix="/api/v2/mlops", tags=["MLOps"])


# ============================================================================
# Session State Management
# ============================================================================

class SessionStore:
    """
    In-memory session store for development.
    In production, use Redis or database-backed sessions.
    """
    
    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._default_session = "default"
    
    def get(self, session_id: str = None) -> Dict[str, Any]:
        session_id = session_id or self._default_session
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "data": None,
                "target_column": None,
                "problem_type": None,
                "preprocessing_pipeline": None,
                "feature_engineer": None,
                "X_train": None,
                "X_test": None,
                "y_train": None,
                "y_test": None,
                "ml_pipeline": None,
                "trained_models": [],
                "best_model": None,
            }
        return self._sessions[session_id]
    
    def clear(self, session_id: str = None):
        session_id = session_id or self._default_session
        if session_id in self._sessions:
            del self._sessions[session_id]


# Global session store
session_store = SessionStore()


def get_session(session_id: str = None) -> Dict[str, Any]:
    """Dependency to get current session."""
    return session_store.get(session_id)


# ============================================================================
# Data Upload & Cleaning Endpoints
# ============================================================================

@router.post("/upload-data", response_model=CleaningResponse)
async def upload_data(
    file: UploadFile = File(...),
    apply_cleaning: bool = True,
    session: Dict = Depends(get_session)
):
    """
    Upload and optionally clean a CSV/Excel file.
    
    - Accepts CSV or Excel files
    - Applies rule-based + AI-powered cleaning if enabled
    - Stores data in session for subsequent operations
    """
    try:
        # Read file
        contents = await file.read()
        file_ext = file.filename.split(".")[-1].lower()
        
        if file_ext == "csv":
            df = pd.read_csv(BytesIO(contents))
        elif file_ext in ["xlsx", "xls"]:
            df = pd.read_excel(BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        raw_shape = list(df.shape)
        
        if apply_cleaning:
            # Import services
            from services.ai_router import AIRouter, DataSourceType as DST, CleaningRequest
            from scripts.data_cleaning import DataCleaning
            
            # Rule-based cleaning first
            cleaner = DataCleaning()
            df = cleaner.clean_data(df)
            
            # AI-powered cleaning
            ai_router = AIRouter()
            request = CleaningRequest(
                data=df,
                source_type=DST.UPLOAD,
                batch_size=20
            )
            result = ai_router.clean(request)
            df = result.cleaned_data
            model_used = result.model_used
            processing_time = result.processing_time_ms
        else:
            model_used = None
            processing_time = 0
        
        # Store in session
        session["data"] = df
        session["source_type"] = "upload"
        
        return CleaningResponse(
            success=True,
            cleaned_data=df.head(100).to_dict(orient="records"),
            raw_shape=raw_shape,
            cleaned_shape=list(df.shape),
            model_used=model_used,
            processing_time_ms=processing_time,
            message=f"Successfully processed {file.filename}"
        )
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clean-data", response_model=CleaningResponse)
async def clean_data(
    config: CleaningConfig,
    session: Dict = Depends(get_session)
):
    """
    Clean data currently in session using source-aware AI routing.
    """
    if session["data"] is None:
        raise HTTPException(status_code=400, detail="No data in session. Upload data first.")
    
    try:
        from services.ai_router import AIRouter, DataSourceType as DST, CleaningRequest
        from scripts.data_cleaning import DataCleaning
        
        df = session["data"]
        raw_shape = list(df.shape)
        
        # Map source type
        source_map = {
            DataSourceType.DATABASE: DST.DATABASE,
            DataSourceType.UPLOAD: DST.UPLOAD,
            DataSourceType.API: DST.API
        }
        
        # Rule-based cleaning
        cleaner = DataCleaning()
        df = cleaner.clean_data(df)
        
        # AI-powered cleaning
        ai_router = AIRouter()
        request = CleaningRequest(
            data=df,
            source_type=source_map[config.source_type],
            batch_size=config.batch_size,
            max_retries=config.max_retries
        )
        result = ai_router.clean(request)
        
        # Update session
        session["data"] = result.cleaned_data
        
        return CleaningResponse(
            success=True,
            cleaned_data=result.cleaned_data.head(100).to_dict(orient="records"),
            raw_shape=raw_shape,
            cleaned_shape=list(result.cleaned_data.shape),
            model_used=result.model_used,
            processing_time_ms=result.processing_time_ms,
            message="Data cleaned successfully",
            errors=result.errors
        )
        
    except Exception as e:
        logger.error(f"Cleaning error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clean-database", response_model=CleaningResponse)
async def clean_database(
    query: DatabaseQueryRequest,
    session: Dict = Depends(get_session)
):
    """
    Execute database query and clean results.
    """
    try:
        from sqlalchemy import create_engine
        from services.ai_router import AIRouter, DataSourceType as DST, CleaningRequest
        from scripts.data_cleaning import DataCleaning
        
        # Execute query
        engine = create_engine(query.db_url)
        df = pd.read_sql(query.query, engine)
        raw_shape = list(df.shape)
        
        # Rule-based cleaning
        cleaner = DataCleaning()
        df = cleaner.clean_data(df)
        
        # AI cleaning with database-specific model
        ai_router = AIRouter()
        request = CleaningRequest(
            data=df,
            source_type=DST.DATABASE,
            batch_size=20
        )
        result = ai_router.clean(request)
        
        # Store in session
        session["data"] = result.cleaned_data
        session["source_type"] = "database"
        
        return CleaningResponse(
            success=True,
            cleaned_data=result.cleaned_data.head(100).to_dict(orient="records"),
            raw_shape=raw_shape,
            cleaned_shape=list(result.cleaned_data.shape),
            model_used=result.model_used,
            processing_time_ms=result.processing_time_ms,
            message="Database data cleaned successfully"
        )
        
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clean-api", response_model=CleaningResponse)
async def clean_api_data(
    request: APIDataRequest,
    session: Dict = Depends(get_session)
):
    """
    Fetch and clean data from external API.
    """
    try:
        import aiohttp
        from services.ai_router import AIRouter, DataSourceType as DST, CleaningRequest
        from scripts.data_cleaning import DataCleaning
        
        # Fetch data
        async with aiohttp.ClientSession() as client:
            async with client.get(request.api_url, headers=request.headers) as resp:
                if resp.status != 200:
                    raise HTTPException(status_code=400, detail="Failed to fetch API data")
                data = await resp.json()
        
        df = pd.DataFrame(data)
        raw_shape = list(df.shape)
        
        # Rule-based cleaning
        cleaner = DataCleaning()
        df = cleaner.clean_data(df)
        
        # AI cleaning with API-specific model
        ai_router = AIRouter()
        clean_request = CleaningRequest(
            data=df,
            source_type=DST.API,
            batch_size=20
        )
        result = ai_router.clean(clean_request)
        
        # Store in session
        session["data"] = result.cleaned_data
        session["source_type"] = "api"
        
        return CleaningResponse(
            success=True,
            cleaned_data=result.cleaned_data.head(100).to_dict(orient="records"),
            raw_shape=raw_shape,
            cleaned_shape=list(result.cleaned_data.shape),
            model_used=result.model_used,
            processing_time_ms=result.processing_time_ms,
            message="API data cleaned successfully"
        )
        
    except Exception as e:
        logger.error(f"API data error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Target Selection & Problem Detection
# ============================================================================

@router.get("/suggest-targets", response_model=List[TargetSuggestion])
async def suggest_targets(session: Dict = Depends(get_session)):
    """
    Suggest potential target columns based on data analysis.
    """
    if session["data"] is None:
        raise HTTPException(status_code=400, detail="No data in session")
    
    from services.problem_detection import ProblemDetector
    
    detector = ProblemDetector()
    suggestions = detector.suggest_target(session["data"])
    
    return [TargetSuggestion(**s) for s in suggestions]


@router.post("/select-target", response_model=ProblemDetectionResponse)
async def select_target(
    request: SelectTargetRequest,
    session: Dict = Depends(get_session)
):
    """
    Select target column and detect problem type.
    """
    if session["data"] is None:
        raise HTTPException(status_code=400, detail="No data in session")
    
    df = session["data"]
    
    if request.target_column not in df.columns:
        raise HTTPException(
            status_code=400, 
            detail=f"Column '{request.target_column}' not found"
        )
    
    from services.problem_detection import ProblemDetector
    from services.problem_detection import ProblemType as PT
    
    detector = ProblemDetector()
    
    # Map force type
    force_type = None
    if request.force_problem_type:
        force_map = {
            ProblemType.BINARY_CLASSIFICATION: PT.BINARY_CLASSIFICATION,
            ProblemType.MULTICLASS_CLASSIFICATION: PT.MULTICLASS_CLASSIFICATION,
            ProblemType.REGRESSION: PT.REGRESSION
        }
        force_type = force_map.get(request.force_problem_type)
    
    result = detector.detect(df, request.target_column, force_type)
    
    # Store in session
    session["target_column"] = request.target_column
    session["problem_type"] = result.problem_type.value
    
    return ProblemDetectionResponse(
        problem_type=ProblemType(result.problem_type.value),
        target_column=request.target_column,
        unique_values=result.target_analysis.unique_values,
        is_trainable=result.is_trainable,
        recommended_algorithms=[
            {
                "name": algo.name,
                "algorithm_id": algo.algorithm_id,
                "priority": algo.priority,
                "reason": algo.reason
            }
            for algo in result.recommended_algorithms
        ],
        data_issues=[issue.value for issue in result.data_issues],
        warnings=result.warnings,
        dataset_stats=result.dataset_stats
    )


# ============================================================================
# Preprocessing
# ============================================================================

@router.post("/preprocess", response_model=PreprocessingResponse)
async def preprocess_data(
    config: PreprocessingConfig,
    session: Dict = Depends(get_session)
):
    """
    Apply preprocessing pipeline to data.
    """
    if session["data"] is None:
        raise HTTPException(status_code=400, detail="No data in session")
    if session["target_column"] is None:
        raise HTTPException(status_code=400, detail="Target not selected")
    
    from services.preprocessing import (
        PreprocessingPipeline, 
        PreprocessingConfig as PPConfig,
        MissingValueStrategy as MVS,
        ScalingStrategy as SS,
        EncodingStrategy as ES
    )
    from services.feature_engineering import FeatureEngineer, SplitConfig
    
    df = session["data"]
    target = session["target_column"]
    
    # Map strategies
    mvs_map = {
        MissingValueStrategy.MEAN: MVS.MEAN,
        MissingValueStrategy.MEDIAN: MVS.MEDIAN,
        MissingValueStrategy.MODE: MVS.MODE,
        MissingValueStrategy.DROP: MVS.DROP
    }
    
    ss_map = {
        ScalingStrategy.STANDARD: SS.STANDARD,
        ScalingStrategy.MINMAX: SS.MINMAX,
        ScalingStrategy.ROBUST: SS.ROBUST,
        ScalingStrategy.NONE: SS.NONE
    }
    
    es_map = {
        EncodingStrategy.LABEL: ES.LABEL,
        EncodingStrategy.ONEHOT: ES.ONEHOT,
        EncodingStrategy.ORDINAL: ES.ORDINAL,
        EncodingStrategy.NONE: ES.NONE
    }
    
    # Create preprocessing config
    pp_config = PPConfig(
        missing_strategy=mvs_map[config.missing_strategy],
        scaling_strategy=ss_map[config.scaling_strategy],
        encoding_strategy=es_map[config.encoding_strategy],
        drop_constant_columns=config.drop_constant_columns,
        outlier_handling=config.handle_outliers,
        outlier_threshold=config.outlier_threshold
    )
    
    # Apply preprocessing
    pipeline = PreprocessingPipeline(pp_config)
    original_shape = list(df.shape)
    
    df_processed = pipeline.fit_transform(df, target)
    
    # Split data
    X = df_processed.drop(columns=[target])
    y = df_processed[target]
    
    is_classification = session["problem_type"] in ["binary_classification", "multiclass_classification"]
    
    split_config = SplitConfig(
        test_size=0.2,
        stratify=is_classification
    )
    
    X_train, X_test, y_train, y_test = FeatureEngineer.split_data(X, y, split_config)
    
    # Store in session
    session["preprocessing_pipeline"] = pipeline
    session["X_train"] = X_train
    session["X_test"] = X_test
    session["y_train"] = y_train
    session["y_test"] = y_test
    
    return PreprocessingResponse(
        success=True,
        original_shape=original_shape,
        processed_shape=list(df_processed.shape),
        numeric_columns=pipeline._numeric_columns,
        categorical_columns=pipeline._categorical_columns,
        dropped_columns=pipeline._dropped_columns,
        feature_names=X.columns.tolist(),
        message="Preprocessing completed successfully"
    )


# ============================================================================
# Training
# ============================================================================

@router.post("/train-model", response_model=TrainModelResponse)
async def train_model(
    request: TrainModelRequest,
    session: Dict = Depends(get_session)
):
    """
    Train a single model with specified configuration.
    """
    if session["X_train"] is None:
        raise HTTPException(status_code=400, detail="Data not preprocessed")
    
    from services.ml_pipeline import MLPipeline, AlgorithmType as AT, TrainingConfig
    
    # Map algorithm
    algo_map = {
        AlgorithmType.LINEAR_REGRESSION: AT.LINEAR_REGRESSION,
        AlgorithmType.RIDGE: AT.RIDGE,
        AlgorithmType.LASSO: AT.LASSO,
        AlgorithmType.RANDOM_FOREST_REG: AT.RANDOM_FOREST_REG,
        AlgorithmType.XGBOOST_REG: AT.XGBOOST_REG,
        AlgorithmType.LOGISTIC_REGRESSION: AT.LOGISTIC_REGRESSION,
        AlgorithmType.RANDOM_FOREST_CLF: AT.RANDOM_FOREST_CLF,
        AlgorithmType.SVM_CLF: AT.SVM_CLF,
        AlgorithmType.XGBOOST_CLF: AT.XGBOOST_CLF,
    }
    
    try:
        pipeline = session.get("ml_pipeline") or MLPipeline()
        
        config = TrainingConfig(
            algorithm=algo_map[request.algorithm],
            hyperparameters=request.hyperparameters or {},
            cv_folds=request.cv_folds
        )
        
        result = pipeline.train(
            session["X_train"],
            session["y_train"],
            config
        )
        
        # Store
        session["ml_pipeline"] = pipeline
        session["trained_models"].append(result.algorithm)
        if pipeline._best_model:
            session["best_model"] = pipeline._best_model.algorithm
        
        return TrainModelResponse(
            success=True,
            result=TrainingResult(
                algorithm=result.algorithm,
                cv_mean=result.cv_mean,
                cv_std=result.cv_std,
                cv_scores=result.cv_scores,
                training_time_seconds=result.training_time_seconds,
                feature_importances=result.feature_importances,
                success=result.success,
                error_message=result.error_message
            ),
            message=f"Model {result.algorithm} trained successfully"
        )
        
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare-models", response_model=CompareModelsResponse)
async def compare_models(
    request: TrainMultipleRequest,
    session: Dict = Depends(get_session)
):
    """
    Train multiple models and generate leaderboard.
    """
    if session["X_train"] is None:
        raise HTTPException(status_code=400, detail="Data not preprocessed")
    
    from services.ml_pipeline import MLPipeline, AlgorithmType as AT
    
    algo_map = {
        AlgorithmType.LINEAR_REGRESSION: AT.LINEAR_REGRESSION,
        AlgorithmType.RIDGE: AT.RIDGE,
        AlgorithmType.LASSO: AT.LASSO,
        AlgorithmType.RANDOM_FOREST_REG: AT.RANDOM_FOREST_REG,
        AlgorithmType.XGBOOST_REG: AT.XGBOOST_REG,
        AlgorithmType.LOGISTIC_REGRESSION: AT.LOGISTIC_REGRESSION,
        AlgorithmType.RANDOM_FOREST_CLF: AT.RANDOM_FOREST_CLF,
        AlgorithmType.SVM_CLF: AT.SVM_CLF,
        AlgorithmType.XGBOOST_CLF: AT.XGBOOST_CLF,
    }
    
    try:
        pipeline = MLPipeline()
        
        algorithms = [algo_map[a] for a in request.algorithms]
        
        results = pipeline.train_multiple(
            session["X_train"],
            session["y_train"],
            algorithms,
            cv_folds=request.cv_folds
        )
        
        leaderboard = pipeline.get_leaderboard()
        
        # Store
        session["ml_pipeline"] = pipeline
        session["trained_models"] = [r.algorithm for r in results]
        if pipeline._best_model:
            session["best_model"] = pipeline._best_model.algorithm
        
        return CompareModelsResponse(
            success=True,
            leaderboard=[
                LeaderboardEntry(
                    rank=entry["rank"],
                    algorithm=entry["algorithm"],
                    cv_mean=entry["cv_mean"],
                    cv_std=entry["cv_std"],
                    training_time=entry["training_time"],
                    success=entry["success"]
                )
                for entry in leaderboard
            ],
            best_model=pipeline._best_model.algorithm if pipeline._best_model else "",
            message=f"Trained {len(results)} models"
        )
        
    except Exception as e:
        logger.error(f"Comparison error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Hyperparameter Tuning
# ============================================================================

@router.post("/tune-model", response_model=TuningResponse)
async def tune_model(
    request: TuneModelRequest,
    session: Dict = Depends(get_session)
):
    """
    Tune hyperparameters for an algorithm.
    """
    if session["X_train"] is None:
        raise HTTPException(status_code=400, detail="Data not preprocessed")
    
    from services.hyperparameter_tuning import (
        HyperparameterTuner, 
        TuningConfig,
        TuningMethod as TM
    )
    from services.ml_pipeline import AlgorithmType as AT
    
    algo_map = {
        AlgorithmType.RIDGE: AT.RIDGE,
        AlgorithmType.LASSO: AT.LASSO,
        AlgorithmType.RANDOM_FOREST_REG: AT.RANDOM_FOREST_REG,
        AlgorithmType.XGBOOST_REG: AT.XGBOOST_REG,
        AlgorithmType.LOGISTIC_REGRESSION: AT.LOGISTIC_REGRESSION,
        AlgorithmType.RANDOM_FOREST_CLF: AT.RANDOM_FOREST_CLF,
        AlgorithmType.SVM_CLF: AT.SVM_CLF,
        AlgorithmType.XGBOOST_CLF: AT.XGBOOST_CLF,
    }
    
    method_map = {
        TuningMethod.GRID_SEARCH: TM.GRID_SEARCH,
        TuningMethod.RANDOM_SEARCH: TM.RANDOM_SEARCH,
        TuningMethod.OPTUNA: TM.OPTUNA
    }
    
    try:
        tuner = HyperparameterTuner()
        
        config = TuningConfig(
            method=method_map[request.method],
            n_trials=request.n_trials,
            cv_folds=request.cv_folds,
            timeout=request.timeout_seconds
        )
        
        result = tuner.tune(
            session["X_train"],
            session["y_train"],
            algo_map[request.algorithm],
            config,
            request.custom_search_space
        )
        
        return TuningResponse(
            success=True,
            algorithm=result.algorithm,
            method=result.method,
            best_params=result.best_params,
            best_score=result.best_score,
            n_trials_completed=result.n_trials_completed,
            total_time_seconds=result.total_time_seconds,
            message=f"Tuning completed with best score {result.best_score:.4f}"
        )
        
    except Exception as e:
        logger.error(f"Tuning error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MLOps Routes - Model Management
# ============================================================================

@mlops_router.post("/save-model", response_model=ModelVersionInfo)
async def save_model(
    request: SaveModelRequest,
    user_id: str = "default",
    session: Dict = Depends(get_session)
):
    """
    Save trained model with versioning.
    """
    if session["ml_pipeline"] is None or session["ml_pipeline"]._best_model is None:
        raise HTTPException(status_code=400, detail="No trained model to save")
    
    from services.model_manager import ModelManager
    
    manager = ModelManager()
    best = session["ml_pipeline"]._best_model
    
    version = manager.save_model(
        user_id=user_id,
        project_id=request.project_id,
        model=best.model,
        metrics={"cv_mean": best.cv_mean, "cv_std": best.cv_std},
        hyperparameters=best.hyperparameters,
        feature_names=session["X_train"].columns.tolist(),
        target_name=session["target_column"],
        training_data=session["data"],
        preprocessing_pipeline=session["preprocessing_pipeline"],
        notes=request.notes
    )
    
    return ModelVersionInfo(
        version_id=version.version_id,
        algorithm=version.algorithm,
        created_at=version.created_at,
        status=version.status.value,
        metrics=version.metrics,
        notes=version.notes
    )


@mlops_router.get("/versions/{project_id}", response_model=List[ModelVersionInfo])
async def list_versions(
    project_id: str,
    user_id: str = "default"
):
    """
    List all versions for a project.
    """
    from services.model_manager import ModelManager
    
    manager = ModelManager()
    versions = manager.list_versions(user_id, project_id)
    
    return [
        ModelVersionInfo(
            version_id=v.version_id,
            algorithm=v.algorithm,
            created_at=v.created_at,
            status=v.status.value,
            metrics=v.metrics,
            notes=v.notes
        )
        for v in versions
    ]


@mlops_router.post("/export-model", response_model=ExportModelResponse)
async def export_model(
    request: ExportModelRequest,
    user_id: str = "default"
):
    """
    Export model package for deployment.
    """
    from services.model_manager import ModelManager
    
    manager = ModelManager()
    
    try:
        export_path = manager.export_model(
            user_id=user_id,
            project_id=request.project_id,
            version_id=request.version_id
        )
        
        files = [f.name for f in export_path.iterdir()]
        
        return ExportModelResponse(
            success=True,
            export_path=str(export_path),
            files=files,
            message="Model exported successfully"
        )
        
    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@mlops_router.post("/rollback", response_model=Dict[str, Any])
async def rollback_version(
    request: RollbackRequest,
    user_id: str = "default"
):
    """
    Rollback to a previous model version.
    """
    from services.model_manager import ModelManager
    
    manager = ModelManager()
    
    try:
        success = manager.rollback(
            user_id=user_id,
            project_id=request.project_id,
            target_version_id=request.version_id
        )
        
        return {
            "success": success,
            "message": f"Rolled back to version {request.version_id}"
        }
        
    except Exception as e:
        logger.error(f"Rollback error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@mlops_router.post("/retrain", response_model=RetrainResponse)
async def retrain_model(
    request: RetrainModelRequest,
    user_id: str = "default",
    session: Dict = Depends(get_session)
):
    """
    Retrain model with new data or tuned hyperparameters.
    """
    if session["X_train"] is None:
        raise HTTPException(status_code=400, detail="No training data")
    
    from services.model_manager import ModelManager
    from services.ml_pipeline import MLPipeline, AlgorithmType as AT, TrainingConfig
    from services.hyperparameter_tuning import HyperparameterTuner, TuningConfig
    
    manager = ModelManager()
    
    try:
        # Load previous version
        model, metadata, preprocessing = manager.load_model(
            user_id, request.project_id, request.version_id
        )
        
        previous_version_id = request.version_id or metadata.get("version_id", "unknown")
        algorithm_name = metadata["algorithm"]
        
        # Map algorithm
        algo_map = {
            "RandomForestClassifier": AT.RANDOM_FOREST_CLF,
            "RandomForestRegressor": AT.RANDOM_FOREST_REG,
            "LogisticRegression": AT.LOGISTIC_REGRESSION,
            "XGBClassifier": AT.XGBOOST_CLF,
            "XGBRegressor": AT.XGBOOST_REG,
            "Ridge": AT.RIDGE,
            "Lasso": AT.LASSO,
            "LinearRegression": AT.LINEAR_REGRESSION,
            "SVC": AT.SVM_CLF,
        }
        
        algorithm = algo_map.get(algorithm_name)
        if not algorithm:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        # Optionally tune hyperparameters
        best_params = metadata.get("hyperparameters", {})
        
        if request.tune_hyperparameters and request.tuning_config:
            tuner = HyperparameterTuner()
            tuning_result = tuner.tune(
                session["X_train"],
                session["y_train"],
                algorithm
            )
            best_params = tuning_result.best_params
        
        # Train with new data
        pipeline = MLPipeline()
        config = TrainingConfig(
            algorithm=algorithm,
            hyperparameters=best_params
        )
        
        result = pipeline.train(
            session["X_train"],
            session["y_train"],
            config
        )
        
        # Save new version
        new_version = manager.save_model(
            user_id=user_id,
            project_id=request.project_id,
            model=result.model,
            metrics={"cv_mean": result.cv_mean, "cv_std": result.cv_std},
            hyperparameters=best_params,
            feature_names=session["X_train"].columns.tolist(),
            target_name=session["target_column"],
            preprocessing_pipeline=session["preprocessing_pipeline"],
            notes=request.notes,
            parent_version=previous_version_id
        )
        
        return RetrainResponse(
            success=True,
            new_version_id=new_version.version_id,
            previous_version_id=previous_version_id,
            metrics_comparison={
                "previous": metadata.get("metrics", {}),
                "new": {"cv_mean": result.cv_mean, "cv_std": result.cv_std}
            },
            message="Model retrained successfully"
        )
        
    except Exception as e:
        logger.error(f"Retrain error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@mlops_router.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    user_id: str = "default"
):
    """
    Make predictions using saved model.
    """
    from services.model_manager import ModelManager
    
    manager = ModelManager()
    
    try:
        inference_pipeline = manager.create_inference_pipeline(
            user_id, request.project_id, request.version_id
        )
        
        df = pd.DataFrame(request.data)
        predictions = inference_pipeline.predict(df)
        
        probabilities = None
        try:
            probabilities = inference_pipeline.predict_proba(df).tolist()
        except:
            pass
        
        return PredictResponse(
            success=True,
            predictions=predictions.tolist(),
            probabilities=probabilities,
            message="Predictions generated successfully"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Health & Status
# ============================================================================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    API health check.
    """
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        services={
            "ai_router": "available",
            "ml_pipeline": "available",
            "model_manager": "available"
        },
        timestamp=datetime.now().isoformat()
    )


@router.get("/session-state", response_model=SessionState)
async def get_session_state(session: Dict = Depends(get_session)):
    """
    Get current session state.
    """
    return SessionState(
        has_data=session["data"] is not None,
        data_shape=list(session["data"].shape) if session["data"] is not None else None,
        data_source=session.get("source_type"),
        target_selected=session["target_column"] is not None,
        target_column=session["target_column"],
        problem_type=session["problem_type"],
        preprocessing_done=session["preprocessing_pipeline"] is not None,
        models_trained=len(session["trained_models"]),
        best_model=session["best_model"]
    )


@router.post("/reset-session")
async def reset_session(session_id: str = None):
    """
    Reset current session.
    """
    session_store.clear(session_id)
    return {"success": True, "message": "Session reset"}
