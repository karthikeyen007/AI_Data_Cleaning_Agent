"""
Production API Routes - Async Training, Validation, Explainability, Cost Governance

This module adds production-grade endpoints for:
- Async training job submission and tracking
- Data validation and drift detection
- Model explainability (SHAP, feature importance)
- Cost governance and budget tracking
- Unified pipeline export
"""

import logging
from datetime import datetime
from typing import Dict, Any, List

import pandas as pd
from fastapi import APIRouter, HTTPException, Depends

from .schemas import (
    # Async Training
    SubmitTrainingRequest, SubmitComparisonRequest,
    JobSubmitResponse, JobStatusResponse, JobListResponse,
    JobStatus,
    
    # Data Validation
    ValidationResponse, ValidationIssue,
    DriftResponse, DriftReport,
    
    # Explainability
    ExplainRequest, LocalExplanationResponse,
    GlobalExplanationResponse, FeatureImportanceItem,
    
    # Cost Governance
    CostEstimateResponse, DailySpendResponse, DatasetLimitsResponse,
    
    # Unified Pipeline
    ExportUnifiedPipelineRequest, UnifiedPipelineExportResponse
)

from .routes import get_session, session_store

logger = logging.getLogger(__name__)

# ============================================================================
# Router Setup
# ============================================================================

production_router = APIRouter(prefix="/api/v2/production", tags=["Production"])


# ============================================================================
# Async Training Endpoints
# ============================================================================

@production_router.post("/jobs/train", response_model=JobSubmitResponse)
async def submit_training_job(
    request: SubmitTrainingRequest,
    session: Dict = Depends(get_session)
):
    """
    Submit an async training job.
    
    Returns immediately with a job_id for status tracking.
    """
    if session["X_train"] is None:
        raise HTTPException(status_code=400, detail="No training data. Run preprocessing first.")
    
    from services.async_training import get_async_service
    
    service = get_async_service()
    
    job_id = service.submit_training(
        X_train=session["X_train"],
        y_train=session["y_train"],
        algorithm=request.algorithm,
        config={
            "hyperparameters": request.hyperparameters or {},
            "cv_folds": request.cv_folds
        },
        timeout=request.timeout_seconds
    )
    
    return JobSubmitResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message=f"Training job submitted for {request.algorithm}"
    )


@production_router.post("/jobs/compare", response_model=JobSubmitResponse)
async def submit_comparison_job(
    request: SubmitComparisonRequest,
    session: Dict = Depends(get_session)
):
    """
    Submit async model comparison job.
    """
    if session["X_train"] is None:
        raise HTTPException(status_code=400, detail="No training data")
    
    from services.async_training import get_async_service
    
    service = get_async_service()
    
    job_id = service.submit_comparison(
        X_train=session["X_train"],
        y_train=session["y_train"],
        algorithms=request.algorithms,
        config={"cv_folds": request.cv_folds},
        timeout=request.timeout_seconds
    )
    
    return JobSubmitResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message=f"Comparison job submitted for {len(request.algorithms)} algorithms"
    )


@production_router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get status and results of a training job.
    """
    from services.async_training import get_async_service
    
    service = get_async_service()
    job = service.get_job_status(job_id)
    
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatusResponse(
        job_id=job["job_id"],
        status=JobStatus(job["status"]),
        algorithm=job["algorithm"],
        progress=job["progress"],
        current_step=job["current_step"],
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        result=job.get("result"),
        error=job.get("error"),
        logs=job.get("logs", [])
    )


@production_router.get("/jobs", response_model=JobListResponse)
async def list_jobs(status: str = None, limit: int = 50):
    """
    List training jobs, optionally filtered by status.
    """
    from services.async_training import get_async_service, JobStatus as JS
    
    service = get_async_service()
    
    status_filter = JS(status) if status else None
    jobs = service.tracker.list_jobs(status=status_filter, limit=limit)
    
    return JobListResponse(
        jobs=[
            JobStatusResponse(
                job_id=j.job_id,
                status=JobStatus(j.status.value),
                algorithm=j.algorithm,
                progress=j.progress,
                current_step=j.current_step,
                started_at=j.started_at,
                completed_at=j.completed_at,
                result=j.result,
                error=j.error
            )
            for j in jobs
        ],
        total=len(jobs)
    )


@production_router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running job."""
    from services.async_training import get_async_service
    
    service = get_async_service()
    success = service.cancel_job(job_id)
    
    return {"success": success, "message": "Job cancelled" if success else "Could not cancel job"}


# ============================================================================
# Data Validation Endpoints
# ============================================================================

@production_router.post("/validate", response_model=ValidationResponse)
async def validate_data(
    data: List[Dict[str, Any]],
    project_id: str = None,
    version_id: str = None,
    user_id: str = "default"
):
    """
    Validate inference data against the training schema.
    
    If project_id is provided, uses the schema from that model.
    Otherwise, uses the current session's preprocessing schema.
    """
    from services.data_validation import DataValidator
    from services.model_manager import ModelManager
    
    df = pd.DataFrame(data)
    
    if project_id:
        # Load schema from saved model
        manager = ModelManager()
        _, metadata, _ = manager.load_model(user_id, project_id, version_id)
        
        # Create validator with schema from metadata
        validator = DataValidator()
        if metadata.get("schema"):
            from services.data_validation import DataSchema
            validator.schema = DataSchema.from_dict(metadata["schema"])
        else:
            # Build schema from feature names
            validator.learn_schema(pd.DataFrame(columns=metadata.get("feature_names", [])))
    else:
        # Use current session data
        session = session_store.get()
        if session.get("X_train") is None:
            raise HTTPException(status_code=400, detail="No training data for validation")
        
        validator = DataValidator()
        validator.learn_schema(session["X_train"])
    
    result = validator.validate(df)
    
    return ValidationResponse(
        is_valid=result.is_valid,
        schema_match=result.schema_match,
        errors_count=result.errors_count,
        warnings_count=result.warnings_count,
        issues=[
            ValidationIssue(
                type=i.issue_type.value,
                severity=i.severity.value,
                column=i.column,
                message=i.message,
                details=i.details
            )
            for i in result.issues
        ],
        validated_at=result.validated_at
    )


@production_router.post("/detect-drift", response_model=DriftResponse)
async def detect_drift(
    data: List[Dict[str, Any]],
    session: Dict = Depends(get_session)
):
    """
    Detect data drift between training and inference data.
    """
    if session.get("X_train") is None:
        raise HTTPException(status_code=400, detail="No training data for comparison")
    
    from services.data_validation import DataDriftDetector
    
    df = pd.DataFrame(data)
    
    detector = DataDriftDetector(session["X_train"])
    drift_report = detector.detect_drift(df)
    
    drifted = [col for col, info in drift_report.items() if info["has_drift"]]
    
    return DriftResponse(
        drift_detected=len(drifted) > 0,
        drifted_columns=drifted,
        reports=[
            DriftReport(
                column=col,
                has_drift=info["has_drift"],
                mean_drift_zscore=info["mean_drift_zscore"],
                std_ratio=info["std_ratio"],
                reference_mean=info["reference_mean"],
                current_mean=info["current_mean"]
            )
            for col, info in drift_report.items()
        ]
    )


# ============================================================================
# Explainability Endpoints
# ============================================================================

@production_router.post("/explain", response_model=LocalExplanationResponse)
async def explain_prediction(
    request: ExplainRequest,
    user_id: str = "default"
):
    """
    Explain a prediction with feature contributions.
    """
    from services.model_manager import ModelManager
    from services.explainability import ExplainabilityService
    
    manager = ModelManager()
    
    try:
        model, metadata, preprocessing = manager.load_model(
            user_id, request.project_id, request.version_id
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model not found: {str(e)}")
    
    df = pd.DataFrame(request.data)
    
    # Apply preprocessing if available
    if preprocessing:
        df = preprocessing.transform(df)
    
    # Explain
    explainer = ExplainabilityService(model)
    explanation = explainer.explain_prediction(df.head(1), top_n=request.top_n_features)
    
    return LocalExplanationResponse(
        prediction=explanation.prediction,
        probability=explanation.probability,
        base_value=explanation.base_value,
        feature_contributions=explanation.feature_contributions,
        top_positive_features=[
            {"feature": f, "contribution": v}
            for f, v in explanation.top_positive
        ],
        top_negative_features=[
            {"feature": f, "contribution": v}
            for f, v in explanation.top_negative
        ]
    )


@production_router.get("/explain/{project_id}/global", response_model=GlobalExplanationResponse)
async def get_global_explanation(
    project_id: str,
    version_id: str = None,
    user_id: str = "default",
    session: Dict = Depends(get_session)
):
    """
    Get global model explanation with feature importances.
    """
    from services.model_manager import ModelManager
    from services.explainability import ExplainabilityService
    
    manager = ModelManager()
    
    try:
        model, metadata, _ = manager.load_model(user_id, project_id, version_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model not found: {str(e)}")
    
    # Use session data or create sample data
    X = session.get("X_train")
    if X is None:
        # Create dummy data for explanation
        feature_names = metadata.get("feature_names", [])
        X = pd.DataFrame([[0] * len(feature_names)], columns=feature_names)
    
    explainer = ExplainabilityService(model)
    importances = explainer.get_feature_importance(X)
    coef_report = explainer.get_coefficient_report()
    
    return GlobalExplanationResponse(
        model_type=type(model).__name__,
        explainer_type=explainer._select_importance_method(),
        feature_importances=[
            FeatureImportanceItem(
                feature=imp.feature,
                importance=imp.importance,
                rank=imp.rank,
                std=imp.std
            )
            for imp in importances
        ],
        coefficient_report=coef_report,
        shap_available=True  # SHAP is installed
    )


# ============================================================================
# Cost Governance Endpoints
# ============================================================================

@production_router.post("/cost/estimate", response_model=CostEstimateResponse)
async def estimate_cost(
    rows: int,
    columns: int,
    model: str = "gemini-2.0-flash",
    operation: str = "cleaning"
):
    """
    Estimate cost for an AI operation.
    """
    from services.observability import get_cost_governor, TokenEstimator
    
    governor = get_cost_governor()
    estimate = governor.estimate_and_check((rows, columns), model, operation)
    
    return CostEstimateResponse(
        operation=estimate.operation,
        estimated_tokens=estimate.estimated_tokens,
        estimated_cost_usd=estimate.estimated_cost_usd,
        model_used=estimate.model_used,
        within_budget=estimate.within_budget,
        warnings=estimate.warnings
    )


@production_router.get("/cost/daily", response_model=DailySpendResponse)
async def get_daily_spend():
    """
    Get today's spending report.
    """
    from services.observability import get_cost_governor
    
    governor = get_cost_governor()
    
    return DailySpendResponse(
        date=datetime.now().strftime("%Y-%m-%d"),
        total_spent_usd=governor.get_daily_spend(),
        remaining_budget_usd=governor.get_remaining_budget(),
        daily_limit_usd=governor.daily_budget_usd
    )


@production_router.post("/cost/check-limits", response_model=DatasetLimitsResponse)
async def check_dataset_limits(rows: int, columns: int):
    """
    Check if dataset is within size limits.
    """
    from services.observability import get_cost_governor
    
    governor = get_cost_governor()
    within_limits, violations = governor.check_dataset_limits(rows, columns)
    
    return DatasetLimitsResponse(
        rows=rows,
        columns=columns,
        max_rows=governor.max_rows,
        max_columns=governor.max_columns,
        within_limits=within_limits,
        violations=violations
    )


# ============================================================================
# Unified Pipeline Export
# ============================================================================

@production_router.post("/export/unified-pipeline", response_model=UnifiedPipelineExportResponse)
async def export_unified_pipeline(
    request: ExportUnifiedPipelineRequest,
    user_id: str = "default",
    session: Dict = Depends(get_session)
):
    """
    Export a complete unified pipeline (preprocessing + model) as a single file.
    
    The exported pipeline can be loaded and used outside this platform with:
    
    ```python
    from services.unified_pipeline import UnifiedPipeline
    pipeline = UnifiedPipeline.load("pipeline.pkl")
    predictions = pipeline.predict(new_data)
    ```
    """
    from services.model_manager import ModelManager
    from services.unified_pipeline import UnifiedPipeline
    from pathlib import Path
    
    manager = ModelManager()
    
    try:
        model, metadata, preprocessing = manager.load_model(
            user_id, request.project_id, request.version_id
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model not found: {str(e)}")
    
    # Determine problem type
    problem_type = metadata.get("problem_type", "unknown")
    if problem_type == "unknown" and session.get("problem_type"):
        problem_type = session["problem_type"]
    
    # Create unified pipeline
    unified = UnifiedPipeline.from_components(
        preprocessing_pipeline=preprocessing,
        model=model,
        feature_names=metadata.get("feature_names", []),
        target_name=metadata.get("target_name", "target"),
        problem_type=problem_type,
        metrics=metadata.get("metrics", {})
    )
    
    # Export
    export_dir = Path("exports")
    export_path = export_dir / request.output_filename
    unified.save(export_path)
    
    return UnifiedPipelineExportResponse(
        success=True,
        export_path=str(export_path.absolute()),
        pipeline_info=unified.get_info(),
        message="Pipeline exported successfully. Can be loaded with UnifiedPipeline.load()"
    )


@production_router.get("/system/logs")
async def get_operation_logs(
    operation_type: str = None,
    level: str = None,
    limit: int = 100
):
    """
    Get system operation logs.
    """
    from services.observability import get_operation_logger, OperationType, LogLevel
    
    op_logger = get_operation_logger()
    
    op_type = OperationType(operation_type) if operation_type else None
    log_level = LogLevel(level) if level else None
    
    logs = op_logger.get_logs(
        operation_type=op_type,
        level=log_level,
        limit=limit
    )
    
    return {
        "logs": [l.to_dict() for l in logs],
        "total": len(logs)
    }


@production_router.get("/system/stats")
async def get_system_stats():
    """
    Get system operation statistics.
    """
    from services.observability import get_operation_logger
    from services.async_training import get_async_service
    
    op_logger = get_operation_logger()
    async_service = get_async_service()
    
    return {
        "operation_stats": op_logger.get_operation_stats(),
        "active_jobs": len(async_service.tracker.list_jobs()),
        "timestamp": datetime.now().isoformat()
    }
