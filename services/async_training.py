"""
Async Training Service - Background Job Processing

Production-grade async training with:
- Background worker execution
- Job ID generation and tracking
- Status monitoring
- Training logs and metrics storage
- Timeout handling
"""

import logging
import threading
import uuid
import time
import traceback
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future
import pandas as pd

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Status of a training job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class TrainingLog:
    """A single log entry during training."""
    timestamp: str
    level: str
    message: str
    metrics: Optional[Dict[str, float]] = None


@dataclass
class TrainingJob:
    """Represents an async training job."""
    job_id: str
    status: JobStatus
    algorithm: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: float = 0.0  # 0.0 to 1.0
    current_step: str = ""
    logs: List[TrainingLog] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "algorithm": self.algorithm,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "progress": self.progress,
            "current_step": self.current_step,
            "logs": [
                {
                    "timestamp": l.timestamp,
                    "level": l.level,
                    "message": l.message,
                    "metrics": l.metrics
                }
                for l in self.logs
            ],
            "result": self.result,
            "error": self.error,
            "metrics": self.metrics,
            "config": self.config
        }


class JobTracker:
    """
    Thread-safe job tracking for async operations.
    
    Provides:
    - Job status updates
    - Progress tracking
    - Log collection
    - Metrics aggregation
    """
    
    def __init__(self):
        self._jobs: Dict[str, TrainingJob] = {}
        self._lock = threading.Lock()
    
    def create_job(self, algorithm: str, config: Dict[str, Any] = None) -> str:
        """Create a new job and return its ID."""
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        
        job = TrainingJob(
            job_id=job_id,
            status=JobStatus.PENDING,
            algorithm=algorithm,
            config=config or {}
        )
        
        with self._lock:
            self._jobs[job_id] = job
        
        logger.info(f"Created job: {job_id} for algorithm: {algorithm}")
        return job_id
    
    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get job by ID."""
        with self._lock:
            return self._jobs.get(job_id)
    
    def update_status(self, job_id: str, status: JobStatus) -> None:
        """Update job status."""
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.status = status
                
                if status == JobStatus.RUNNING and job.started_at is None:
                    job.started_at = datetime.now().isoformat()
                elif status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.TIMEOUT]:
                    job.completed_at = datetime.now().isoformat()
    
    def update_progress(self, job_id: str, progress: float, step: str = "") -> None:
        """Update job progress (0.0 to 1.0)."""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].progress = min(max(progress, 0.0), 1.0)
                if step:
                    self._jobs[job_id].current_step = step
    
    def add_log(
        self, 
        job_id: str, 
        message: str, 
        level: str = "INFO",
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Add a log entry to job."""
        log_entry = TrainingLog(
            timestamp=datetime.now().isoformat(),
            level=level,
            message=message,
            metrics=metrics
        )
        
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].logs.append(log_entry)
                if metrics:
                    self._jobs[job_id].metrics.update(metrics)
    
    def set_result(self, job_id: str, result: Dict[str, Any]) -> None:
        """Set job result on completion."""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].result = result
    
    def set_error(self, job_id: str, error: str) -> None:
        """Set job error on failure."""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].error = error
    
    def list_jobs(
        self, 
        status: Optional[JobStatus] = None,
        limit: int = 50
    ) -> List[TrainingJob]:
        """List jobs, optionally filtered by status."""
        with self._lock:
            jobs = list(self._jobs.values())
        
        if status:
            jobs = [j for j in jobs if j.status == status]
        
        # Sort by most recent first
        jobs.sort(key=lambda j: j.started_at or "", reverse=True)
        
        return jobs[:limit]
    
    def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """Remove jobs older than max_age_hours."""
        cutoff = datetime.now().timestamp() - (max_age_hours * 3600)
        removed = 0
        
        with self._lock:
            to_remove = []
            for job_id, job in self._jobs.items():
                if job.completed_at:
                    try:
                        completed_time = datetime.fromisoformat(job.completed_at).timestamp()
                        if completed_time < cutoff:
                            to_remove.append(job_id)
                    except:
                        pass
            
            for job_id in to_remove:
                del self._jobs[job_id]
                removed += 1
        
        if removed > 0:
            logger.info(f"Cleaned up {removed} old jobs")
        
        return removed


class AsyncTrainingService:
    """
    Async training execution service.
    
    Handles:
    - Background training jobs
    - Progress callbacks
    - Timeout management
    - Resource cleanup
    """
    
    # Default settings
    MAX_WORKERS = 4
    DEFAULT_TIMEOUT = 3600  # 1 hour
    
    def __init__(self, max_workers: int = None):
        """
        Initialize async training service.
        
        Args:
            max_workers: Maximum concurrent training jobs
        """
        self.max_workers = max_workers or self.MAX_WORKERS
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._tracker = JobTracker()
        self._futures: Dict[str, Future] = {}
        
        logger.info(f"AsyncTrainingService initialized with {self.max_workers} workers")
    
    @property
    def tracker(self) -> JobTracker:
        """Access job tracker."""
        return self._tracker
    
    def submit_training(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        algorithm: str,
        config: Dict[str, Any] = None,
        timeout: int = None
    ) -> str:
        """
        Submit a training job for async execution.
        
        Args:
            X_train: Training features
            y_train: Training target
            algorithm: Algorithm name
            config: Training configuration
            timeout: Max seconds for training
            
        Returns:
            Job ID for tracking
        """
        timeout = timeout or self.DEFAULT_TIMEOUT
        config = config or {}
        
        # Create job
        job_id = self._tracker.create_job(algorithm, config)
        
        # Submit to executor
        future = self._executor.submit(
            self._run_training,
            job_id,
            X_train.copy(),
            y_train.copy(),
            algorithm,
            config,
            timeout
        )
        
        self._futures[job_id] = future
        
        logger.info(f"Submitted training job: {job_id}")
        return job_id
    
    def _run_training(
        self,
        job_id: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        algorithm: str,
        config: Dict[str, Any],
        timeout: int
    ) -> None:
        """Execute training in background thread."""
        start_time = time.time()
        
        try:
            self._tracker.update_status(job_id, JobStatus.RUNNING)
            self._tracker.add_log(job_id, f"Starting training with {algorithm}")
            self._tracker.update_progress(job_id, 0.1, "Initializing")
            
            # Import ML pipeline
            from services.ml_pipeline import MLPipeline, AlgorithmType, TrainingConfig
            
            # Map algorithm name to enum
            algo_map = {
                "random_forest_clf": AlgorithmType.RANDOM_FOREST_CLF,
                "random_forest_reg": AlgorithmType.RANDOM_FOREST_REG,
                "xgboost_clf": AlgorithmType.XGBOOST_CLF,
                "xgboost_reg": AlgorithmType.XGBOOST_REG,
                "logistic_regression": AlgorithmType.LOGISTIC_REGRESSION,
                "linear_regression": AlgorithmType.LINEAR_REGRESSION,
                "ridge": AlgorithmType.RIDGE,
                "lasso": AlgorithmType.LASSO,
                "svm_clf": AlgorithmType.SVM_CLF,
            }
            
            algo_type = algo_map.get(algorithm.lower())
            if algo_type is None:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            self._tracker.update_progress(job_id, 0.2, "Creating pipeline")
            self._tracker.add_log(job_id, "Pipeline initialized")
            
            # Create training config
            training_config = TrainingConfig(
                algorithm=algo_type,
                hyperparameters=config.get("hyperparameters", {}),
                cv_folds=config.get("cv_folds", 5)
            )
            
            # Check timeout
            if time.time() - start_time > timeout:
                raise TimeoutError("Training timeout exceeded")
            
            self._tracker.update_progress(job_id, 0.3, "Training model")
            self._tracker.add_log(job_id, "Starting model training")
            
            # Train
            pipeline = MLPipeline()
            result = pipeline.train(X_train, y_train, training_config)
            
            # Check for timeout during training
            elapsed = time.time() - start_time
            if elapsed > timeout:
                self._tracker.update_status(job_id, JobStatus.TIMEOUT)
                self._tracker.add_log(job_id, f"Training timed out after {elapsed:.0f}s", "ERROR")
                return
            
            self._tracker.update_progress(job_id, 0.9, "Finalizing")
            self._tracker.add_log(
                job_id, 
                f"Training completed: CV={result.cv_mean:.4f}",
                metrics={"cv_mean": result.cv_mean, "cv_std": result.cv_std}
            )
            
            # Store result
            self._tracker.set_result(job_id, {
                "algorithm": result.algorithm,
                "cv_mean": result.cv_mean,
                "cv_std": result.cv_std,
                "cv_scores": result.cv_scores,
                "training_time_seconds": result.training_time_seconds,
                "feature_importances": result.feature_importances,
                "success": result.success
            })
            
            self._tracker.update_progress(job_id, 1.0, "Complete")
            self._tracker.update_status(job_id, JobStatus.COMPLETED)
            self._tracker.add_log(job_id, "Job completed successfully")
            
            logger.info(f"Job {job_id} completed successfully in {elapsed:.1f}s")
            
        except TimeoutError as e:
            self._tracker.update_status(job_id, JobStatus.TIMEOUT)
            self._tracker.set_error(job_id, str(e))
            self._tracker.add_log(job_id, f"Timeout: {str(e)}", "ERROR")
            logger.warning(f"Job {job_id} timed out")
            
        except Exception as e:
            self._tracker.update_status(job_id, JobStatus.FAILED)
            self._tracker.set_error(job_id, str(e))
            self._tracker.add_log(job_id, f"Error: {str(e)}", "ERROR")
            self._tracker.add_log(job_id, traceback.format_exc(), "DEBUG")
            logger.error(f"Job {job_id} failed: {str(e)}")
    
    def submit_comparison(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        algorithms: List[str],
        config: Dict[str, Any] = None,
        timeout: int = None
    ) -> str:
        """
        Submit a model comparison job.
        
        Args:
            X_train: Training features
            y_train: Training target
            algorithms: List of algorithms to compare
            config: Training configuration
            timeout: Max seconds for training
            
        Returns:
            Job ID for tracking
        """
        timeout = timeout or self.DEFAULT_TIMEOUT * 2  # Double for comparison
        config = config or {}
        config["algorithms"] = algorithms
        
        # Create job
        job_id = self._tracker.create_job("comparison", config)
        
        # Submit to executor
        future = self._executor.submit(
            self._run_comparison,
            job_id,
            X_train.copy(),
            y_train.copy(),
            algorithms,
            config,
            timeout
        )
        
        self._futures[job_id] = future
        
        logger.info(f"Submitted comparison job: {job_id} with {len(algorithms)} algorithms")
        return job_id
    
    def _run_comparison(
        self,
        job_id: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        algorithms: List[str],
        config: Dict[str, Any],
        timeout: int
    ) -> None:
        """Execute model comparison in background."""
        start_time = time.time()
        
        try:
            self._tracker.update_status(job_id, JobStatus.RUNNING)
            self._tracker.add_log(job_id, f"Starting comparison with {len(algorithms)} algorithms")
            
            from services.ml_pipeline import MLPipeline, AlgorithmType
            
            algo_map = {
                "random_forest_clf": AlgorithmType.RANDOM_FOREST_CLF,
                "random_forest_reg": AlgorithmType.RANDOM_FOREST_REG,
                "xgboost_clf": AlgorithmType.XGBOOST_CLF,
                "xgboost_reg": AlgorithmType.XGBOOST_REG,
                "logistic_regression": AlgorithmType.LOGISTIC_REGRESSION,
                "linear_regression": AlgorithmType.LINEAR_REGRESSION,
                "ridge": AlgorithmType.RIDGE,
                "lasso": AlgorithmType.LASSO,
                "svm_clf": AlgorithmType.SVM_CLF,
            }
            
            algo_types = [algo_map[a.lower()] for a in algorithms if a.lower() in algo_map]
            
            pipeline = MLPipeline()
            results = []
            
            for i, algo in enumerate(algo_types):
                if time.time() - start_time > timeout:
                    self._tracker.update_status(job_id, JobStatus.TIMEOUT)
                    return
                
                progress = (i + 1) / len(algo_types)
                self._tracker.update_progress(job_id, progress * 0.9, f"Training {algo.value}")
                self._tracker.add_log(job_id, f"Training {algo.value}")
                
                try:
                    result = pipeline.train(X_train, y_train, algo)
                    results.append({
                        "algorithm": result.algorithm,
                        "cv_mean": result.cv_mean,
                        "cv_std": result.cv_std,
                        "success": result.success
                    })
                    
                    self._tracker.add_log(
                        job_id,
                        f"{algo.value}: CV={result.cv_mean:.4f}",
                        metrics={f"{algo.value}_cv": result.cv_mean}
                    )
                except Exception as e:
                    results.append({
                        "algorithm": algo.value,
                        "cv_mean": 0,
                        "cv_std": 0,
                        "success": False,
                        "error": str(e)
                    })
            
            # Get leaderboard
            leaderboard = pipeline.get_leaderboard()
            
            self._tracker.set_result(job_id, {
                "results": results,
                "leaderboard": leaderboard,
                "best_model": pipeline._best_model.algorithm if pipeline._best_model else None
            })
            
            self._tracker.update_progress(job_id, 1.0, "Complete")
            self._tracker.update_status(job_id, JobStatus.COMPLETED)
            self._tracker.add_log(job_id, "Comparison completed")
            
        except Exception as e:
            self._tracker.update_status(job_id, JobStatus.FAILED)
            self._tracker.set_error(job_id, str(e))
            self._tracker.add_log(job_id, f"Error: {str(e)}", "ERROR")
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a job."""
        job = self._tracker.get_job(job_id)
        return job.to_dict() if job else None
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job."""
        if job_id in self._futures:
            future = self._futures[job_id]
            if future.cancel():
                self._tracker.update_status(job_id, JobStatus.CANCELLED)
                self._tracker.add_log(job_id, "Job cancelled by user", "WARNING")
                logger.info(f"Job {job_id} cancelled")
                return True
        return False
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor."""
        self._executor.shutdown(wait=wait)
        logger.info("AsyncTrainingService shutdown complete")


# Global instance
_async_service: Optional[AsyncTrainingService] = None


def get_async_service() -> AsyncTrainingService:
    """Get or create global async training service."""
    global _async_service
    if _async_service is None:
        _async_service = AsyncTrainingService()
    return _async_service
