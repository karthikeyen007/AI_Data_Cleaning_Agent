"""
Observability & Cost Governance Service

Production-grade monitoring and cost control:
- Structured logging for all operations
- Token estimation for AI cleaning
- Dataset size guards
- Cost tracking and budgets
- Operation metrics collection
- Resource usage monitoring
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from functools import wraps
import json
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Types of tracked operations."""
    DATA_CLEANING = "data_cleaning"
    MODEL_TRAINING = "model_training"
    PREDICTION = "prediction"
    PREPROCESSING = "preprocessing"
    TUNING = "tuning"
    EXPORT = "export"
    API_CALL = "api_call"


class LogLevel(Enum):
    """Log levels for structured logging."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class OperationLog:
    """Structured log entry for an operation."""
    timestamp: str
    operation_type: str
    level: str
    message: str
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "operation_type": self.operation_type,
            "level": self.level,
            "message": self.message,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "error": self.error,
            "user_id": self.user_id,
            "session_id": self.session_id
        }


@dataclass
class CostEstimate:
    """Cost estimate for an operation."""
    operation: str
    estimated_tokens: int
    estimated_cost_usd: float
    model_used: str
    within_budget: bool
    warnings: List[str] = field(default_factory=list)


@dataclass
class ResourceUsage:
    """Resource usage metrics."""
    memory_mb: float
    cpu_percent: float
    disk_io_mb: float
    timestamp: str


class TokenEstimator:
    """
    Estimate tokens for AI operations.
    
    Uses heuristics based on data size and content.
    """
    
    # Average characters per token (OpenAI tokenizer approximation)
    CHARS_PER_TOKEN = 4
    
    # Model-specific multipliers
    MODEL_MULTIPLIERS = {
        "gpt-4": 1.0,
        "gpt-4.1-nano": 0.8,
        "gpt-5-mini": 1.2,
        "gemini-2.5-pro": 0.9,
        "gemini-2.0-flash": 0.7,
        "default": 1.0
    }
    
    # Cost per 1K tokens (input + output estimated)
    MODEL_COSTS = {
        "gpt-4": 0.06,           # $30/1M input + $60/1M output avg
        "gpt-4.1-nano": 0.002,   # Cheaper nano model
        "gpt-5-mini": 0.01,      # Mid-tier
        "gemini-2.5-pro": 0.005, # Google pricing
        "gemini-2.0-flash": 0.001, # Flash is cheap
        "default": 0.01
    }
    
    @classmethod
    def estimate_dataframe_tokens(
        cls, 
        df_shape: tuple,
        avg_cell_length: int = 20
    ) -> int:
        """
        Estimate tokens for a DataFrame.
        
        Args:
            df_shape: (rows, columns)
            avg_cell_length: Average characters per cell
            
        Returns:
            Estimated token count
        """
        rows, cols = df_shape
        
        # Estimate characters
        total_chars = rows * cols * avg_cell_length
        
        # Add overhead for formatting (CSV, prompts, etc.)
        overhead = rows * 10  # ~10 chars per row for delimiters
        prompt_overhead = 500  # Fixed prompt overhead
        
        total_chars += overhead + prompt_overhead
        
        # Convert to tokens
        tokens = total_chars // cls.CHARS_PER_TOKEN
        
        return tokens
    
    @classmethod
    def estimate_cost(
        cls,
        tokens: int,
        model: str,
        include_output: bool = True
    ) -> float:
        """
        Estimate cost in USD.
        
        Args:
            tokens: Input token count
            model: Model name
            include_output: Include estimated output tokens
            
        Returns:
            Estimated cost in USD
        """
        # Normalize model name
        model_key = model.lower()
        for key in cls.MODEL_COSTS:
            if key in model_key:
                model_key = key
                break
        else:
            model_key = "default"
        
        cost_per_k = cls.MODEL_COSTS.get(model_key, cls.MODEL_COSTS["default"])
        
        # Estimate input cost
        input_cost = (tokens / 1000) * cost_per_k
        
        # Add output cost (typically 2x input for cleaning tasks)
        if include_output:
            output_tokens = tokens * 1.5  # Assume 1.5x output
            output_cost = (output_tokens / 1000) * cost_per_k * 2  # Output usually 2x
            total_cost = input_cost + output_cost
        else:
            total_cost = input_cost
        
        return round(total_cost, 6)


class CostGovernor:
    """
    Cost control and budget management.
    
    Enforces:
    - Dataset size limits
    - Token budgets
    - Cost caps
    - Model routing based on cost
    """
    
    # Default limits
    DEFAULT_MAX_ROWS = 100000
    DEFAULT_MAX_COLUMNS = 500
    DEFAULT_MAX_TOKENS_PER_REQUEST = 500000
    DEFAULT_DAILY_BUDGET_USD = 10.0
    
    def __init__(
        self,
        max_rows: int = None,
        max_columns: int = None,
        max_tokens_per_request: int = None,
        daily_budget_usd: float = None
    ):
        """
        Initialize cost governor.
        
        Args:
            max_rows: Maximum rows per dataset
            max_columns: Maximum columns per dataset
            max_tokens_per_request: Token limit per request
            daily_budget_usd: Daily spending limit
        """
        self.max_rows = max_rows or self.DEFAULT_MAX_ROWS
        self.max_columns = max_columns or self.DEFAULT_MAX_COLUMNS
        self.max_tokens_per_request = max_tokens_per_request or self.DEFAULT_MAX_TOKENS_PER_REQUEST
        self.daily_budget_usd = daily_budget_usd or self.DEFAULT_DAILY_BUDGET_USD
        
        self._daily_spend: Dict[str, float] = {}  # date -> spend
        self._lock = threading.Lock()
        
        logger.info(f"CostGovernor initialized: max_rows={self.max_rows}, daily_budget=${self.daily_budget_usd}")
    
    def check_dataset_limits(
        self,
        rows: int,
        columns: int
    ) -> tuple[bool, List[str]]:
        """
        Check if dataset is within limits.
        
        Returns:
            (is_within_limits, list_of_violations)
        """
        violations = []
        
        if rows > self.max_rows:
            violations.append(
                f"Dataset has {rows} rows (max: {self.max_rows}). Consider sampling."
            )
        
        if columns > self.max_columns:
            violations.append(
                f"Dataset has {columns} columns (max: {self.max_columns}). Consider feature selection."
            )
        
        return len(violations) == 0, violations
    
    def estimate_and_check(
        self,
        df_shape: tuple,
        model: str,
        operation: str = "cleaning"
    ) -> CostEstimate:
        """
        Estimate cost and check against budget.
        
        Args:
            df_shape: (rows, columns)
            model: Model to use
            operation: Operation type
            
        Returns:
            CostEstimate with warnings if applicable
        """
        tokens = TokenEstimator.estimate_dataframe_tokens(df_shape)
        estimated_cost = TokenEstimator.estimate_cost(tokens, model)
        
        warnings = []
        within_budget = True
        
        # Check against limits
        is_valid, violations = self.check_dataset_limits(*df_shape)
        warnings.extend(violations)
        
        if tokens > self.max_tokens_per_request:
            warnings.append(
                f"Request would use ~{tokens:,} tokens (max: {self.max_tokens_per_request:,})"
            )
            within_budget = False
        
        # Check daily budget
        today = datetime.now().strftime("%Y-%m-%d")
        with self._lock:
            current_spend = self._daily_spend.get(today, 0)
        
        if current_spend + estimated_cost > self.daily_budget_usd:
            warnings.append(
                f"Would exceed daily budget (${current_spend:.2f} spent, "
                f"+${estimated_cost:.4f} = ${self.daily_budget_usd:.2f} limit)"
            )
            within_budget = False
        
        return CostEstimate(
            operation=operation,
            estimated_tokens=tokens,
            estimated_cost_usd=estimated_cost,
            model_used=model,
            within_budget=within_budget and is_valid,
            warnings=warnings
        )
    
    def record_spend(self, amount_usd: float) -> None:
        """Record spending for today."""
        today = datetime.now().strftime("%Y-%m-%d")
        
        with self._lock:
            current = self._daily_spend.get(today, 0)
            self._daily_spend[today] = current + amount_usd
        
        logger.info(f"Recorded spend: ${amount_usd:.4f}")
    
    def get_daily_spend(self) -> float:
        """Get total spend for today."""
        today = datetime.now().strftime("%Y-%m-%d")
        
        with self._lock:
            return self._daily_spend.get(today, 0)
    
    def get_remaining_budget(self) -> float:
        """Get remaining budget for today."""
        return max(0, self.daily_budget_usd - self.get_daily_spend())
    
    def suggest_cheaper_model(self, current_model: str) -> Optional[str]:
        """Suggest a cheaper model alternative."""
        cheaper_options = {
            "gpt-4": "gpt-4.1-nano",
            "gpt-5-mini": "gpt-4.1-nano",
            "gemini-2.5-pro": "gemini-2.0-flash",
        }
        
        for key, value in cheaper_options.items():
            if key in current_model.lower():
                return value
        
        return "gemini-2.0-flash"  # Default cheap option


class OperationLogger:
    """
    Structured logging for ML operations.
    
    Provides:
    - JSON-structured logs
    - Operation timing
    - Error tracking
    - Log aggregation
    """
    
    def __init__(self, log_path: Optional[Path] = None):
        """
        Initialize operation logger.
        
        Args:
            log_path: Path to log file (optional)
        """
        self.log_path = Path(log_path) if log_path else None
        self._logs: List[OperationLog] = []
        self._lock = threading.Lock()
        
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def log(
        self,
        operation_type: OperationType,
        message: str,
        level: LogLevel = LogLevel.INFO,
        duration_ms: float = None,
        metadata: Dict[str, Any] = None,
        error: str = None,
        user_id: str = None,
        session_id: str = None
    ) -> OperationLog:
        """
        Log an operation.
        
        Args:
            operation_type: Type of operation
            message: Log message
            level: Log level
            duration_ms: Operation duration
            metadata: Additional metadata
            error: Error message if any
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            Created OperationLog
        """
        log_entry = OperationLog(
            timestamp=datetime.now().isoformat(),
            operation_type=operation_type.value,
            level=level.value,
            message=message,
            duration_ms=duration_ms,
            metadata=metadata or {},
            error=error,
            user_id=user_id,
            session_id=session_id
        )
        
        with self._lock:
            self._logs.append(log_entry)
        
        # Also log to standard logger
        log_method = getattr(logger, level.value.lower(), logger.info)
        log_method(f"[{operation_type.value}] {message}")
        
        # Write to file if configured
        if self.log_path:
            self._write_log(log_entry)
        
        return log_entry
    
    def _write_log(self, log_entry: OperationLog) -> None:
        """Write log entry to file."""
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(log_entry.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to write log: {str(e)}")
    
    def get_logs(
        self,
        operation_type: OperationType = None,
        level: LogLevel = None,
        since: datetime = None,
        limit: int = 100
    ) -> List[OperationLog]:
        """
        Get filtered logs.
        
        Args:
            operation_type: Filter by operation type
            level: Filter by level
            since: Only logs after this time
            limit: Maximum logs to return
            
        Returns:
            List of matching OperationLog entries
        """
        with self._lock:
            logs = list(self._logs)
        
        if operation_type:
            logs = [l for l in logs if l.operation_type == operation_type.value]
        
        if level:
            logs = [l for l in logs if l.level == level.value]
        
        if since:
            since_str = since.isoformat()
            logs = [l for l in logs if l.timestamp >= since_str]
        
        return logs[-limit:]
    
    def get_error_logs(self, limit: int = 50) -> List[OperationLog]:
        """Get recent error logs."""
        return self.get_logs(level=LogLevel.ERROR, limit=limit)
    
    def get_operation_stats(self) -> Dict[str, Any]:
        """Get statistics about logged operations."""
        with self._lock:
            logs = list(self._logs)
        
        stats = {
            "total_logs": len(logs),
            "by_operation": {},
            "by_level": {},
            "avg_duration_ms": {}
        }
        
        for log in logs:
            # Count by operation
            op = log.operation_type
            stats["by_operation"][op] = stats["by_operation"].get(op, 0) + 1
            
            # Count by level
            lvl = log.level
            stats["by_level"][lvl] = stats["by_level"].get(lvl, 0) + 1
            
            # Track durations
            if log.duration_ms is not None:
                if op not in stats["avg_duration_ms"]:
                    stats["avg_duration_ms"][op] = []
                stats["avg_duration_ms"][op].append(log.duration_ms)
        
        # Calculate averages
        for op, durations in stats["avg_duration_ms"].items():
            stats["avg_duration_ms"][op] = sum(durations) / len(durations) if durations else 0
        
        return stats


def timed_operation(operation_type: OperationType):
    """
    Decorator to time and log operations.
    
    Usage:
        @timed_operation(OperationType.MODEL_TRAINING)
        def train_model(...):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            op_logger = get_operation_logger()
            
            try:
                result = func(*args, **kwargs)
                
                duration_ms = (time.time() - start_time) * 1000
                op_logger.log(
                    operation_type=operation_type,
                    message=f"{func.__name__} completed",
                    duration_ms=duration_ms
                )
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                op_logger.log(
                    operation_type=operation_type,
                    message=f"{func.__name__} failed",
                    level=LogLevel.ERROR,
                    duration_ms=duration_ms,
                    error=str(e)
                )
                raise
        
        return wrapper
    return decorator


# Global instances
_operation_logger: Optional[OperationLogger] = None
_cost_governor: Optional[CostGovernor] = None


def get_operation_logger() -> OperationLogger:
    """Get or create global operation logger."""
    global _operation_logger
    if _operation_logger is None:
        log_dir = Path(os.getenv("ML_LOGS_DIR", "logs"))
        _operation_logger = OperationLogger(log_dir / "operations.jsonl")
    return _operation_logger


def get_cost_governor() -> CostGovernor:
    """Get or create global cost governor."""
    global _cost_governor
    if _cost_governor is None:
        _cost_governor = CostGovernor(
            max_rows=int(os.getenv("MAX_DATASET_ROWS", 100000)),
            daily_budget_usd=float(os.getenv("DAILY_AI_BUDGET_USD", 10.0))
        )
    return _cost_governor
