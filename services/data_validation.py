"""
Data Validation Service - Schema & Drift Protection

Production-grade data validation for ML inference:
- Schema validation (columns, dtypes)
- Missing column detection
- Unseen categorical value detection  
- Null ratio explosion detection
- Data type mismatch detection
- Training vs inference data comparison
"""

import logging
from enum import Enum
from typing import Optional, Dict, Any, List, Set, Union
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationIssueType(Enum):
    """Types of validation issues."""
    MISSING_COLUMN = "missing_column"
    EXTRA_COLUMN = "extra_column"
    DTYPE_MISMATCH = "dtype_mismatch"
    UNSEEN_CATEGORY = "unseen_category"
    NULL_EXPLOSION = "null_explosion"
    VALUE_OUT_OF_RANGE = "value_out_of_range"
    SCHEMA_VIOLATION = "schema_violation"


@dataclass
class ValidationIssue:
    """A single validation issue."""
    issue_type: ValidationIssueType
    severity: ValidationSeverity
    column: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    issues: List[ValidationIssue]
    schema_match: bool
    warnings_count: int
    errors_count: int
    validated_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "issues": [
                {
                    "type": i.issue_type.value,
                    "severity": i.severity.value,
                    "column": i.column,
                    "message": i.message,
                    "details": i.details
                }
                for i in self.issues
            ],
            "schema_match": self.schema_match,
            "warnings_count": self.warnings_count,
            "errors_count": self.errors_count,
            "validated_at": self.validated_at
        }


@dataclass
class DataSchema:
    """Schema definition for a dataset."""
    columns: List[str]
    dtypes: Dict[str, str]
    categorical_values: Dict[str, List[Any]]
    numeric_ranges: Dict[str, Dict[str, float]]  # {col: {min, max, mean, std}}
    null_ratios: Dict[str, float]
    created_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "columns": self.columns,
            "dtypes": self.dtypes,
            "categorical_values": self.categorical_values,
            "numeric_ranges": self.numeric_ranges,
            "null_ratios": self.null_ratios,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataSchema":
        return cls(
            columns=data["columns"],
            dtypes=data["dtypes"],
            categorical_values=data["categorical_values"],
            numeric_ranges=data["numeric_ranges"],
            null_ratios=data["null_ratios"],
            created_at=data["created_at"]
        )


class DataValidator:
    """
    Production-grade data validation service.
    
    Validates inference data against training data schema to catch:
    - Schema mismatches
    - Data drift
    - Invalid values
    """
    
    # Thresholds
    NULL_EXPLOSION_THRESHOLD = 2.0  # 2x increase in null ratio
    RANGE_EXTENSION_THRESHOLD = 3.0  # 3 std deviations
    MAX_UNSEEN_CATEGORIES_RATIO = 0.1  # 10% unseen values triggers warning
    
    def __init__(self, schema: Optional[DataSchema] = None):
        """
        Initialize validator with optional schema.
        
        Args:
            schema: Pre-defined data schema
        """
        self.schema = schema
        logger.info("DataValidator initialized")
    
    def learn_schema(
        self, 
        df: pd.DataFrame,
        categorical_columns: Optional[List[str]] = None
    ) -> DataSchema:
        """
        Learn data schema from training data.
        
        Args:
            df: Training DataFrame
            categorical_columns: Explicitly categorical columns
            
        Returns:
            Learned DataSchema
        """
        columns = df.columns.tolist()
        dtypes = {col: str(df[col].dtype) for col in columns}
        
        # Detect categorical columns
        if categorical_columns is None:
            categorical_columns = [
                col for col in columns 
                if df[col].dtype == 'object' or df[col].nunique() < 50
            ]
        
        # Store categorical values
        categorical_values = {}
        for col in categorical_columns:
            if col in df.columns:
                categorical_values[col] = df[col].dropna().unique().tolist()
        
        # Store numeric ranges
        numeric_ranges = {}
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_ranges[col] = {
                    "min": float(df[col].min()) if not df[col].isna().all() else 0,
                    "max": float(df[col].max()) if not df[col].isna().all() else 0,
                    "mean": float(df[col].mean()) if not df[col].isna().all() else 0,
                    "std": float(df[col].std()) if not df[col].isna().all() else 0
                }
        
        # Store null ratios
        null_ratios = {col: float(df[col].isna().mean()) for col in columns}
        
        self.schema = DataSchema(
            columns=columns,
            dtypes=dtypes,
            categorical_values=categorical_values,
            numeric_ranges=numeric_ranges,
            null_ratios=null_ratios,
            created_at=datetime.now().isoformat()
        )
        
        logger.info(f"Schema learned: {len(columns)} columns, {len(categorical_columns)} categorical")
        
        return self.schema
    
    def validate(
        self, 
        df: pd.DataFrame,
        strict: bool = False
    ) -> ValidationResult:
        """
        Validate data against learned schema.
        
        Args:
            df: DataFrame to validate
            strict: If True, treat warnings as errors
            
        Returns:
            ValidationResult with issues found
        """
        if self.schema is None:
            raise ValueError("No schema available. Call learn_schema first or provide schema.")
        
        issues: List[ValidationIssue] = []
        
        # Check for missing columns
        missing_cols = set(self.schema.columns) - set(df.columns)
        for col in missing_cols:
            issues.append(ValidationIssue(
                issue_type=ValidationIssueType.MISSING_COLUMN,
                severity=ValidationSeverity.ERROR,
                column=col,
                message=f"Required column '{col}' is missing",
                details={"expected_dtype": self.schema.dtypes.get(col)}
            ))
        
        # Check for extra columns
        extra_cols = set(df.columns) - set(self.schema.columns)
        for col in extra_cols:
            issues.append(ValidationIssue(
                issue_type=ValidationIssueType.EXTRA_COLUMN,
                severity=ValidationSeverity.INFO,
                column=col,
                message=f"Extra column '{col}' not in training schema",
                details={"action": "will_be_ignored"}
            ))
        
        # Check dtype mismatches
        for col in df.columns:
            if col in self.schema.dtypes:
                expected_dtype = self.schema.dtypes[col]
                actual_dtype = str(df[col].dtype)
                
                # Allow compatible dtypes
                if not self._dtypes_compatible(expected_dtype, actual_dtype):
                    issues.append(ValidationIssue(
                        issue_type=ValidationIssueType.DTYPE_MISMATCH,
                        severity=ValidationSeverity.WARNING,
                        column=col,
                        message=f"Column '{col}' dtype mismatch: expected {expected_dtype}, got {actual_dtype}",
                        details={
                            "expected": expected_dtype,
                            "actual": actual_dtype
                        }
                    ))
        
        # Check for unseen categorical values
        for col, known_values in self.schema.categorical_values.items():
            if col in df.columns:
                actual_values = set(df[col].dropna().unique())
                known_set = set(known_values)
                unseen = actual_values - known_set
                
                if unseen:
                    unseen_ratio = len(unseen) / max(len(actual_values), 1)
                    severity = (
                        ValidationSeverity.WARNING 
                        if unseen_ratio < self.MAX_UNSEEN_CATEGORIES_RATIO 
                        else ValidationSeverity.ERROR
                    )
                    
                    issues.append(ValidationIssue(
                        issue_type=ValidationIssueType.UNSEEN_CATEGORY,
                        severity=severity,
                        column=col,
                        message=f"Column '{col}' has {len(unseen)} unseen categorical values",
                        details={
                            "unseen_count": len(unseen),
                            "unseen_samples": list(unseen)[:10],
                            "unseen_ratio": round(unseen_ratio, 4)
                        }
                    ))
        
        # Check for null explosion
        for col, train_null_ratio in self.schema.null_ratios.items():
            if col in df.columns:
                current_null_ratio = df[col].isna().mean()
                
                if train_null_ratio > 0:
                    ratio_increase = current_null_ratio / max(train_null_ratio, 0.001)
                else:
                    ratio_increase = current_null_ratio * 100 if current_null_ratio > 0 else 0
                
                if ratio_increase > self.NULL_EXPLOSION_THRESHOLD:
                    issues.append(ValidationIssue(
                        issue_type=ValidationIssueType.NULL_EXPLOSION,
                        severity=ValidationSeverity.WARNING,
                        column=col,
                        message=f"Column '{col}' null ratio increased {ratio_increase:.1f}x",
                        details={
                            "training_null_ratio": round(train_null_ratio, 4),
                            "current_null_ratio": round(current_null_ratio, 4),
                            "increase_factor": round(ratio_increase, 2)
                        }
                    ))
        
        # Check for values out of range
        for col, ranges in self.schema.numeric_ranges.items():
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                col_data = df[col].dropna()
                
                if len(col_data) > 0:
                    col_min = col_data.min()
                    col_max = col_data.max()
                    
                    train_min = ranges["min"]
                    train_max = ranges["max"]
                    train_std = ranges.get("std", 0)
                    
                    # Check if values exceed range by more than threshold
                    lower_bound = train_min - self.RANGE_EXTENSION_THRESHOLD * train_std
                    upper_bound = train_max + self.RANGE_EXTENSION_THRESHOLD * train_std
                    
                    if col_min < lower_bound or col_max > upper_bound:
                        issues.append(ValidationIssue(
                            issue_type=ValidationIssueType.VALUE_OUT_OF_RANGE,
                            severity=ValidationSeverity.WARNING,
                            column=col,
                            message=f"Column '{col}' values outside training range",
                            details={
                                "training_range": [train_min, train_max],
                                "current_range": [float(col_min), float(col_max)],
                                "acceptable_range": [lower_bound, upper_bound]
                            }
                        ))
        
        # Calculate summary
        errors_count = sum(1 for i in issues if i.severity == ValidationSeverity.ERROR)
        warnings_count = sum(1 for i in issues if i.severity == ValidationSeverity.WARNING)
        
        is_valid = errors_count == 0
        if strict:
            is_valid = errors_count == 0 and warnings_count == 0
        
        schema_match = len(missing_cols) == 0
        
        result = ValidationResult(
            is_valid=is_valid,
            issues=issues,
            schema_match=schema_match,
            warnings_count=warnings_count,
            errors_count=errors_count,
            validated_at=datetime.now().isoformat()
        )
        
        logger.info(
            f"Validation complete: valid={is_valid}, errors={errors_count}, warnings={warnings_count}"
        )
        
        return result
    
    def _dtypes_compatible(self, expected: str, actual: str) -> bool:
        """Check if two dtypes are compatible."""
        # Numeric compatibility
        numeric_types = {'int64', 'int32', 'float64', 'float32', 'int', 'float'}
        if expected in numeric_types and actual in numeric_types:
            return True
        
        # Object/string compatibility
        string_types = {'object', 'string', 'str'}
        if expected in string_types and actual in string_types:
            return True
        
        return expected == actual
    
    def save_schema(self, path: Union[str, Path]) -> None:
        """Save schema to disk."""
        if self.schema is None:
            raise ValueError("No schema to save")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(self.schema.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Schema saved to {path}")
    
    @classmethod
    def load_schema(cls, path: Union[str, Path]) -> "DataValidator":
        """Load validator with schema from disk."""
        path = Path(path)
        
        with open(path, "r") as f:
            data = json.load(f)
        
        schema = DataSchema.from_dict(data)
        
        validator = cls(schema)
        logger.info(f"Schema loaded from {path}")
        
        return validator
    
    def get_validation_summary(self, result: ValidationResult) -> str:
        """Get human-readable validation summary."""
        lines = [
            f"Data Validation Summary ({result.validated_at})",
            "=" * 50,
            f"Status: {'✅ VALID' if result.is_valid else '❌ INVALID'}",
            f"Schema Match: {'✅' if result.schema_match else '❌'}",
            f"Errors: {result.errors_count}",
            f"Warnings: {result.warnings_count}",
            ""
        ]
        
        if result.issues:
            lines.append("Issues:")
            for issue in result.issues:
                icon = {
                    ValidationSeverity.INFO: "ℹ️",
                    ValidationSeverity.WARNING: "⚠️",
                    ValidationSeverity.ERROR: "❌",
                    ValidationSeverity.CRITICAL: "🔴"
                }.get(issue.severity, "•")
                
                lines.append(f"  {icon} [{issue.column}] {issue.message}")
        
        return "\n".join(lines)


class DataDriftDetector:
    """
    Detect statistical drift between training and inference data.
    
    Uses statistical tests to determine if data distribution has shifted.
    """
    
    DRIFT_THRESHOLD = 0.05  # p-value threshold for drift detection
    
    def __init__(self, reference_data: Optional[pd.DataFrame] = None):
        """Initialize with optional reference data."""
        self.reference_stats: Dict[str, Dict[str, float]] = {}
        
        if reference_data is not None:
            self.fit(reference_data)
    
    def fit(self, df: pd.DataFrame) -> None:
        """Learn reference statistics from training data."""
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                self.reference_stats[col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "median": float(df[col].median()),
                    "q1": float(df[col].quantile(0.25)),
                    "q3": float(df[col].quantile(0.75))
                }
        
        logger.info(f"Reference statistics computed for {len(self.reference_stats)} columns")
    
    def detect_drift(
        self, 
        df: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """
        Detect drift in new data compared to reference.
        
        Returns:
            Dict with drift information per column
        """
        drift_report = {}
        
        for col, ref_stats in self.reference_stats.items():
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            current_mean = df[col].mean()
            current_std = df[col].std()
            
            # Simple drift detection using z-score of mean difference
            if ref_stats["std"] > 0:
                mean_drift_zscore = abs(current_mean - ref_stats["mean"]) / ref_stats["std"]
            else:
                mean_drift_zscore = 0
            
            std_ratio = current_std / max(ref_stats["std"], 0.001)
            
            has_drift = mean_drift_zscore > 2 or std_ratio > 2 or std_ratio < 0.5
            
            drift_report[col] = {
                "has_drift": has_drift,
                "mean_drift_zscore": round(float(mean_drift_zscore), 4),
                "std_ratio": round(float(std_ratio), 4),
                "reference_mean": ref_stats["mean"],
                "current_mean": round(float(current_mean), 4),
                "reference_std": ref_stats["std"],
                "current_std": round(float(current_std), 4)
            }
        
        drifted_cols = [col for col, info in drift_report.items() if info["has_drift"]]
        logger.info(f"Drift detection: {len(drifted_cols)}/{len(drift_report)} columns show drift")
        
        return drift_report
