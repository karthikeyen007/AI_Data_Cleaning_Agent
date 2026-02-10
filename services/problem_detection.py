"""
Problem Detection Service

Automatically detects ML problem type and provides algorithm suggestions:
- Classification vs Regression detection
- Target column type analysis
- Algorithm recommendation based on data characteristics
- Data requirements validation
"""

import logging
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ProblemType(Enum):
    """Types of ML problems."""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    UNKNOWN = "unknown"


class DataIssue(Enum):
    """Types of data issues that may affect training."""
    INSUFFICIENT_SAMPLES = "insufficient_samples"
    CLASS_IMBALANCE = "class_imbalance"
    HIGH_MISSING_RATIO = "high_missing_ratio"
    LOW_VARIANCE_TARGET = "low_variance_target"
    HIGH_CARDINALITY_TARGET = "high_cardinality_target"
    NO_NUMERIC_FEATURES = "no_numeric_features"
    TOO_MANY_FEATURES = "too_many_features"


@dataclass
class AlgorithmRecommendation:
    """Recommendation for an ML algorithm."""
    name: str
    algorithm_id: str
    priority: int  # Lower is higher priority
    reason: str
    suitable_for: List[str]
    hyperparameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TargetAnalysis:
    """Analysis of the target column."""
    column_name: str
    dtype: str
    unique_values: int
    missing_count: int
    missing_pct: float
    value_distribution: Dict[Any, int]
    is_numeric: bool
    is_binary: bool
    is_ordinal: bool
    suggested_type: ProblemType
    sample_values: List[Any]


@dataclass
class ProblemDetectionResult:
    """Complete result of problem detection."""
    problem_type: ProblemType
    target_analysis: TargetAnalysis
    recommended_algorithms: List[AlgorithmRecommendation]
    data_issues: List[DataIssue]
    warnings: List[str]
    dataset_stats: Dict[str, Any]
    is_trainable: bool
    min_required_samples: int


class ProblemDetector:
    """
    Automatic problem type detection and algorithm recommendation.
    
    Analyzes dataset characteristics to determine:
    - Classification vs Regression
    - Binary vs Multiclass
    - Suitable algorithms
    - Data quality issues
    """
    
    # Thresholds for detection
    BINARY_THRESHOLD = 2
    CLASSIFICATION_THRESHOLD = 20  # Max unique values for classification
    MIN_SAMPLES_CLASSIFICATION = 50
    MIN_SAMPLES_REGRESSION = 30
    MIN_SAMPLES_PER_CLASS = 10
    IMBALANCE_THRESHOLD = 0.1  # Minority class < 10%
    HIGH_MISSING_THRESHOLD = 0.3  # 30% missing
    
    # Algorithm registry
    CLASSIFICATION_ALGORITHMS = [
        AlgorithmRecommendation(
            name="Random Forest Classifier",
            algorithm_id="random_forest_clf",
            priority=1,
            reason="Robust, handles non-linear relationships, resistant to overfitting",
            suitable_for=["balanced_data", "mixed_features", "interpretability"],
            hyperparameters={"n_estimators": 100, "max_depth": None, "min_samples_split": 2}
        ),
        AlgorithmRecommendation(
            name="XGBoost Classifier",
            algorithm_id="xgboost_clf",
            priority=2,
            reason="State-of-the-art performance, handles imbalanced data well",
            suitable_for=["imbalanced_data", "large_datasets", "high_performance"],
            hyperparameters={"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6}
        ),
        AlgorithmRecommendation(
            name="Logistic Regression",
            algorithm_id="logistic_regression",
            priority=3,
            reason="Fast, interpretable, works well with linearly separable data",
            suitable_for=["linear_data", "interpretability", "small_datasets"],
            hyperparameters={"C": 1.0, "max_iter": 1000}
        ),
        AlgorithmRecommendation(
            name="Support Vector Machine",
            algorithm_id="svm_clf",
            priority=4,
            reason="Effective in high dimensions, good for small-medium datasets",
            suitable_for=["small_datasets", "high_dimensions", "clear_margins"],
            hyperparameters={"C": 1.0, "kernel": "rbf"}
        ),
    ]
    
    REGRESSION_ALGORITHMS = [
        AlgorithmRecommendation(
            name="Random Forest Regressor",
            algorithm_id="random_forest_reg",
            priority=1,
            reason="Robust, handles non-linear relationships, feature importance",
            suitable_for=["non_linear", "mixed_features", "outliers"],
            hyperparameters={"n_estimators": 100, "max_depth": None}
        ),
        AlgorithmRecommendation(
            name="XGBoost Regressor",
            algorithm_id="xgboost_reg",
            priority=2,
            reason="State-of-the-art performance, handles complex relationships",
            suitable_for=["large_datasets", "high_performance", "complex_patterns"],
            hyperparameters={"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6}
        ),
        AlgorithmRecommendation(
            name="Ridge Regression",
            algorithm_id="ridge",
            priority=3,
            reason="Linear model with regularization, prevents overfitting",
            suitable_for=["linear_data", "multicollinearity", "interpretability"],
            hyperparameters={"alpha": 1.0}
        ),
        AlgorithmRecommendation(
            name="Lasso Regression",
            algorithm_id="lasso",
            priority=4,
            reason="Feature selection built-in, sparse solutions",
            suitable_for=["feature_selection", "linear_data", "sparse_features"],
            hyperparameters={"alpha": 1.0}
        ),
        AlgorithmRecommendation(
            name="Linear Regression",
            algorithm_id="linear_regression",
            priority=5,
            reason="Simple, fast, interpretable baseline",
            suitable_for=["linear_data", "baseline", "small_datasets"],
            hyperparameters={}
        ),
    ]
    
    def __init__(self):
        """Initialize the problem detector."""
        logger.info("ProblemDetector initialized")
    
    def _analyze_target(self, y: pd.Series, column_name: str) -> TargetAnalysis:
        """
        Analyze the target column characteristics.
        
        Args:
            y: Target series
            column_name: Name of the target column
            
        Returns:
            TargetAnalysis with detailed statistics
        """
        dtype = str(y.dtype)
        unique_values = y.nunique()
        missing_count = y.isna().sum()
        missing_pct = (missing_count / len(y)) * 100
        
        # Value distribution (top 20)
        value_counts = y.value_counts().head(20).to_dict()
        
        # Type detection
        is_numeric = pd.api.types.is_numeric_dtype(y)
        is_binary = unique_values == 2
        is_ordinal = is_numeric and unique_values <= 10
        
        # Suggest problem type
        if is_binary:
            suggested_type = ProblemType.BINARY_CLASSIFICATION
        elif unique_values <= self.CLASSIFICATION_THRESHOLD:
            if is_numeric and not y.dtype == 'int64':
                # Float with few unique values - likely regression
                if unique_values > 10:
                    suggested_type = ProblemType.REGRESSION
                else:
                    suggested_type = ProblemType.MULTICLASS_CLASSIFICATION
            else:
                suggested_type = ProblemType.MULTICLASS_CLASSIFICATION
        elif is_numeric:
            suggested_type = ProblemType.REGRESSION
        else:
            # High cardinality categorical - might be regression encoded
            suggested_type = ProblemType.UNKNOWN
        
        return TargetAnalysis(
            column_name=column_name,
            dtype=dtype,
            unique_values=unique_values,
            missing_count=missing_count,
            missing_pct=round(missing_pct, 2),
            value_distribution=value_counts,
            is_numeric=is_numeric,
            is_binary=is_binary,
            is_ordinal=is_ordinal,
            suggested_type=suggested_type,
            sample_values=y.dropna().head(5).tolist()
        )
    
    def _check_data_issues(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        problem_type: ProblemType
    ) -> Tuple[List[DataIssue], List[str]]:
        """
        Check for data quality issues.
        
        Args:
            X: Features DataFrame
            y: Target series
            problem_type: Detected problem type
            
        Returns:
            Tuple of (issues list, warnings list)
        """
        issues = []
        warnings = []
        
        n_samples = len(X)
        n_features = len(X.columns)
        
        # Check sample count
        if problem_type in [ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION]:
            if n_samples < self.MIN_SAMPLES_CLASSIFICATION:
                issues.append(DataIssue.INSUFFICIENT_SAMPLES)
                warnings.append(f"Only {n_samples} samples. Minimum recommended: {self.MIN_SAMPLES_CLASSIFICATION}")
            
            # Check class balance
            class_counts = y.value_counts()
            min_class_pct = class_counts.min() / class_counts.sum()
            if min_class_pct < self.IMBALANCE_THRESHOLD:
                issues.append(DataIssue.CLASS_IMBALANCE)
                warnings.append(f"Minority class has only {min_class_pct*100:.1f}% of samples")
            
            # Check samples per class
            if class_counts.min() < self.MIN_SAMPLES_PER_CLASS:
                warnings.append(f"Some classes have fewer than {self.MIN_SAMPLES_PER_CLASS} samples")
                
        else:  # Regression
            if n_samples < self.MIN_SAMPLES_REGRESSION:
                issues.append(DataIssue.INSUFFICIENT_SAMPLES)
                warnings.append(f"Only {n_samples} samples. Minimum recommended: {self.MIN_SAMPLES_REGRESSION}")
            
            # Check target variance
            if y.std() < 0.001:
                issues.append(DataIssue.LOW_VARIANCE_TARGET)
                warnings.append("Target variable has very low variance")
        
        # Check missing values in features
        missing_ratio = X.isna().sum().sum() / (n_samples * n_features)
        if missing_ratio > self.HIGH_MISSING_THRESHOLD:
            issues.append(DataIssue.HIGH_MISSING_RATIO)
            warnings.append(f"Dataset has {missing_ratio*100:.1f}% missing values")
        
        # Check for numeric features
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            issues.append(DataIssue.NO_NUMERIC_FEATURES)
            warnings.append("No numeric features found. Ensure categorical encoding is applied.")
        
        # Check feature-to-sample ratio
        if n_features > n_samples / 2:
            issues.append(DataIssue.TOO_MANY_FEATURES)
            warnings.append(f"High feature-to-sample ratio ({n_features} features, {n_samples} samples)")
        
        return issues, warnings
    
    def _recommend_algorithms(
        self, 
        problem_type: ProblemType,
        n_samples: int,
        n_features: int,
        issues: List[DataIssue]
    ) -> List[AlgorithmRecommendation]:
        """
        Recommend algorithms based on problem characteristics.
        
        Args:
            problem_type: Type of problem
            n_samples: Number of samples
            n_features: Number of features
            issues: Detected data issues
            
        Returns:
            List of algorithm recommendations, prioritized
        """
        if problem_type == ProblemType.REGRESSION:
            algorithms = self.REGRESSION_ALGORITHMS.copy()
        elif problem_type in [ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION]:
            algorithms = self.CLASSIFICATION_ALGORITHMS.copy()
        else:
            return []
        
        # Adjust priorities based on data characteristics
        for algo in algorithms:
            # Boost XGBoost for imbalanced data
            if DataIssue.CLASS_IMBALANCE in issues and "imbalanced_data" in algo.suitable_for:
                algo.priority -= 1
            
            # Boost linear models for small datasets
            if n_samples < 500 and "small_datasets" in algo.suitable_for:
                algo.priority -= 1
            
            # Boost tree models for large feature sets
            if n_features > 50 and "high_dimensions" in algo.suitable_for:
                algo.priority -= 1
        
        # Sort by priority
        algorithms.sort(key=lambda x: x.priority)
        
        return algorithms
    
    def detect(
        self, 
        df: pd.DataFrame, 
        target_column: str,
        force_type: Optional[ProblemType] = None
    ) -> ProblemDetectionResult:
        """
        Detect problem type and analyze dataset.
        
        Args:
            df: Full DataFrame including target
            target_column: Name of the target column
            force_type: Override automatic detection with specific type
            
        Returns:
            ProblemDetectionResult with full analysis
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        logger.info(f"Detecting problem type for target: {target_column}")
        
        # Analyze target
        target_analysis = self._analyze_target(y, target_column)
        
        # Determine problem type
        if force_type:
            problem_type = force_type
        else:
            problem_type = target_analysis.suggested_type
        
        logger.info(f"Detected problem type: {problem_type.value}")
        
        # Check for issues
        issues, warnings = self._check_data_issues(X, y, problem_type)
        
        # Get recommendations
        recommendations = self._recommend_algorithms(
            problem_type,
            len(X),
            len(X.columns),
            issues
        )
        
        # Dataset stats
        dataset_stats = {
            "n_samples": len(df),
            "n_features": len(X.columns),
            "n_numeric_features": len(X.select_dtypes(include=[np.number]).columns),
            "n_categorical_features": len(X.select_dtypes(include=['object', 'category']).columns),
            "total_missing": X.isna().sum().sum(),
            "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        }
        
        # Determine trainability
        is_trainable = (
            problem_type != ProblemType.UNKNOWN and
            DataIssue.INSUFFICIENT_SAMPLES not in issues and
            DataIssue.NO_NUMERIC_FEATURES not in issues
        )
        
        # Minimum samples
        if problem_type == ProblemType.REGRESSION:
            min_samples = self.MIN_SAMPLES_REGRESSION
        else:
            min_samples = self.MIN_SAMPLES_CLASSIFICATION
        
        result = ProblemDetectionResult(
            problem_type=problem_type,
            target_analysis=target_analysis,
            recommended_algorithms=recommendations,
            data_issues=issues,
            warnings=warnings,
            dataset_stats=dataset_stats,
            is_trainable=is_trainable,
            min_required_samples=min_samples
        )
        
        if warnings:
            for warning in warnings:
                logger.warning(warning)
        
        return result
    
    def suggest_target(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Suggest potential target columns.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of potential targets with scores
        """
        suggestions = []
        
        for col in df.columns:
            series = df[col]
            unique_count = series.nunique()
            missing_pct = (series.isna().sum() / len(series)) * 100
            
            score = 0
            reasons = []
            
            # Low cardinality is good for classification
            if 2 <= unique_count <= 20:
                score += 30
                reasons.append("Suitable cardinality for classification")
            
            # Numeric with high variance is good for regression
            if pd.api.types.is_numeric_dtype(series) and unique_count > 20:
                if series.std() > 0:
                    score += 25
                    reasons.append("Numeric with variance - suitable for regression")
            
            # Low missing values
            if missing_pct < 5:
                score += 20
                reasons.append("Low missing values")
            elif missing_pct < 20:
                score += 10
            
            # Column name heuristics
            target_keywords = ['target', 'label', 'class', 'outcome', 'result', 'y', 'prediction']
            if any(keyword in col.lower() for keyword in target_keywords):
                score += 25
                reasons.append("Column name suggests target")
            
            if score > 0:
                suggestions.append({
                    "column": col,
                    "score": score,
                    "unique_values": unique_count,
                    "missing_pct": round(missing_pct, 2),
                    "dtype": str(series.dtype),
                    "reasons": reasons
                })
        
        # Sort by score descending
        suggestions.sort(key=lambda x: x["score"], reverse=True)
        
        return suggestions[:5]  # Top 5 suggestions
