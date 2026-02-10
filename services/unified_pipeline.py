"""
Unified Pipeline - Standalone Inference Package

Production-grade unified pipeline that bundles:
- Preprocessing transformations
- Feature engineering
- Trained model
All into a single pickle file that can be loaded and used outside the platform.

Usage:
    # Export
    pipeline = UnifiedPipeline.from_components(preprocessing, model, config)
    pipeline.save("pipeline.pkl")
    
    # Use anywhere
    pipeline = UnifiedPipeline.load("pipeline.pkl")
    predictions = pipeline.predict(new_data)
"""

import logging
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json

from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


@dataclass
class PipelineMetadata:
    """Metadata about the pipeline."""
    version: str
    created_at: str
    algorithm: str
    feature_names: List[str]
    target_name: str
    problem_type: str
    metrics: Dict[str, float]
    preprocessing_config: Dict[str, Any]
    schema: Dict[str, Any]  # Expected input schema


class UnifiedPipeline(BaseEstimator):
    """
    Unified inference pipeline that bundles all components.
    
    This class implements sklearn's estimator interface for compatibility.
    Can be saved as a single file and used independently.
    
    Supports:
    - Direct predict() calls on raw data
    - Schema validation before inference
    - Preprocessing + feature engineering + model inference
    - Probability predictions for classifiers
    """
    
    VERSION = "2.0.0"
    
    def __init__(
        self,
        preprocessing_pipeline=None,
        feature_engineer=None,
        model: BaseEstimator = None,
        metadata: PipelineMetadata = None
    ):
        """
        Initialize unified pipeline.
        
        Args:
            preprocessing_pipeline: Fitted preprocessing transformer
            feature_engineer: Fitted feature engineer (optional)
            model: Trained model
            metadata: Pipeline metadata
        """
        self.preprocessing_pipeline = preprocessing_pipeline
        self.feature_engineer = feature_engineer
        self.model = model
        self.metadata = metadata or PipelineMetadata(
            version=self.VERSION,
            created_at=datetime.now().isoformat(),
            algorithm="unknown",
            feature_names=[],
            target_name="target",
            problem_type="unknown",
            metrics={},
            preprocessing_config={},
            schema={}
        )
        
        self._is_fitted = model is not None
    
    @classmethod
    def from_components(
        cls,
        preprocessing_pipeline,
        model: BaseEstimator,
        feature_names: List[str],
        target_name: str,
        problem_type: str = "unknown",
        metrics: Dict[str, float] = None,
        feature_engineer=None,
        preprocessing_config: Dict[str, Any] = None
    ) -> "UnifiedPipeline":
        """
        Create unified pipeline from individual components.
        
        Args:
            preprocessing_pipeline: Fitted preprocessing pipeline
            model: Trained model
            feature_names: List of feature names
            target_name: Target column name
            problem_type: 'classification' or 'regression'
            metrics: Model metrics
            feature_engineer: Feature engineer (optional)
            preprocessing_config: Preprocessing configuration
            
        Returns:
            UnifiedPipeline instance
        """
        # Build schema
        schema = {
            "required_columns": feature_names,
            "n_features": len(feature_names)
        }
        
        metadata = PipelineMetadata(
            version=cls.VERSION,
            created_at=datetime.now().isoformat(),
            algorithm=type(model).__name__,
            feature_names=feature_names,
            target_name=target_name,
            problem_type=problem_type,
            metrics=metrics or {},
            preprocessing_config=preprocessing_config or {},
            schema=schema
        )
        
        return cls(
            preprocessing_pipeline=preprocessing_pipeline,
            feature_engineer=feature_engineer,
            model=model,
            metadata=metadata
        )
    
    def fit(self, X, y=None):
        """Fit is not supported - pipeline must be pre-fitted."""
        raise NotImplementedError(
            "UnifiedPipeline is for inference only. "
            "Use from_components() with pre-fitted components."
        )
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform raw data through preprocessing pipeline.
        
        Args:
            X: Input DataFrame
            
        Returns:
            Transformed features array
        """
        if not self._is_fitted:
            raise ValueError("Pipeline not fitted")
        
        X_processed = X.copy()
        
        # Apply preprocessing
        if self.preprocessing_pipeline is not None:
            X_processed = self.preprocessing_pipeline.transform(X_processed)
        
        # Apply feature engineering
        if self.feature_engineer is not None:
            X_processed = self.feature_engineer.transform(X_processed)
        
        # Convert to array if DataFrame
        if isinstance(X_processed, pd.DataFrame):
            X_processed = X_processed.values
        
        return X_processed
    
    def predict(self, X: Union[pd.DataFrame, Dict, List[Dict]]) -> np.ndarray:
        """
        Make predictions on raw input data.
        
        Handles:
        - DataFrames
        - Single dict (one row)
        - List of dicts (multiple rows)
        
        Args:
            X: Input data
            
        Returns:
            Predictions array
        """
        # Convert to DataFrame if needed
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif isinstance(X, list):
            X = pd.DataFrame(X)
        
        # Transform
        X_transformed = self.transform(X)
        
        # Predict
        return self.model.predict(X_transformed)
    
    def predict_proba(self, X: Union[pd.DataFrame, Dict, List[Dict]]) -> np.ndarray:
        """
        Get prediction probabilities (classification only).
        
        Args:
            X: Input data
            
        Returns:
            Probability array
        """
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model does not support predict_proba")
        
        # Convert to DataFrame if needed
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif isinstance(X, list):
            X = pd.DataFrame(X)
        
        # Transform
        X_transformed = self.transform(X)
        
        return self.model.predict_proba(X_transformed)
    
    def validate_input(self, X: pd.DataFrame) -> tuple[bool, List[str]]:
        """
        Validate input data against expected schema.
        
        Args:
            X: Input DataFrame
            
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required columns
        required = set(self.metadata.feature_names)
        actual = set(X.columns)
        
        missing = required - actual
        if missing:
            issues.append(f"Missing columns: {list(missing)}")
        
        extra = actual - required
        if extra:
            issues.append(f"Extra columns (will be ignored): {list(extra)}")
        
        # Check for all nulls
        for col in X.columns:
            if X[col].isna().all():
                issues.append(f"Column '{col}' is completely null")
        
        return len([i for i in issues if "Missing" in i]) == 0, issues
    
    def get_info(self) -> Dict[str, Any]:
        """Get pipeline information."""
        return {
            "version": self.metadata.version,
            "created_at": self.metadata.created_at,
            "algorithm": self.metadata.algorithm,
            "problem_type": self.metadata.problem_type,
            "n_features": len(self.metadata.feature_names),
            "feature_names": self.metadata.feature_names,
            "target_name": self.metadata.target_name,
            "metrics": self.metadata.metrics,
            "is_fitted": self._is_fitted
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save unified pipeline to a single file.
        
        Args:
            path: Output path (should end in .pkl)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save everything as one object
        joblib.dump(self, path)
        
        # Also save metadata as JSON for inspection
        metadata_path = path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(self.get_info(), f, indent=2, default=str)
        
        logger.info(f"Unified pipeline saved to {path}")
        logger.info(f"Metadata saved to {metadata_path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "UnifiedPipeline":
        """
        Load unified pipeline from file.
        
        Args:
            path: Path to saved pipeline
            
        Returns:
            Loaded UnifiedPipeline
        """
        path = Path(path)
        
        pipeline = joblib.load(path)
        
        if not isinstance(pipeline, cls):
            raise ValueError(f"Loaded object is not a UnifiedPipeline: {type(pipeline)}")
        
        logger.info(f"Loaded pipeline: {pipeline.metadata.algorithm}")
        
        return pipeline
    
    def __repr__(self):
        return (
            f"UnifiedPipeline("
            f"algorithm={self.metadata.algorithm}, "
            f"n_features={len(self.metadata.feature_names)}, "
            f"fitted={self._is_fitted})"
        )


def create_unified_pipeline(
    preprocessing_pipeline,
    model: BaseEstimator,
    X_train: pd.DataFrame,
    target_name: str,
    problem_type: str,
    metrics: Dict[str, float] = None
) -> UnifiedPipeline:
    """
    Convenience function to create a unified pipeline.
    
    Args:
        preprocessing_pipeline: Fitted preprocessing
        model: Trained model
        X_train: Training features (for metadata)
        target_name: Target column name
        problem_type: Problem type
        metrics: Model metrics
        
    Returns:
        UnifiedPipeline ready for export
    """
    return UnifiedPipeline.from_components(
        preprocessing_pipeline=preprocessing_pipeline,
        model=model,
        feature_names=X_train.columns.tolist(),
        target_name=target_name,
        problem_type=problem_type,
        metrics=metrics
    )


# Example usage documentation
"""
USAGE EXAMPLE - CREATING AND USING A UNIFIED PIPELINE
======================================================

# During training (in your platform):
from services.unified_pipeline import UnifiedPipeline

pipeline = UnifiedPipeline.from_components(
    preprocessing_pipeline=fitted_preprocessor,
    model=trained_model,
    feature_names=X_train.columns.tolist(),
    target_name="price",
    problem_type="regression",
    metrics={"rmse": 0.15, "r2": 0.92}
)

pipeline.save("my_model_pipeline.pkl")

# Anywhere else (no platform needed):
pipeline = UnifiedPipeline.load("my_model_pipeline.pkl")

# Predict on new data
new_data = pd.DataFrame([
    {"feature1": 10, "feature2": "A", "feature3": 0.5},
    {"feature1": 20, "feature2": "B", "feature3": 0.8}
])

predictions = pipeline.predict(new_data)
print(predictions)

# Or predict single row
prediction = pipeline.predict({"feature1": 15, "feature2": "C", "feature3": 0.6})
print(prediction)

# Get probabilities (classification only)
probabilities = pipeline.predict_proba(new_data)

# Validate input before prediction
is_valid, issues = pipeline.validate_input(new_data)
if not is_valid:
    print("Validation issues:", issues)
"""
