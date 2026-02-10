"""
Feature Engineering Service

Provides advanced feature engineering capabilities:
- Automatic feature generation
- Polynomial features
- Interaction terms
- Datetime feature extraction
- Text feature extraction
- Feature selection (variance, correlation, importance)
- Train/test splitting with stratification
"""

import logging
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    RFE
)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class FeatureSelectionMethod(Enum):
    """Feature selection methods."""
    VARIANCE = "variance"
    CORRELATION = "correlation"
    MUTUAL_INFO = "mutual_info"
    F_SCORE = "f_score"
    RFE = "rfe"
    IMPORTANCE = "importance"


@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering."""
    create_polynomial: bool = False
    polynomial_degree: int = 2
    create_interactions: bool = False
    interaction_columns: List[str] = None
    extract_datetime: bool = True
    selection_method: Optional[FeatureSelectionMethod] = None
    selection_k: int = 10  # Top k features to select
    variance_threshold: float = 0.01
    correlation_threshold: float = 0.95  # Remove if correlation > threshold
    
    def __post_init__(self):
        if self.interaction_columns is None:
            self.interaction_columns = []


@dataclass
class SplitConfig:
    """Configuration for train/test split."""
    test_size: float = 0.2
    validation_size: float = 0.0  # If > 0, creates train/val/test split
    random_state: int = 42
    stratify: bool = True  # For classification problems
    shuffle: bool = True


class FeatureEngineer:
    """
    Production-grade feature engineering service.
    
    Handles feature creation, transformation, and selection
    with full serialization support for inference pipelines.
    """
    
    def __init__(self, config: Optional[FeatureEngineeringConfig] = None):
        """
        Initialize the feature engineer.
        
        Args:
            config: Feature engineering configuration
        """
        self.config = config or FeatureEngineeringConfig()
        self.fitted = False
        
        # Fitted transformers
        self._polynomial: Optional[PolynomialFeatures] = None
        self._selector: Optional[Any] = None
        self._feature_importances: Dict[str, float] = {}
        
        # Column tracking
        self._input_columns: List[str] = []
        self._datetime_columns: List[str] = []
        self._generated_columns: List[str] = []
        self._selected_columns: List[str] = []
        self._dropped_columns: List[str] = []
        
        logger.info("FeatureEngineer initialized")
    
    def _extract_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from datetime columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with extracted datetime features
        """
        df = df.copy()
        
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                self._datetime_columns.append(col)
                
                # Extract components
                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_day"] = df[col].dt.day
                df[f"{col}_dayofweek"] = df[col].dt.dayofweek
                df[f"{col}_hour"] = df[col].dt.hour
                df[f"{col}_is_weekend"] = df[col].dt.dayofweek.isin([5, 6]).astype(int)
                
                # Track generated columns
                self._generated_columns.extend([
                    f"{col}_year", f"{col}_month", f"{col}_day",
                    f"{col}_dayofweek", f"{col}_hour", f"{col}_is_weekend"
                ])
                
                # Drop original datetime column
                df = df.drop(columns=[col])
                self._dropped_columns.append(col)
                
                logger.info(f"Extracted 6 features from datetime column: {col}")
        
        return df
    
    def _create_polynomial_features(
        self, 
        df: pd.DataFrame, 
        numeric_columns: List[str]
    ) -> pd.DataFrame:
        """
        Create polynomial features from numeric columns.
        
        Args:
            df: Input DataFrame
            numeric_columns: Columns to use for polynomial features
            
        Returns:
            DataFrame with polynomial features added
        """
        if not numeric_columns or len(numeric_columns) < 2:
            return df
        
        # Limit columns to prevent explosion
        cols_to_use = numeric_columns[:5]  # Max 5 columns
        
        self._polynomial = PolynomialFeatures(
            degree=self.config.polynomial_degree,
            include_bias=False,
            interaction_only=self.config.create_interactions
        )
        
        poly_features = self._polynomial.fit_transform(df[cols_to_use])
        poly_names = self._polynomial.get_feature_names_out(cols_to_use)
        
        # Only keep new features (not original columns)
        new_features_mask = ~np.isin(poly_names, cols_to_use)
        new_poly_features = poly_features[:, new_features_mask]
        new_poly_names = poly_names[new_features_mask]
        
        poly_df = pd.DataFrame(new_poly_features, columns=new_poly_names, index=df.index)
        
        self._generated_columns.extend(new_poly_names.tolist())
        logger.info(f"Created {len(new_poly_names)} polynomial features")
        
        return pd.concat([df, poly_df], axis=1)
    
    def _create_interaction_features(
        self, 
        df: pd.DataFrame, 
        columns: List[str]
    ) -> pd.DataFrame:
        """
        Create interaction features between specified columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to create interactions between
            
        Returns:
            DataFrame with interaction features
        """
        df = df.copy()
        
        if len(columns) < 2:
            return df
        
        from itertools import combinations
        
        for col1, col2 in combinations(columns, 2):
            if col1 in df.columns and col2 in df.columns:
                # Only for numeric columns
                if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                    interaction_name = f"{col1}_x_{col2}"
                    df[interaction_name] = df[col1] * df[col2]
                    self._generated_columns.append(interaction_name)
        
        logger.info(f"Created interaction features from {len(columns)} columns")
        return df
    
    def _remove_high_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove highly correlated features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with high-correlation features removed
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return df
        
        corr_matrix = df[numeric_cols].corr().abs()
        
        # Upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find columns with correlation > threshold
        to_drop = [column for column in upper.columns if any(upper[column] > self.config.correlation_threshold)]
        
        if to_drop:
            df = df.drop(columns=to_drop)
            self._dropped_columns.extend(to_drop)
            logger.info(f"Removed {len(to_drop)} highly correlated features")
        
        return df
    
    def _select_features_variance(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select features using variance threshold.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (transformed DataFrame, selected column names)
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return df, []
        
        selector = VarianceThreshold(threshold=self.config.variance_threshold)
        selector.fit(df[numeric_cols])
        
        selected_mask = selector.get_support()
        selected_cols = [col for col, selected in zip(numeric_cols, selected_mask) if selected]
        
        non_numeric = [col for col in df.columns if col not in numeric_cols]
        
        self._selector = selector
        
        return df[selected_cols + non_numeric], selected_cols
    
    def _select_features_importance(
        self, 
        df: pd.DataFrame, 
        y: pd.Series,
        is_classification: bool
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select features based on model importance.
        
        Args:
            df: Input DataFrame
            y: Target series
            is_classification: Whether this is a classification problem
            
        Returns:
            Tuple of (transformed DataFrame, selected column names)
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return df, []
        
        # Use Random Forest for importance
        if is_classification:
            model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        
        X = df[numeric_cols].fillna(0)
        model.fit(X, y)
        
        # Get importances
        importances = dict(zip(numeric_cols, model.feature_importances_))
        self._feature_importances = importances
        
        # Select top k
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        k = min(self.config.selection_k, len(sorted_features))
        selected_cols = [f[0] for f in sorted_features[:k]]
        
        non_numeric = [col for col in df.columns if col not in numeric_cols]
        
        logger.info(f"Selected top {k} features by importance")
        
        return df[selected_cols + non_numeric], selected_cols
    
    def fit(
        self, 
        df: pd.DataFrame, 
        target: Optional[pd.Series] = None,
        is_classification: bool = True
    ) -> "FeatureEngineer":
        """
        Fit the feature engineer on training data.
        
        Args:
            df: Training DataFrame
            target: Target series (required for some selection methods)
            is_classification: Whether this is a classification problem
            
        Returns:
            Self for chaining
        """
        self._input_columns = df.columns.tolist()
        
        logger.info(f"Fitting FeatureEngineer on {len(df)} samples, {len(df.columns)} features")
        
        self.fitted = True
        self._is_classification = is_classification
        
        return self
    
    def transform(
        self, 
        df: pd.DataFrame,
        target: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Transform data with feature engineering.
        
        Args:
            df: DataFrame to transform
            target: Target series (optional, for selection methods)
            
        Returns:
            Transformed DataFrame
        """
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        result = df.copy()
        
        # Extract datetime features
        if self.config.extract_datetime:
            result = self._extract_datetime_features(result)
        
        # Get numeric columns
        numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create polynomial features
        if self.config.create_polynomial:
            result = self._create_polynomial_features(result, numeric_cols)
        
        # Create interaction features
        if self.config.create_interactions and self.config.interaction_columns:
            result = self._create_interaction_features(result, self.config.interaction_columns)
        
        # Remove high correlation
        if self.config.correlation_threshold < 1.0:
            result = self._remove_high_correlation(result)
        
        # Feature selection
        if self.config.selection_method and target is not None:
            if self.config.selection_method == FeatureSelectionMethod.VARIANCE:
                result, self._selected_columns = self._select_features_variance(result)
            elif self.config.selection_method == FeatureSelectionMethod.IMPORTANCE:
                result, self._selected_columns = self._select_features_importance(
                    result, target, self._is_classification
                )
        
        logger.info(f"Transformed to {len(result.columns)} features")
        
        return result
    
    def fit_transform(
        self, 
        df: pd.DataFrame,
        target: Optional[pd.Series] = None,
        is_classification: bool = True
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: Training DataFrame
            target: Target series
            is_classification: Whether classification problem
            
        Returns:
            Transformed DataFrame
        """
        self.fit(df, target, is_classification)
        return self.transform(df, target)
    
    @staticmethod
    def split_data(
        X: pd.DataFrame,
        y: pd.Series,
        config: Optional[SplitConfig] = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
               Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]]:
        """
        Split data into train/test or train/val/test sets.
        
        Args:
            X: Features DataFrame
            y: Target Series
            config: Split configuration
            
        Returns:
            Tuple of split data arrays
        """
        config = config or SplitConfig()
        
        stratify = y if config.stratify and len(y.unique()) < 50 else None
        
        if config.validation_size > 0:
            # Three-way split
            test_val_size = config.test_size + config.validation_size
            
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y,
                test_size=test_val_size,
                random_state=config.random_state,
                stratify=stratify,
                shuffle=config.shuffle
            )
            
            val_ratio = config.validation_size / test_val_size
            stratify_temp = y_temp if config.stratify and len(y_temp.unique()) < 50 else None
            
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp,
                test_size=(1 - val_ratio),
                random_state=config.random_state,
                stratify=stratify_temp,
                shuffle=config.shuffle
            )
            
            logger.info(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
            return X_train, X_val, X_test, y_train, y_val, y_test
        
        else:
            # Two-way split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=config.test_size,
                random_state=config.random_state,
                stratify=stratify,
                shuffle=config.shuffle
            )
            
            logger.info(f"Split: train={len(X_train)}, test={len(X_test)}")
            return X_train, X_test, y_train, y_test
    
    def get_feature_importances(self) -> Dict[str, float]:
        """Get computed feature importances."""
        return self._feature_importances
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the feature engineer to disk.
        
        Args:
            path: Directory to save
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if self._polynomial:
            joblib.dump(self._polynomial, path / "polynomial.pkl")
        if self._selector:
            joblib.dump(self._selector, path / "selector.pkl")
        
        metadata = {
            "config": {
                "create_polynomial": self.config.create_polynomial,
                "polynomial_degree": self.config.polynomial_degree,
                "create_interactions": self.config.create_interactions,
                "extract_datetime": self.config.extract_datetime,
                "correlation_threshold": self.config.correlation_threshold,
            },
            "columns": {
                "input": self._input_columns,
                "datetime": self._datetime_columns,
                "generated": self._generated_columns,
                "selected": self._selected_columns,
                "dropped": self._dropped_columns,
            },
            "feature_importances": self._feature_importances,
            "fitted": self.fitted,
        }
        
        import json
        with open(path / "feature_engineer_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"FeatureEngineer saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "FeatureEngineer":
        """
        Load a saved feature engineer.
        
        Args:
            path: Directory containing saved files
            
        Returns:
            Loaded FeatureEngineer
        """
        path = Path(path)
        
        import json
        with open(path / "feature_engineer_metadata.json", "r") as f:
            metadata = json.load(f)
        
        config = FeatureEngineeringConfig(
            create_polynomial=metadata["config"]["create_polynomial"],
            polynomial_degree=metadata["config"]["polynomial_degree"],
            create_interactions=metadata["config"]["create_interactions"],
            extract_datetime=metadata["config"]["extract_datetime"],
            correlation_threshold=metadata["config"]["correlation_threshold"],
        )
        
        engineer = cls(config)
        
        if (path / "polynomial.pkl").exists():
            engineer._polynomial = joblib.load(path / "polynomial.pkl")
        if (path / "selector.pkl").exists():
            engineer._selector = joblib.load(path / "selector.pkl")
        
        engineer._input_columns = metadata["columns"]["input"]
        engineer._datetime_columns = metadata["columns"]["datetime"]
        engineer._generated_columns = metadata["columns"]["generated"]
        engineer._selected_columns = metadata["columns"]["selected"]
        engineer._dropped_columns = metadata["columns"]["dropped"]
        engineer._feature_importances = metadata["feature_importances"]
        engineer.fitted = metadata["fitted"]
        
        logger.info(f"FeatureEngineer loaded from {path}")
        return engineer
