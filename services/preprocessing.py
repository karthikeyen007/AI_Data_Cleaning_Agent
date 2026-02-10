"""
Preprocessing Pipeline - Data Preparation Service

Provides production-grade data preprocessing capabilities:
- Missing value handling with multiple strategies
- Categorical encoding (label, one-hot, ordinal)
- Numerical scaling (standard, minmax, robust)
- Outlier detection and handling
- Data type inference and conversion
- Pipeline serialization for inference
"""

import logging
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, 
    MinMaxScaler, 
    RobustScaler,
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class MissingValueStrategy(Enum):
    """Strategies for handling missing values."""
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "most_frequent"
    CONSTANT = "constant"
    KNN = "knn"
    DROP = "drop"


class ScalingStrategy(Enum):
    """Strategies for numerical scaling."""
    STANDARD = "standard"      # Z-score normalization
    MINMAX = "minmax"          # Scale to [0, 1]
    ROBUST = "robust"          # Median and IQR based
    NONE = "none"


class EncodingStrategy(Enum):
    """Strategies for categorical encoding."""
    LABEL = "label"
    ONEHOT = "onehot"
    ORDINAL = "ordinal"
    NONE = "none"


@dataclass
class ColumnProfile:
    """Profile of a single column."""
    name: str
    dtype: str
    missing_count: int
    missing_pct: float
    unique_count: int
    is_numeric: bool
    is_categorical: bool
    is_datetime: bool
    is_target_candidate: bool
    sample_values: List[Any] = field(default_factory=list)


@dataclass
class DatasetProfile:
    """Complete profile of a dataset."""
    n_rows: int
    n_cols: int
    columns: List[ColumnProfile]
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    high_cardinality_columns: List[str]  # >50 unique values
    constant_columns: List[str]  # Single unique value
    memory_usage_mb: float


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    missing_strategy: MissingValueStrategy = MissingValueStrategy.MEDIAN
    missing_fill_value: Any = 0
    scaling_strategy: ScalingStrategy = ScalingStrategy.STANDARD
    encoding_strategy: EncodingStrategy = EncodingStrategy.ONEHOT
    drop_constant_columns: bool = True
    handle_high_cardinality: bool = True
    high_cardinality_threshold: int = 50
    outlier_handling: bool = False
    outlier_threshold: float = 3.0  # Z-score threshold


class PreprocessingPipeline:
    """
    Production-grade preprocessing pipeline.
    
    Handles the complete data preparation workflow from raw data
    to ML-ready features with full serialization support.
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()
        self.fitted = False
        self.profile: Optional[DatasetProfile] = None
        
        # Fitted transformers
        self._numeric_imputer: Optional[SimpleImputer] = None
        self._categorical_imputer: Optional[SimpleImputer] = None
        self._scaler: Optional[Any] = None
        self._encoders: Dict[str, Any] = {}
        self._column_transformer: Optional[ColumnTransformer] = None
        
        # Column mappings
        self._original_columns: List[str] = []
        self._numeric_columns: List[str] = []
        self._categorical_columns: List[str] = []
        self._dropped_columns: List[str] = []
        self._encoded_column_names: List[str] = []
        
        logger.info("PreprocessingPipeline initialized")
    
    def profile_dataset(self, df: pd.DataFrame) -> DatasetProfile:
        """
        Generate a comprehensive profile of the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DatasetProfile with column-level and dataset-level statistics
        """
        columns = []
        numeric_cols = []
        categorical_cols = []
        datetime_cols = []
        high_cardinality_cols = []
        constant_cols = []
        
        for col in df.columns:
            series = df[col]
            dtype = str(series.dtype)
            missing_count = series.isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            unique_count = series.nunique()
            
            # Type detection
            is_numeric = pd.api.types.is_numeric_dtype(series)
            is_datetime = pd.api.types.is_datetime64_any_dtype(series)
            is_categorical = not is_numeric and not is_datetime
            
            # Target candidate heuristics
            is_target_candidate = (
                (is_numeric and unique_count <= 20) or  # Small range numeric
                (is_categorical and unique_count <= 10)  # Low cardinality categorical
            )
            
            profile = ColumnProfile(
                name=col,
                dtype=dtype,
                missing_count=missing_count,
                missing_pct=round(missing_pct, 2),
                unique_count=unique_count,
                is_numeric=is_numeric,
                is_categorical=is_categorical,
                is_datetime=is_datetime,
                is_target_candidate=is_target_candidate,
                sample_values=series.dropna().head(5).tolist()
            )
            columns.append(profile)
            
            if is_numeric:
                numeric_cols.append(col)
            elif is_categorical:
                categorical_cols.append(col)
            elif is_datetime:
                datetime_cols.append(col)
            
            if unique_count > self.config.high_cardinality_threshold:
                high_cardinality_cols.append(col)
            
            if unique_count <= 1:
                constant_cols.append(col)
        
        self.profile = DatasetProfile(
            n_rows=len(df),
            n_cols=len(df.columns),
            columns=columns,
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            datetime_columns=datetime_cols,
            high_cardinality_columns=high_cardinality_cols,
            constant_columns=constant_cols,
            memory_usage_mb=round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        )
        
        return self.profile
    
    def _create_numeric_imputer(self) -> SimpleImputer:
        """Create imputer for numeric columns."""
        if self.config.missing_strategy == MissingValueStrategy.KNN:
            return KNNImputer(n_neighbors=5)
        elif self.config.missing_strategy == MissingValueStrategy.CONSTANT:
            return SimpleImputer(strategy="constant", fill_value=self.config.missing_fill_value)
        else:
            return SimpleImputer(strategy=self.config.missing_strategy.value)
    
    def _create_categorical_imputer(self) -> SimpleImputer:
        """Create imputer for categorical columns."""
        if self.config.missing_strategy in [MissingValueStrategy.MEAN, MissingValueStrategy.MEDIAN]:
            # Default to mode for categorical
            return SimpleImputer(strategy="most_frequent")
        elif self.config.missing_strategy == MissingValueStrategy.CONSTANT:
            return SimpleImputer(strategy="constant", fill_value="missing")
        else:
            return SimpleImputer(strategy="most_frequent")
    
    def _create_scaler(self) -> Optional[Any]:
        """Create scaler based on configuration."""
        if self.config.scaling_strategy == ScalingStrategy.STANDARD:
            return StandardScaler()
        elif self.config.scaling_strategy == ScalingStrategy.MINMAX:
            return MinMaxScaler()
        elif self.config.scaling_strategy == ScalingStrategy.ROBUST:
            return RobustScaler()
        return None
    
    def _handle_outliers(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Handle outliers in numeric columns using Z-score method.
        
        Args:
            df: Input DataFrame
            columns: Numeric columns to check
            
        Returns:
            DataFrame with outliers capped
        """
        df_clean = df.copy()
        
        for col in columns:
            if col in df_clean.columns:
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                
                if std > 0:
                    z_scores = np.abs((df_clean[col] - mean) / std)
                    mask = z_scores > self.config.outlier_threshold
                    
                    # Cap outliers at threshold boundaries
                    lower = mean - self.config.outlier_threshold * std
                    upper = mean + self.config.outlier_threshold * std
                    
                    df_clean.loc[df_clean[col] < lower, col] = lower
                    df_clean.loc[df_clean[col] > upper, col] = upper
                    
                    if mask.sum() > 0:
                        logger.info(f"Capped {mask.sum()} outliers in column '{col}'")
        
        return df_clean
    
    def fit(self, df: pd.DataFrame, target_column: Optional[str] = None) -> "PreprocessingPipeline":
        """
        Fit the preprocessing pipeline on the training data.
        
        Args:
            df: Training DataFrame
            target_column: Name of the target column (excluded from transformation)
            
        Returns:
            Self for chaining
        """
        # Profile the dataset
        self.profile_dataset(df)
        
        # Store original columns
        self._original_columns = df.columns.tolist()
        
        # Separate target if provided
        if target_column and target_column in df.columns:
            df = df.drop(columns=[target_column])
        
        # Identify column types
        self._numeric_columns = [
            col for col in self.profile.numeric_columns 
            if col != target_column
        ]
        self._categorical_columns = [
            col for col in self.profile.categorical_columns 
            if col != target_column
        ]
        
        # Drop constant columns if configured
        if self.config.drop_constant_columns:
            self._dropped_columns.extend(self.profile.constant_columns)
            self._numeric_columns = [c for c in self._numeric_columns if c not in self._dropped_columns]
            self._categorical_columns = [c for c in self._categorical_columns if c not in self._dropped_columns]
        
        # Handle high cardinality if configured
        if self.config.handle_high_cardinality:
            for col in self.profile.high_cardinality_columns:
                if col in self._categorical_columns:
                    self._categorical_columns.remove(col)
                    self._dropped_columns.append(col)
                    logger.warning(f"Dropped high cardinality column: {col}")
        
        # Create and fit numeric pipeline
        if self._numeric_columns:
            self._numeric_imputer = self._create_numeric_imputer()
            self._numeric_imputer.fit(df[self._numeric_columns])
            
            self._scaler = self._create_scaler()
            if self._scaler:
                # First impute, then fit scaler
                imputed = self._numeric_imputer.transform(df[self._numeric_columns])
                self._scaler.fit(imputed)
        
        # Create and fit categorical pipeline
        if self._categorical_columns:
            self._categorical_imputer = self._create_categorical_imputer()
            self._categorical_imputer.fit(df[self._categorical_columns].astype(str))
            
            # Create encoders
            if self.config.encoding_strategy == EncodingStrategy.ONEHOT:
                for col in self._categorical_columns:
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                    encoder.fit(df[[col]].astype(str))
                    self._encoders[col] = encoder
            elif self.config.encoding_strategy == EncodingStrategy.LABEL:
                for col in self._categorical_columns:
                    encoder = LabelEncoder()
                    encoder.fit(df[col].astype(str).fillna("missing"))
                    self._encoders[col] = encoder
            elif self.config.encoding_strategy == EncodingStrategy.ORDINAL:
                encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                encoder.fit(df[self._categorical_columns].astype(str))
                self._encoders["_ordinal"] = encoder
        
        self.fitted = True
        logger.info(f"Pipeline fitted on {len(df)} samples")
        logger.info(f"Numeric columns: {len(self._numeric_columns)}")
        logger.info(f"Categorical columns: {len(self._categorical_columns)}")
        logger.info(f"Dropped columns: {len(self._dropped_columns)}")
        
        return self
    
    def transform(self, df: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """
        Transform data using the fitted pipeline.
        
        Args:
            df: DataFrame to transform
            target_column: Target column to preserve unchanged
            
        Returns:
            Transformed DataFrame
        """
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        result_dfs = []
        
        # Preserve target column
        target_series = None
        if target_column and target_column in df.columns:
            target_series = df[target_column].copy()
            df = df.drop(columns=[target_column])
        
        # Handle outliers if configured
        if self.config.outlier_handling and self._numeric_columns:
            df = self._handle_outliers(df, self._numeric_columns)
        
        # Transform numeric columns
        if self._numeric_columns:
            existing_numeric = [c for c in self._numeric_columns if c in df.columns]
            if existing_numeric:
                numeric_data = self._numeric_imputer.transform(df[existing_numeric])
                if self._scaler:
                    numeric_data = self._scaler.transform(numeric_data)
                result_dfs.append(pd.DataFrame(numeric_data, columns=existing_numeric, index=df.index))
        
        # Transform categorical columns
        if self._categorical_columns and self._encoders:
            existing_categorical = [c for c in self._categorical_columns if c in df.columns]
            if existing_categorical:
                cat_imputed = self._categorical_imputer.transform(df[existing_categorical].astype(str))
                cat_df = pd.DataFrame(cat_imputed, columns=existing_categorical, index=df.index)
                
                if self.config.encoding_strategy == EncodingStrategy.ONEHOT:
                    encoded_dfs = []
                    for col in existing_categorical:
                        if col in self._encoders:
                            encoded = self._encoders[col].transform(cat_df[[col]])
                            feature_names = self._encoders[col].get_feature_names_out([col])
                            encoded_dfs.append(pd.DataFrame(encoded, columns=feature_names, index=df.index))
                    if encoded_dfs:
                        result_dfs.append(pd.concat(encoded_dfs, axis=1))
                        
                elif self.config.encoding_strategy == EncodingStrategy.LABEL:
                    for col in existing_categorical:
                        if col in self._encoders:
                            cat_df[col] = self._encoders[col].transform(cat_df[col])
                    result_dfs.append(cat_df)
                    
                elif self.config.encoding_strategy == EncodingStrategy.ORDINAL:
                    if "_ordinal" in self._encoders:
                        encoded = self._encoders["_ordinal"].transform(cat_df)
                        result_dfs.append(pd.DataFrame(encoded, columns=existing_categorical, index=df.index))
                else:
                    result_dfs.append(cat_df)
        
        # Combine all results
        if result_dfs:
            result = pd.concat(result_dfs, axis=1)
        else:
            result = pd.DataFrame(index=df.index)
        
        # Add back target column
        if target_series is not None:
            result[target_column] = target_series
        
        return result
    
    def fit_transform(
        self, 
        df: pd.DataFrame, 
        target_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: Training DataFrame
            target_column: Target column name
            
        Returns:
            Transformed DataFrame
        """
        self.fit(df, target_column)
        return self.transform(df, target_column)
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the fitted pipeline to disk.
        
        Args:
            path: Directory to save pipeline files
        """
        if not self.fitted:
            raise ValueError("Cannot save unfitted pipeline")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save transformers
        if self._numeric_imputer:
            joblib.dump(self._numeric_imputer, path / "numeric_imputer.pkl")
        if self._scaler:
            joblib.dump(self._scaler, path / "scaler.pkl")
        if self._categorical_imputer:
            joblib.dump(self._categorical_imputer, path / "categorical_imputer.pkl")
        if self._encoders:
            joblib.dump(self._encoders, path / "encoders.pkl")
        
        # Save configuration and metadata
        metadata = {
            "config": {
                "missing_strategy": self.config.missing_strategy.value,
                "scaling_strategy": self.config.scaling_strategy.value,
                "encoding_strategy": self.config.encoding_strategy.value,
                "drop_constant_columns": self.config.drop_constant_columns,
                "handle_high_cardinality": self.config.handle_high_cardinality,
                "outlier_handling": self.config.outlier_handling,
            },
            "columns": {
                "original": self._original_columns,
                "numeric": self._numeric_columns,
                "categorical": self._categorical_columns,
                "dropped": self._dropped_columns,
            },
            "fitted": self.fitted
        }
        
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Pipeline saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "PreprocessingPipeline":
        """
        Load a saved pipeline from disk.
        
        Args:
            path: Directory containing pipeline files
            
        Returns:
            Loaded PreprocessingPipeline instance
        """
        path = Path(path)
        
        # Load metadata
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Create config
        config = PreprocessingConfig(
            missing_strategy=MissingValueStrategy(metadata["config"]["missing_strategy"]),
            scaling_strategy=ScalingStrategy(metadata["config"]["scaling_strategy"]),
            encoding_strategy=EncodingStrategy(metadata["config"]["encoding_strategy"]),
            drop_constant_columns=metadata["config"]["drop_constant_columns"],
            handle_high_cardinality=metadata["config"]["handle_high_cardinality"],
            outlier_handling=metadata["config"]["outlier_handling"],
        )
        
        pipeline = cls(config)
        
        # Load transformers
        if (path / "numeric_imputer.pkl").exists():
            pipeline._numeric_imputer = joblib.load(path / "numeric_imputer.pkl")
        if (path / "scaler.pkl").exists():
            pipeline._scaler = joblib.load(path / "scaler.pkl")
        if (path / "categorical_imputer.pkl").exists():
            pipeline._categorical_imputer = joblib.load(path / "categorical_imputer.pkl")
        if (path / "encoders.pkl").exists():
            pipeline._encoders = joblib.load(path / "encoders.pkl")
        
        # Restore column mappings
        pipeline._original_columns = metadata["columns"]["original"]
        pipeline._numeric_columns = metadata["columns"]["numeric"]
        pipeline._categorical_columns = metadata["columns"]["categorical"]
        pipeline._dropped_columns = metadata["columns"]["dropped"]
        pipeline.fitted = metadata["fitted"]
        
        logger.info(f"Pipeline loaded from {path}")
        return pipeline
    
    def get_feature_names(self) -> List[str]:
        """Get the names of all output features after transformation."""
        if not self.fitted:
            raise ValueError("Pipeline must be fitted first")
        
        names = list(self._numeric_columns)
        
        if self.config.encoding_strategy == EncodingStrategy.ONEHOT and self._encoders:
            for col in self._categorical_columns:
                if col in self._encoders:
                    feature_names = self._encoders[col].get_feature_names_out([col])
                    names.extend(feature_names.tolist())
        else:
            names.extend(self._categorical_columns)
        
        return names
