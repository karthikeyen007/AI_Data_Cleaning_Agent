"""
Model Manager Service

Production-grade model versioning and lifecycle management:
- Model storage and retrieval
- Version management with rollback
- Full pipeline export (preprocessing + model)
- Model comparison across versions
- Retraining with version tracking
"""

import logging
import shutil
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import numpy as np
import json
import hashlib
import joblib

from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Status of a model version."""
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"


@dataclass
class ModelVersion:
    """Represents a single model version."""
    version_id: str
    algorithm: str
    created_at: str
    status: ModelStatus
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    training_data_hash: str
    feature_names: List[str]
    target_name: str
    notes: str = ""
    parent_version: Optional[str] = None


@dataclass
class ModelProject:
    """Represents a model project with multiple versions."""
    project_id: str
    user_id: str
    name: str
    description: str
    created_at: str
    updated_at: str
    current_version: Optional[str]
    versions: List[str]
    problem_type: str
    status: str


class ModelManager:
    """
    Production-grade model lifecycle management.
    
    Handles:
    - Model storage with versioning
    - Pipeline bundling (preprocessing + model)
    - Version comparison and rollback
    - Retraining workflows
    """
    
    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """
        Initialize the model manager.
        
        Args:
            base_path: Base directory for model storage
        """
        self.base_path = Path(base_path) if base_path else Path("models")
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self._projects_cache: Dict[str, ModelProject] = {}
        
        logger.info(f"ModelManager initialized with base path: {self.base_path}")
    
    def _generate_version_id(self) -> str:
        """Generate a unique version ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"v_{timestamp}"
    
    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """Compute a hash of the training data for tracking."""
        data_str = pd.util.hash_pandas_object(df).values.tobytes()
        return hashlib.md5(data_str).hexdigest()[:12]
    
    def _get_project_path(self, user_id: str, project_id: str) -> Path:
        """Get the path for a project."""
        return self.base_path / user_id / project_id
    
    def _get_version_path(self, user_id: str, project_id: str, version_id: str) -> Path:
        """Get the path for a specific version."""
        return self._get_project_path(user_id, project_id) / version_id
    
    def _load_project_metadata(self, user_id: str, project_id: str) -> Optional[ModelProject]:
        """Load project metadata from disk."""
        project_path = self._get_project_path(user_id, project_id)
        metadata_path = project_path / "project.json"
        
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, "r") as f:
            data = json.load(f)
        
        return ModelProject(
            project_id=data["project_id"],
            user_id=data["user_id"],
            name=data["name"],
            description=data["description"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            current_version=data.get("current_version"),
            versions=data.get("versions", []),
            problem_type=data.get("problem_type", "unknown"),
            status=data.get("status", "active")
        )
    
    def _save_project_metadata(self, project: ModelProject) -> None:
        """Save project metadata to disk."""
        project_path = self._get_project_path(project.user_id, project.project_id)
        project_path.mkdir(parents=True, exist_ok=True)
        
        data = {
            "project_id": project.project_id,
            "user_id": project.user_id,
            "name": project.name,
            "description": project.description,
            "created_at": project.created_at,
            "updated_at": project.updated_at,
            "current_version": project.current_version,
            "versions": project.versions,
            "problem_type": project.problem_type,
            "status": project.status
        }
        
        with open(project_path / "project.json", "w") as f:
            json.dump(data, f, indent=2)
    
    def create_project(
        self,
        user_id: str,
        project_id: str,
        name: str,
        description: str = "",
        problem_type: str = "unknown"
    ) -> ModelProject:
        """
        Create a new model project.
        
        Args:
            user_id: User identifier
            project_id: Project identifier
            name: Project name
            description: Project description
            problem_type: Type of ML problem
            
        Returns:
            Created ModelProject
        """
        now = datetime.now().isoformat()
        
        project = ModelProject(
            project_id=project_id,
            user_id=user_id,
            name=name,
            description=description,
            created_at=now,
            updated_at=now,
            current_version=None,
            versions=[],
            problem_type=problem_type,
            status="active"
        )
        
        self._save_project_metadata(project)
        self._projects_cache[f"{user_id}/{project_id}"] = project
        
        logger.info(f"Created project: {user_id}/{project_id}")
        
        return project
    
    def get_project(self, user_id: str, project_id: str) -> Optional[ModelProject]:
        """
        Get a project by ID.
        
        Args:
            user_id: User identifier
            project_id: Project identifier
            
        Returns:
            ModelProject or None if not found
        """
        cache_key = f"{user_id}/{project_id}"
        
        if cache_key in self._projects_cache:
            return self._projects_cache[cache_key]
        
        project = self._load_project_metadata(user_id, project_id)
        if project:
            self._projects_cache[cache_key] = project
        
        return project
    
    def save_model(
        self,
        user_id: str,
        project_id: str,
        model: BaseEstimator,
        metrics: Dict[str, float],
        hyperparameters: Dict[str, Any],
        feature_names: List[str],
        target_name: str,
        training_data: Optional[pd.DataFrame] = None,
        preprocessing_pipeline: Any = None,
        notes: str = "",
        parent_version: Optional[str] = None
    ) -> ModelVersion:
        """
        Save a trained model with versioning.
        
        Args:
            user_id: User identifier
            project_id: Project identifier
            model: Trained model
            metrics: Model metrics
            hyperparameters: Hyperparameters used
            feature_names: List of feature names
            target_name: Target column name
            training_data: Training data (for hash)
            preprocessing_pipeline: Preprocessing pipeline
            notes: Version notes
            parent_version: Parent version ID (for retraining)
            
        Returns:
            Created ModelVersion
        """
        # Ensure project exists
        project = self.get_project(user_id, project_id)
        if project is None:
            project = self.create_project(user_id, project_id, project_id)
        
        # Generate version ID
        version_id = self._generate_version_id()
        
        # Compute data hash
        data_hash = ""
        if training_data is not None:
            data_hash = self._compute_data_hash(training_data)
        
        # Get algorithm name
        algorithm = type(model).__name__
        
        # Create version
        version = ModelVersion(
            version_id=version_id,
            algorithm=algorithm,
            created_at=datetime.now().isoformat(),
            status=ModelStatus.TRAINED,
            metrics=metrics,
            hyperparameters=hyperparameters,
            training_data_hash=data_hash,
            feature_names=feature_names,
            target_name=target_name,
            notes=notes,
            parent_version=parent_version
        )
        
        # Create version directory
        version_path = self._get_version_path(user_id, project_id, version_id)
        version_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(model, version_path / "model.pkl")
        
        # Save preprocessing if provided
        if preprocessing_pipeline is not None:
            joblib.dump(preprocessing_pipeline, version_path / "preprocessing.pkl")
        
        # Save version metadata
        version_data = {
            "version_id": version.version_id,
            "algorithm": version.algorithm,
            "created_at": version.created_at,
            "status": version.status.value,
            "metrics": version.metrics,
            "hyperparameters": version.hyperparameters,
            "training_data_hash": version.training_data_hash,
            "feature_names": version.feature_names,
            "target_name": version.target_name,
            "notes": version.notes,
            "parent_version": version.parent_version
        }
        
        with open(version_path / "version.json", "w") as f:
            json.dump(version_data, f, indent=2)
        
        # Save config for reproducibility
        config = {
            "feature_names": feature_names,
            "target_name": target_name,
            "hyperparameters": hyperparameters
        }
        
        with open(version_path / "config.yaml", "w") as f:
            import yaml
            yaml.dump(config, f, default_flow_style=False)
        
        # Update project
        project.versions.append(version_id)
        project.current_version = version_id
        project.updated_at = datetime.now().isoformat()
        self._save_project_metadata(project)
        
        logger.info(f"Saved model version: {user_id}/{project_id}/{version_id}")
        
        return version
    
    def load_model(
        self,
        user_id: str,
        project_id: str,
        version_id: Optional[str] = None
    ) -> Tuple[BaseEstimator, Dict[str, Any], Any]:
        """
        Load a model from storage.
        
        Args:
            user_id: User identifier
            project_id: Project identifier
            version_id: Version ID (default: current version)
            
        Returns:
            Tuple of (model, metadata, preprocessing_pipeline)
        """
        project = self.get_project(user_id, project_id)
        if project is None:
            raise ValueError(f"Project not found: {user_id}/{project_id}")
        
        if version_id is None:
            version_id = project.current_version
        
        if version_id is None:
            raise ValueError("No version available")
        
        version_path = self._get_version_path(user_id, project_id, version_id)
        
        if not version_path.exists():
            raise ValueError(f"Version not found: {version_id}")
        
        # Load model
        model = joblib.load(version_path / "model.pkl")
        
        # Load metadata
        with open(version_path / "version.json", "r") as f:
            metadata = json.load(f)
        
        # Load preprocessing if exists
        preprocessing = None
        if (version_path / "preprocessing.pkl").exists():
            preprocessing = joblib.load(version_path / "preprocessing.pkl")
        
        logger.info(f"Loaded model: {user_id}/{project_id}/{version_id}")
        
        return model, metadata, preprocessing
    
    def get_version(
        self,
        user_id: str,
        project_id: str,
        version_id: str
    ) -> Optional[ModelVersion]:
        """
        Get version metadata.
        
        Args:
            user_id: User identifier
            project_id: Project identifier
            version_id: Version ID
            
        Returns:
            ModelVersion or None
        """
        version_path = self._get_version_path(user_id, project_id, version_id)
        
        if not (version_path / "version.json").exists():
            return None
        
        with open(version_path / "version.json", "r") as f:
            data = json.load(f)
        
        return ModelVersion(
            version_id=data["version_id"],
            algorithm=data["algorithm"],
            created_at=data["created_at"],
            status=ModelStatus(data["status"]),
            metrics=data["metrics"],
            hyperparameters=data["hyperparameters"],
            training_data_hash=data["training_data_hash"],
            feature_names=data["feature_names"],
            target_name=data["target_name"],
            notes=data.get("notes", ""),
            parent_version=data.get("parent_version")
        )
    
    def list_versions(
        self,
        user_id: str,
        project_id: str
    ) -> List[ModelVersion]:
        """
        List all versions for a project.
        
        Args:
            user_id: User identifier
            project_id: Project identifier
            
        Returns:
            List of ModelVersion objects
        """
        project = self.get_project(user_id, project_id)
        if project is None:
            return []
        
        versions = []
        for version_id in project.versions:
            version = self.get_version(user_id, project_id, version_id)
            if version:
                versions.append(version)
        
        # Sort by created_at descending
        versions.sort(key=lambda v: v.created_at, reverse=True)
        
        return versions
    
    def compare_versions(
        self,
        user_id: str,
        project_id: str,
        version_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Compare multiple versions.
        
        Args:
            user_id: User identifier
            project_id: Project identifier
            version_ids: List of version IDs to compare
            
        Returns:
            List of comparison results
        """
        comparisons = []
        
        for version_id in version_ids:
            version = self.get_version(user_id, project_id, version_id)
            if version:
                comparisons.append({
                    "version_id": version.version_id,
                    "algorithm": version.algorithm,
                    "created_at": version.created_at,
                    "metrics": version.metrics,
                    "hyperparameters": version.hyperparameters,
                    "notes": version.notes
                })
        
        return comparisons
    
    def rollback(
        self,
        user_id: str,
        project_id: str,
        target_version_id: str
    ) -> bool:
        """
        Rollback to a previous version.
        
        Args:
            user_id: User identifier
            project_id: Project identifier
            target_version_id: Version ID to rollback to
            
        Returns:
            True if successful
        """
        project = self.get_project(user_id, project_id)
        if project is None:
            raise ValueError(f"Project not found: {user_id}/{project_id}")
        
        if target_version_id not in project.versions:
            raise ValueError(f"Version not found: {target_version_id}")
        
        # Update current version
        project.current_version = target_version_id
        project.updated_at = datetime.now().isoformat()
        self._save_project_metadata(project)
        
        logger.info(f"Rolled back to version: {target_version_id}")
        
        return True
    
    def delete_version(
        self,
        user_id: str,
        project_id: str,
        version_id: str
    ) -> bool:
        """
        Delete a model version.
        
        Args:
            user_id: User identifier
            project_id: Project identifier
            version_id: Version ID to delete
            
        Returns:
            True if successful
        """
        project = self.get_project(user_id, project_id)
        if project is None:
            raise ValueError(f"Project not found: {user_id}/{project_id}")
        
        if version_id not in project.versions:
            return False
        
        # Cannot delete current version
        if version_id == project.current_version:
            raise ValueError("Cannot delete current version. Rollback first.")
        
        # Delete version directory
        version_path = self._get_version_path(user_id, project_id, version_id)
        if version_path.exists():
            shutil.rmtree(version_path)
        
        # Update project
        project.versions.remove(version_id)
        project.updated_at = datetime.now().isoformat()
        self._save_project_metadata(project)
        
        logger.info(f"Deleted version: {version_id}")
        
        return True
    
    def export_model(
        self,
        user_id: str,
        project_id: str,
        version_id: Optional[str] = None,
        output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Export a model package for deployment.
        
        Args:
            user_id: User identifier
            project_id: Project identifier
            version_id: Version ID (default: current)
            output_path: Output directory
            
        Returns:
            Path to exported package
        """
        project = self.get_project(user_id, project_id)
        if project is None:
            raise ValueError(f"Project not found: {user_id}/{project_id}")
        
        if version_id is None:
            version_id = project.current_version
        
        version_path = self._get_version_path(user_id, project_id, version_id)
        
        if output_path is None:
            output_path = Path("exports") / f"{project_id}_{version_id}"
        else:
            output_path = Path(output_path)
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Copy all model files
        for file in version_path.iterdir():
            shutil.copy2(file, output_path / file.name)
        
        # Create deployment manifest
        manifest = {
            "project_id": project_id,
            "version_id": version_id,
            "exported_at": datetime.now().isoformat(),
            "files": [f.name for f in output_path.iterdir()]
        }
        
        with open(output_path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Exported model to: {output_path}")
        
        return output_path
    
    def create_inference_pipeline(
        self,
        user_id: str,
        project_id: str,
        version_id: Optional[str] = None
    ) -> "InferencePipeline":
        """
        Create an inference pipeline from a saved model.
        
        Args:
            user_id: User identifier
            project_id: Project identifier
            version_id: Version ID
            
        Returns:
            InferencePipeline ready for predictions
        """
        model, metadata, preprocessing = self.load_model(
            user_id, project_id, version_id
        )
        
        return InferencePipeline(
            model=model,
            preprocessing=preprocessing,
            feature_names=metadata["feature_names"],
            target_name=metadata["target_name"]
        )


class InferencePipeline:
    """
    Ready-to-use inference pipeline.
    
    Bundles preprocessing and model for seamless predictions.
    """
    
    def __init__(
        self,
        model: BaseEstimator,
        preprocessing: Any,
        feature_names: List[str],
        target_name: str
    ):
        """
        Initialize inference pipeline.
        
        Args:
            model: Trained model
            preprocessing: Preprocessing pipeline
            feature_names: Expected feature names
            target_name: Target column name
        """
        self.model = model
        self.preprocessing = preprocessing
        self.feature_names = feature_names
        self.target_name = target_name
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predictions array
        """
        # Apply preprocessing if available
        if self.preprocessing is not None:
            X_processed = self.preprocessing.transform(X)
        else:
            X_processed = X
        
        return self.model.predict(X_processed)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities (classification only).
        
        Args:
            X: Input features
            
        Returns:
            Probabilities array
        """
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model does not support predict_proba")
        
        if self.preprocessing is not None:
            X_processed = self.preprocessing.transform(X)
        else:
            X_processed = X
        
        return self.model.predict_proba(X_processed)
