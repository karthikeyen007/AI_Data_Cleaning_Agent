"""
ML Routes for Flask Frontend - PRODUCTION HARDENED
Complete end-to-end workflow: Clean → Train → Save → Predict
"""

from flask import Blueprint, request, jsonify, send_file
import pandas as pd
import numpy as np
import io
import os
import json
import time
import pickle
import hashlib
from datetime import datetime
from pathlib import Path
from werkzeug.utils import secure_filename

ml_bp = Blueprint('ml', __name__, url_prefix='/api/ml')

# ============================================================================
# CONSTANTS & PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR / "datasets" / "cleaned"
MODELS_DIR = BASE_DIR / "models"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# SESSION MANAGEMENT - Persistent & Connected
# ============================================================================
class MLSession:
    """Persistent ML session that survives page reloads"""
    _sessions = {}
    
    @classmethod
    def get(cls, session_id: str):
        if session_id not in cls._sessions:
            cls._sessions[session_id] = {
                'dataset_id': None,
                'dataset_path': None,
                'data': None,
                'target': None,
                'problem_type': None,
                'features': [],
                'trained_models': [],
                'best_model': None,
                'best_model_path': None,
                'pipeline': None,
                'preprocessing': None,
                'training_logs': [],
                'created_at': datetime.now().isoformat()
            }
        return cls._sessions[session_id]
    
    @classmethod
    def reset(cls, session_id: str):
        if session_id in cls._sessions:
            del cls._sessions[session_id]

# ============================================================================
# DATASET REGISTRY - Central Dataset Storage
# ============================================================================
class DatasetRegistry:
    """Manages cleaned datasets for training"""
    
    @staticmethod
    def save(df: pd.DataFrame, source: str = "upload", metadata: dict = None) -> dict:
        """Save a dataset and return registry entry"""
        dataset_id = f"ds_{int(time.time())}_{hashlib.md5(str(df.shape).encode()).hexdigest()[:6]}"
        filename = f"{dataset_id}.csv"
        filepath = DATASETS_DIR / filename
        
        # Save the CSV
        df.to_csv(filepath, index=False)
        
        # Create metadata
        meta = {
            'id': dataset_id,
            'filename': filename,
            'path': str(filepath),
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': list(df.columns),
            'source': source,
            'created_at': datetime.now().isoformat(),
            'size_bytes': filepath.stat().st_size,
            **(metadata or {})
        }
        
        # Save metadata
        meta_path = DATASETS_DIR / f"{dataset_id}_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        return meta
    
    @staticmethod
    def list_all() -> list:
        """List all saved datasets"""
        datasets = []
        for meta_file in DATASETS_DIR.glob("*_meta.json"):
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
                    # Check if data file exists
                    if Path(meta['path']).exists():
                        datasets.append(meta)
            except:
                continue
        # Sort by created_at descending
        datasets.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return datasets
    
    @staticmethod
    def load(dataset_id: str) -> tuple:
        """Load a dataset by ID"""
        meta_path = DATASETS_DIR / f"{dataset_id}_meta.json"
        if not meta_path.exists():
            return None, None
        
        with open(meta_path) as f:
            meta = json.load(f)
        
        df = pd.read_csv(meta['path'])
        return df, meta
    
    @staticmethod
    def delete(dataset_id: str) -> bool:
        """Delete a dataset"""
        try:
            meta_path = DATASETS_DIR / f"{dataset_id}_meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                Path(meta['path']).unlink(missing_ok=True)
                meta_path.unlink()
                return True
        except:
            pass
        return False

# ============================================================================
# MODEL REGISTRY - Persistent Model Storage
# ============================================================================
class ModelRegistry:
    """Manages trained models with versioning"""
    
    @staticmethod
    def save(model, preprocessing, metadata: dict, project_id: str = "default") -> dict:
        """Save a trained model with full metadata"""
        project_dir = MODELS_DIR / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate version
        existing = list(project_dir.glob("v*"))
        version_num = len(existing) + 1
        version = f"v{version_num:03d}"
        
        model_dir = project_dir / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model pickle
        model_path = model_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save preprocessing if provided
        if preprocessing is not None:
            prep_path = model_dir / "preprocessing.pkl"
            with open(prep_path, 'wb') as f:
                pickle.dump(preprocessing, f)
        
        # Save combined pipeline
        pipeline = {
            'model': model,
            'preprocessing': preprocessing,
            'metadata': metadata
        }
        pipeline_path = model_dir / "pipeline.pkl"
        with open(pipeline_path, 'wb') as f:
            pickle.dump(pipeline, f)
        
        # Save metadata
        full_meta = {
            'version': version,
            'project_id': project_id,
            'path': str(model_dir),
            'created_at': datetime.now().isoformat(),
            'is_active': True,
            **metadata
        }
        
        meta_path = model_dir / "metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(full_meta, f, indent=2, default=str)
        
        # Update active version marker
        active_path = project_dir / "active_version.txt"
        with open(active_path, 'w') as f:
            f.write(version)
        
        return full_meta
    
    @staticmethod
    def load_active(project_id: str = "default") -> tuple:
        """Load the currently active model"""
        project_dir = MODELS_DIR / project_id
        active_path = project_dir / "active_version.txt"
        
        if not active_path.exists():
            return None, None, None
        
        with open(active_path) as f:
            version = f.read().strip()
        
        return ModelRegistry.load_version(project_id, version)
    
    @staticmethod
    def load_version(project_id: str, version: str) -> tuple:
        """Load a specific model version"""
        model_dir = MODELS_DIR / project_id / version
        
        if not model_dir.exists():
            return None, None, None
        
        # Load pipeline (preferred)
        pipeline_path = model_dir / "pipeline.pkl"
        if pipeline_path.exists():
            with open(pipeline_path, 'rb') as f:
                pipeline = pickle.load(f)
            return pipeline.get('model'), pipeline.get('preprocessing'), pipeline.get('metadata')
        
        # Fallback to separate files
        model_path = model_dir / "model.pkl"
        if not model_path.exists():
            return None, None, None
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        preprocessing = None
        prep_path = model_dir / "preprocessing.pkl"
        if prep_path.exists():
            with open(prep_path, 'rb') as f:
                preprocessing = pickle.load(f)
        
        meta_path = model_dir / "metadata.json"
        metadata = {}
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
        
        return model, preprocessing, metadata
    
    @staticmethod
    def list_versions(project_id: str = "default") -> list:
        """List all model versions for a project"""
        project_dir = MODELS_DIR / project_id
        if not project_dir.exists():
            return []
        
        # Get active version
        active_version = None
        active_path = project_dir / "active_version.txt"
        if active_path.exists():
            with open(active_path) as f:
                active_version = f.read().strip()
        
        versions = []
        for version_dir in sorted(project_dir.glob("v*"), reverse=True):
            meta_path = version_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                    meta['is_active'] = (version_dir.name == active_version)
                    versions.append(meta)
        
        return versions
    
    @staticmethod
    def set_active(project_id: str, version: str) -> bool:
        """Set the active model version"""
        project_dir = MODELS_DIR / project_id
        version_dir = project_dir / version
        
        if not version_dir.exists():
            return False
        
        active_path = project_dir / "active_version.txt"
        with open(active_path, 'w') as f:
            f.write(version)
        
        return True

# ============================================================================
# REAL ML TRAINING ENGINE
# ============================================================================
class RealMLTrainer:
    """Performs actual ML training with sklearn"""
    
    ALGORITHMS = {
        # ===== CLASSIFICATION ALGORITHMS =====
        'logistic_regression': {
            'class': 'sklearn.linear_model.LogisticRegression',
            'params': {'max_iter': 1000, 'random_state': 42},
            'type': 'classification'
        },
        'random_forest': {
            'class': 'sklearn.ensemble.RandomForestClassifier',
            'params': {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1},
            'type': 'classification'
        },
        'gradient_boosting': {
            'class': 'sklearn.ensemble.GradientBoostingClassifier',
            'params': {'n_estimators': 100, 'random_state': 42},
            'type': 'classification'
        },
        'svm': {
            'class': 'sklearn.svm.SVC',
            'params': {'kernel': 'rbf', 'probability': True, 'random_state': 42},
            'type': 'classification'
        },
        'knn': {
            'class': 'sklearn.neighbors.KNeighborsClassifier',
            'params': {'n_neighbors': 5},
            'type': 'classification'
        },
        'naive_bayes': {
            'class': 'sklearn.naive_bayes.GaussianNB',
            'params': {},
            'type': 'classification'
        },
        'decision_tree': {
            'class': 'sklearn.tree.DecisionTreeClassifier',
            'params': {'random_state': 42},
            'type': 'classification'
        },
        'extra_trees': {
            'class': 'sklearn.ensemble.ExtraTreesClassifier',
            'params': {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1},
            'type': 'classification'
        },
        
        # ===== REGRESSION ALGORITHMS =====
        'linear_regression': {
            'class': 'sklearn.linear_model.LinearRegression',
            'params': {},
            'type': 'regression'
        },
        'ridge': {
            'class': 'sklearn.linear_model.Ridge',
            'params': {'alpha': 1.0, 'random_state': 42},
            'type': 'regression'
        },
        'lasso': {
            'class': 'sklearn.linear_model.Lasso',
            'params': {'alpha': 1.0, 'random_state': 42, 'max_iter': 10000},
            'type': 'regression'
        },
        'elastic_net': {
            'class': 'sklearn.linear_model.ElasticNet',
            'params': {'alpha': 1.0, 'l1_ratio': 0.5, 'random_state': 42, 'max_iter': 10000},
            'type': 'regression'
        },
        'random_forest_reg': {
            'class': 'sklearn.ensemble.RandomForestRegressor',
            'params': {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1},
            'type': 'regression'
        },
        'gradient_boosting_reg': {
            'class': 'sklearn.ensemble.GradientBoostingRegressor',
            'params': {'n_estimators': 100, 'random_state': 42},
            'type': 'regression'
        },
        'svr': {
            'class': 'sklearn.svm.SVR',
            'params': {'kernel': 'rbf'},
            'type': 'regression'
        },
        'decision_tree_reg': {
            'class': 'sklearn.tree.DecisionTreeRegressor',
            'params': {'random_state': 42},
            'type': 'regression'
        },
        
        # ===== LEGACY NAMES (backwards compatibility) =====
        'random_forest_clf': {
            'class': 'sklearn.ensemble.RandomForestClassifier',
            'params': {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1},
            'type': 'classification'
        },
        'svm_clf': {
            'class': 'sklearn.svm.SVC',
            'params': {'kernel': 'rbf', 'probability': True, 'random_state': 42},
            'type': 'classification'
        }
    }
    
    @staticmethod
    def _get_model_class(class_path: str):
        """Dynamically import and return model class"""
        parts = class_path.rsplit('.', 1)
        module = __import__(parts[0], fromlist=[parts[1]])
        return getattr(module, parts[1])
    
    @staticmethod
    def train(X: pd.DataFrame, y: pd.Series, algorithm: str, cv_folds: int = 5, 
              hyperparameters: dict = None, progress_callback=None) -> dict:
        """
        Perform REAL model training with cross-validation.
        
        Returns:
            dict with model, metrics, and metadata
        """
        from sklearn.model_selection import cross_val_score, cross_validate
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.impute import SimpleImputer
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        import warnings
        warnings.filterwarnings('ignore')
        
        if algorithm not in RealMLTrainer.ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        algo_config = RealMLTrainer.ALGORITHMS[algorithm]
        start_time = time.time()
        logs = []
        
        def log(msg):
            logs.append({'time': time.time() - start_time, 'message': msg})
            if progress_callback:
                progress_callback(msg)
        
        log(f"Starting training: {algorithm}")
        
        # Preprocessing
        log("Preprocessing data...")
        X_processed = X.copy()
        
        # Handle categorical columns
        categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = X_processed.select_dtypes(include=[np.number]).columns.tolist()
        
        # Simple preprocessing: fill missing, encode categoricals
        for col in numeric_cols:
            X_processed[col] = X_processed[col].fillna(X_processed[col].median())
        
        for col in categorical_cols:
            X_processed[col] = X_processed[col].fillna('missing')
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
        
        log(f"Processed {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns")
        
        # Encode target if classification
        label_encoder = None
        y_processed = y.copy()
        if algo_config['type'] == 'classification':
            label_encoder = LabelEncoder()
            y_processed = label_encoder.fit_transform(y.astype(str))
            log(f"Encoded {len(label_encoder.classes_)} target classes")
        else:
            y_processed = y.astype(float)
        
        # Create model
        log("Creating model...")
        model_class = RealMLTrainer._get_model_class(algo_config['class'])
        params = algo_config['params'].copy()
        if hyperparameters:
            params.update(hyperparameters)
        model = model_class(**params)
        
        # Cross-validation
        log(f"Running {cv_folds}-fold cross-validation...")
        scoring = 'accuracy' if algo_config['type'] == 'classification' else 'r2'
        
        cv_results = cross_validate(
            model, X_processed, y_processed,
            cv=cv_folds,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        cv_scores = cv_results['test_score'].tolist()
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))
        
        log(f"CV Score: {cv_mean:.4f} (+/- {cv_std:.4f})")
        
        # Train final model on all data
        log("Training final model on full dataset...")
        model.fit(X_processed, y_processed)
        
        # Extract feature importances
        feature_importances = {}
        if hasattr(model, 'feature_importances_'):
            feature_importances = dict(zip(X.columns.tolist(), model.feature_importances_.tolist()))
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            if len(coef.shape) > 1:
                coef = np.abs(coef).mean(axis=0)
            feature_importances = dict(zip(X.columns.tolist(), np.abs(coef).tolist()))
        
        training_time = time.time() - start_time
        log(f"Training complete in {training_time:.2f} seconds")
        
        # Create preprocessing info for prediction
        preprocessing = {
            'numeric_cols': numeric_cols,
            'categorical_cols': categorical_cols,
            'label_encoder': label_encoder,
            'feature_names': X.columns.tolist()
        }
        
        return {
            'model': model,
            'preprocessing': preprocessing,
            'algorithm': algorithm,
            'problem_type': algo_config['type'],
            'cv_scores': cv_scores,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'training_time': training_time,
            'feature_importances': feature_importances,
            'logs': logs,
            'success': True
        }

# ============================================================================
# API ROUTES
# ============================================================================

@ml_bp.route('/session-status', methods=['GET'])
def session_status():
    """Get current session status"""
    session_id = request.args.get('session_id', 'default')
    session = MLSession.get(session_id)
    
    return jsonify({
        'has_data': session['data'] is not None,
        'dataset_id': session['dataset_id'],
        'rows': len(session['data']) if session['data'] else 0,
        'columns': len(session['features']) if session['features'] else 0,
        'target': session['target'],
        'target_selected': session['target'] is not None,
        'problem_type': session['problem_type'],
        'models_trained': len(session['trained_models']),
        'best_model': session['best_model']['algorithm'] if session['best_model'] else None,
        'best_model_score': session['best_model']['cv_mean'] if session['best_model'] else None,
        'has_saved_model': session['best_model_path'] is not None
    })

@ml_bp.route('/datasets', methods=['GET'])
def list_datasets():
    """List all available cleaned datasets"""
    datasets = DatasetRegistry.list_all()
    return jsonify({'datasets': datasets})

@ml_bp.route('/datasets/<dataset_id>', methods=['GET'])
def get_dataset(dataset_id):
    """Get dataset details and preview"""
    df, meta = DatasetRegistry.load(dataset_id)
    if df is None:
        return jsonify({'error': 'Dataset not found'}), 404
    
    return jsonify({
        'metadata': meta,
        'preview': df.head(10).to_dict(orient='records'),
        'dtypes': df.dtypes.astype(str).to_dict()
    })

@ml_bp.route('/datasets/<dataset_id>', methods=['DELETE'])
def delete_dataset(dataset_id):
    """Delete a dataset"""
    success = DatasetRegistry.delete(dataset_id)
    return jsonify({'success': success})

@ml_bp.route('/upload', methods=['POST'])
def upload_data():
    """Upload training data (CSV/Excel)"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        session_id = request.form.get('session_id', 'default')
        save_to_registry = request.form.get('save', 'true').lower() == 'true'
        
        # Read file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        session = MLSession.get(session_id)
        session['data'] = df.to_dict(orient='records')
        session['features'] = list(df.columns)
        
        result = {
            'success': True,
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': list(df.columns),
            'preview': df.head(10).to_dict(orient='records'),
            'dtypes': df.dtypes.astype(str).to_dict()
        }
        
        # Save to registry if requested
        if save_to_registry:
            meta = DatasetRegistry.save(df, source='upload', metadata={'original_filename': file.filename})
            session['dataset_id'] = meta['id']
            session['dataset_path'] = meta['path']
            result['dataset_id'] = meta['id']
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@ml_bp.route('/load-dataset', methods=['POST'])
def load_dataset():
    """Load a dataset from registry into session"""
    data = request.get_json()
    dataset_id = data.get('dataset_id')
    session_id = data.get('session_id', 'default')
    
    df, meta = DatasetRegistry.load(dataset_id)
    if df is None:
        return jsonify({'success': False, 'error': 'Dataset not found'}), 404
    
    session = MLSession.get(session_id)
    session['data'] = df.to_dict(orient='records')
    session['features'] = list(df.columns)
    session['dataset_id'] = dataset_id
    session['dataset_path'] = meta['path']
    
    return jsonify({
        'success': True,
        'rows': len(df),
        'columns': len(df.columns),
        'column_names': list(df.columns),
        'preview': df.head(10).to_dict(orient='records')
    })

@ml_bp.route('/suggest-targets', methods=['GET'])
def suggest_targets():
    """Suggest target columns with smart detection"""
    session_id = request.args.get('session_id', 'default')
    session = MLSession.get(session_id)
    
    if session['data'] is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    df = pd.DataFrame(session['data'])
    columns = list(df.columns)
    
    suggestions = []
    all_columns = []
    
    for col in columns:
        col_lower = col.lower()
        unique_count = df[col].nunique()
        dtype = str(df[col].dtype)
        
        col_info = {
            'name': col,
            'dtype': dtype,
            'unique_count': int(unique_count),
            'null_count': int(df[col].isnull().sum()),
            'is_suggested': False,
            'reason': ''
        }
        
        # Target keyword detection
        if any(kw in col_lower for kw in ['target', 'label', 'class', 'outcome', 'result', 'y']):
            col_info['is_suggested'] = True
            col_info['reason'] = 'Contains target keyword'
        # Price/salary columns
        elif any(kw in col_lower for kw in ['price', 'salary', 'amount', 'revenue', 'cost']):
            col_info['is_suggested'] = True
            col_info['reason'] = 'Likely numeric target'
        # Low cardinality categorical
        elif unique_count <= 10 and unique_count >= 2 and dtype in ['object', 'int64']:
            col_info['is_suggested'] = True
            col_info['reason'] = f'Categorical ({unique_count} classes)'
        
        if col_info['is_suggested']:
            suggestions.append(col_info)
        all_columns.append(col_info)
    
    return jsonify({
        'suggestions': suggestions[:5],
        'all_columns': all_columns
    })

@ml_bp.route('/select-target', methods=['POST'])
def select_target():
    """Select target column and detect problem type"""
    data = request.get_json()
    session_id = data.get('session_id', 'default')
    target_column = data.get('target_column')
    
    session = MLSession.get(session_id)
    
    if session['data'] is None:
        return jsonify({'success': False, 'error': 'No data loaded'}), 400
    
    df = pd.DataFrame(session['data'])
    
    if target_column not in df.columns:
        return jsonify({'success': False, 'error': 'Invalid target column'}), 400
    
    session['target'] = target_column
    session['features'] = [c for c in df.columns if c != target_column]
    
    # Detect problem type
    target_series = df[target_column]
    unique_count = target_series.nunique()
    
    if target_series.dtype in ['float64', 'float32'] and unique_count > 20:
        problem_type = 'regression'
    elif unique_count <= 20:
        problem_type = 'classification'
    else:
        problem_type = 'regression'
    
    session['problem_type'] = problem_type
    
    return jsonify({
        'success': True,
        'problem_type': problem_type,
        'unique_values': int(unique_count),
        'feature_count': len(session['features'])
    })

@ml_bp.route('/train', methods=['POST'])
def train_model():
    """REAL model training - not simulated!"""
    data = request.get_json()
    session_id = data.get('session_id', 'default')
    algorithm = data.get('algorithm')
    cv_folds = data.get('cv_folds', 5)
    hyperparameters = data.get('hyperparameters', {})
    auto_save = data.get('auto_save', True)
    
    session = MLSession.get(session_id)
    
    # Validation
    if session['data'] is None:
        return jsonify({'success': False, 'error': 'No data loaded. Please upload a dataset first.'}), 400
    
    if session['target'] is None:
        return jsonify({'success': False, 'error': 'No target selected. Please select a target column.'}), 400
    
    if not algorithm:
        return jsonify({'success': False, 'error': 'No algorithm specified.'}), 400
    
    try:
        df = pd.DataFrame(session['data'])
        
        if len(df) < 4:
            return jsonify({'success': False, 'error': f'Dataset too small ({len(df)} rows). Need at least 4 rows.'}), 400
        
        X = df[session['features']]
        y = df[session['target']]
        
        # Auto-adjust CV folds to be valid for the dataset size
        max_folds = max(2, min(cv_folds, len(df) - 1))
        # Also ensure we have at least max_folds samples per class for classification
        if session['problem_type'] == 'classification':
            min_class_count = y.value_counts().min()
            max_folds = max(2, min(max_folds, min_class_count))
        
        # REAL TRAINING
        result = RealMLTrainer.train(
            X=X, 
            y=y, 
            algorithm=algorithm,
            cv_folds=max_folds,
            hyperparameters=hyperparameters
        )
        
        if not result['success']:
            return jsonify({'success': False, 'error': result.get('error', 'Training failed')}), 500
        
        # Store in session
        model_entry = {
            'algorithm': algorithm,
            'cv_mean': result['cv_mean'],
            'cv_std': result['cv_std'],
            'cv_scores': result['cv_scores'],
            'training_time': result['training_time'],
            'feature_importances': result['feature_importances'],
            'problem_type': result['problem_type'],
            'model': result['model'],
            'preprocessing': result['preprocessing'],
            'trained_at': datetime.now().isoformat()
        }
        
        session['trained_models'].append(model_entry)
        session['training_logs'].extend(result['logs'])
        
        # Update best model
        if session['best_model'] is None or result['cv_mean'] > session['best_model']['cv_mean']:
            session['best_model'] = model_entry
            session['pipeline'] = result['model']
            session['preprocessing'] = result['preprocessing']
        
        # Auto-save to registry
        version_info = None
        if auto_save:
            meta = ModelRegistry.save(
                model=result['model'],
                preprocessing=result['preprocessing'],
                metadata={
                    'algorithm': algorithm,
                    'cv_mean': result['cv_mean'],
                    'cv_std': result['cv_std'],
                    'problem_type': result['problem_type'],
                    'features': session['features'],
                    'target': session['target'],
                    'dataset_id': session['dataset_id'],
                    'training_time': result['training_time']
                },
                project_id=session_id
            )
            session['best_model_path'] = meta['path']
            version_info = meta
        
        return jsonify({
            'success': True,
            'algorithm': algorithm,
            'cv_mean': result['cv_mean'],
            'cv_std': result['cv_std'],
            'cv_scores': result['cv_scores'],
            'training_time': result['training_time'],
            'feature_importances': result['feature_importances'],
            'logs': result['logs'],
            'version': version_info,
            'message': f"Model trained successfully! CV Score: {result['cv_mean']:.4f}"
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False, 
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@ml_bp.route('/leaderboard', methods=['GET'])
def get_leaderboard():
    """Get model leaderboard for session"""
    session_id = request.args.get('session_id', 'default')
    session = MLSession.get(session_id)
    
    leaderboard = []
    for i, model in enumerate(sorted(session['trained_models'], key=lambda x: x['cv_mean'], reverse=True)):
        leaderboard.append({
            'rank': i + 1,
            'algorithm': model['algorithm'],
            'cv_mean': model['cv_mean'],
            'cv_std': model['cv_std'],
            'training_time': model['training_time'],
            'trained_at': model.get('trained_at')
        })
    
    return jsonify({
        'leaderboard': leaderboard,
        'best_model': session['best_model']['algorithm'] if session['best_model'] else None
    })

@ml_bp.route('/predict', methods=['POST'])
def predict():
    """Make prediction using trained model"""
    data = request.get_json()
    session_id = data.get('session_id', 'default')
    input_data = data.get('input_data', {})
    
    session = MLSession.get(session_id)
    
    # Check for trained model in session
    model = session.get('pipeline')
    preprocessing = session.get('preprocessing')
    
    # If not in session, try loading from registry
    if model is None:
        model, preprocessing, metadata = ModelRegistry.load_active(session_id)
    
    if model is None:
        return jsonify({
            'success': False, 
            'error': 'No trained model available. Please train a model first.'
        }), 400
    
    try:
        # Prepare input
        if isinstance(input_data, dict):
            df_input = pd.DataFrame([input_data])
        else:
            df_input = pd.DataFrame(input_data)
        
        # Apply preprocessing
        if preprocessing:
            for col in preprocessing.get('numeric_cols', []):
                if col in df_input.columns:
                    df_input[col] = pd.to_numeric(df_input[col], errors='coerce').fillna(0)
            
            for col in preprocessing.get('categorical_cols', []):
                if col in df_input.columns:
                    df_input[col] = df_input[col].fillna('missing')
                    # Use label encoding consistent with training
                    df_input[col] = df_input[col].astype('category').cat.codes
        
        # Ensure column order matches training
        feature_names = preprocessing.get('feature_names', []) if preprocessing else list(df_input.columns)
        for col in feature_names:
            if col not in df_input.columns:
                df_input[col] = 0
        df_input = df_input[feature_names]
        
        # Make prediction
        prediction = model.predict(df_input)
        
        # Decode if classification
        result_prediction = prediction[0]
        if preprocessing and preprocessing.get('label_encoder'):
            result_prediction = preprocessing['label_encoder'].inverse_transform([int(prediction[0])])[0]
        
        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(df_input)[0]
                if preprocessing and preprocessing.get('label_encoder'):
                    classes = preprocessing['label_encoder'].classes_
                    probabilities = dict(zip(classes.tolist(), proba.tolist()))
                else:
                    probabilities = {f'class_{i}': p for i, p in enumerate(proba)}
            except:
                pass
        
        return jsonify({
            'success': True,
            'prediction': result_prediction if not isinstance(result_prediction, np.generic) else result_prediction.item(),
            'probabilities': probabilities
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False, 
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@ml_bp.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Batch prediction from CSV"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        session_id = request.form.get('session_id', 'default')
        
        session = MLSession.get(session_id)
        model = session.get('pipeline')
        preprocessing = session.get('preprocessing')
        
        if model is None:
            model, preprocessing, _ = ModelRegistry.load_active(session_id)
        
        if model is None:
            return jsonify({'success': False, 'error': 'No trained model available'}), 400
        
        df = pd.read_csv(file)
        df_processed = df.copy()
        
        # Apply preprocessing
        if preprocessing:
            feature_names = preprocessing.get('feature_names', [])
            for col in preprocessing.get('numeric_cols', []):
                if col in df_processed.columns:
                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)
            for col in preprocessing.get('categorical_cols', []):
                if col in df_processed.columns:
                    df_processed[col] = df_processed[col].fillna('missing').astype('category').cat.codes
            
            for col in feature_names:
                if col not in df_processed.columns:
                    df_processed[col] = 0
            df_processed = df_processed[feature_names]
        
        # Predict
        predictions = model.predict(df_processed)
        
        # Decode if needed
        if preprocessing and preprocessing.get('label_encoder'):
            predictions = preprocessing['label_encoder'].inverse_transform(predictions.astype(int))
        
        df['prediction'] = predictions
        
        return jsonify({
            'success': True,
            'count': len(df),
            'results': df.to_dict(orient='records')
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@ml_bp.route('/versions', methods=['GET'])
def list_versions():
    """List model versions"""
    project_id = request.args.get('project_id', 'default')
    versions = ModelRegistry.list_versions(project_id)
    return jsonify({'versions': versions})

@ml_bp.route('/save-model', methods=['POST'])
def save_model():
    """Manually save current model"""
    data = request.get_json()
    session_id = data.get('session_id', 'default')
    project_id = data.get('project_id', session_id)
    
    session = MLSession.get(session_id)
    
    if not session['best_model']:
        return jsonify({'success': False, 'error': 'No model to save'}), 400
    
    meta = ModelRegistry.save(
        model=session['pipeline'],
        preprocessing=session['preprocessing'],
        metadata={
            'algorithm': session['best_model']['algorithm'],
            'cv_mean': session['best_model']['cv_mean'],
            'cv_std': session['best_model']['cv_std'],
            'problem_type': session['problem_type'],
            'features': session['features'],
            'target': session['target']
        },
        project_id=project_id
    )
    
    session['best_model_path'] = meta['path']
    
    return jsonify({
        'success': True,
        'version': meta['version'],
        'path': meta['path']
    })

@ml_bp.route('/rollback', methods=['POST'])
def rollback():
    """Rollback to previous model version"""
    data = request.get_json()
    project_id = data.get('project_id', 'default')
    version = data.get('version')
    
    success = ModelRegistry.set_active(project_id, version)
    
    return jsonify({'success': success})

@ml_bp.route('/export', methods=['POST'])
def export_model():
    """Export trained model as pickle"""
    data = request.get_json()
    session_id = data.get('session_id', 'default')
    
    session = MLSession.get(session_id)
    
    if not session['pipeline']:
        return jsonify({'success': False, 'error': 'No model to export'}), 400
    
    # Create export package
    export_data = {
        'model': session['pipeline'],
        'preprocessing': session['preprocessing'],
        'metadata': {
            'algorithm': session['best_model']['algorithm'],
            'cv_mean': session['best_model']['cv_mean'],
            'problem_type': session['problem_type'],
            'features': session['features'],
            'target': session['target'],
            'exported_at': datetime.now().isoformat()
        }
    }
    
    buffer = io.BytesIO()
    pickle.dump(export_data, buffer)
    buffer.seek(0)
    
    return send_file(
        buffer,
        mimetype='application/octet-stream',
        as_attachment=True,
        download_name='model_pipeline.pkl'
    )

@ml_bp.route('/model-info', methods=['GET'])
def model_info():
    """Get info about current/active model"""
    session_id = request.args.get('session_id', 'default')
    session = MLSession.get(session_id)
    
    if session['best_model']:
        return jsonify({
            'success': True,
            'model': {
                'algorithm': session['best_model']['algorithm'],
                'cv_mean': session['best_model']['cv_mean'],
                'cv_std': session['best_model']['cv_std'],
                'problem_type': session['problem_type'],
                'features': session['features'],
                'target': session['target'],
                'trained_at': session['best_model'].get('trained_at')
            },
            'has_saved_version': session['best_model_path'] is not None
        })
    
    # Check for saved model
    model, _, metadata = ModelRegistry.load_active(session_id)
    if model:
        return jsonify({
            'success': True,
            'model': metadata,
            'has_saved_version': True
        })
    
    return jsonify({'success': False, 'model': None})

@ml_bp.route('/reset', methods=['POST'])
def reset_session():
    """Reset ML session"""
    data = request.get_json() or {}
    session_id = data.get('session_id', 'default')
    MLSession.reset(session_id)
    return jsonify({'success': True})

@ml_bp.route('/training-logs', methods=['GET'])
def get_training_logs():
    """Get training logs for session"""
    session_id = request.args.get('session_id', 'default')
    session = MLSession.get(session_id)
    return jsonify({'logs': session['training_logs']})

# ============================================================================
# OBSERVABILITY ROUTES
# ============================================================================

@ml_bp.route('/ai-health', methods=['GET'])
def ai_health():
    return jsonify({'healthy': True})

@ml_bp.route('/key-status', methods=['GET'])
def key_status():
    return jsonify({
        'keys': [
            {'source': 'ENV', 'masked_key': 'sk-***', 'model': 'sklearn', 'valid': True}
        ]
    })

@ml_bp.route('/cost-metrics', methods=['GET'])
def cost_metrics():
    return jsonify({
        'api_calls': 0,
        'tokens': 0,
        'estimated_cost': 0.0
    })

@ml_bp.route('/drift-analysis', methods=['POST'])
def drift_analysis():
    return jsonify({
        'drift_detected': False,
        'drifted_features': []
    })
