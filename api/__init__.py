"""
API Package - FastAPI Routes for AutoML Platform

Production-grade REST API for:
- Data upload and cleaning
- ML training and evaluation
- Model management
- Hyperparameter tuning
"""

from .routes import router, mlops_router

__all__ = ["router", "mlops_router"]
