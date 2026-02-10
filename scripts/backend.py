import sys
import os
import pandas as pd
import io
import aiohttp
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
from pydantic import BaseModel
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure the scripts folder is in Python's path
script_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(script_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
from ai_agent import AIAgent  # Import AI Agent
from data_cleaning import DataCleaning  # Import Rule-Based Data Cleaning

def sanitize_df_for_json(df):
    """Replace NaN/Inf with None for JSON compatibility."""
    return df.replace({np.nan: None, np.inf: None, -np.inf: None})

# Import new AutoML API routers
try:
    from api.routes import router as automl_router, mlops_router
    from api.production_routes import production_router
    AUTOML_AVAILABLE = True
except ImportError as e:
    print(f"Warning: AutoML API not available - {e}")
    AUTOML_AVAILABLE = False

# Import multi-key validation and DataSourceType
try:
    from euri_client import KeyValidator, APIKeyMasker, DataSourceType
    MULTIKEY_AVAILABLE = True
except ImportError:
    from euri_client import DataSourceType
    MULTIKEY_AVAILABLE = False


# ==============================================================================
# STARTUP KEY VALIDATION
# ==============================================================================
def validate_api_keys():
    """Validate API keys at startup (warning mode, not blocking)."""
    if not MULTIKEY_AVAILABLE:
        print("⚠️ Multi-key validation not available")
        return
    
    print("\n" + "=" * 60)
    print("🔐 API KEY VALIDATION")
    print("=" * 60)
    
    is_valid, issues = KeyValidator.validate_all(strict=False)
    
    if is_valid:
        print("✅ All API keys configured correctly")
    else:
        print("⚠️ API Key Configuration Issues:")
        for issue in issues:
            print(f"   • {issue}")
        print("\n⚠️ Some features may not work until keys are configured.")
        print("   Set the required keys in your .env file.")
    
    print("=" * 60 + "\n")

# Run validation on import
validate_api_keys()


app = FastAPI(
    title="AI Data Cleaning + AutoML Platform",
    description="Production-grade FastAPI backend for AI-powered data cleaning and AutoML using Euri API",
    version="2.1.0"
)

# Add CORS middleware for Flask frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register AutoML API routers if available
if AUTOML_AVAILABLE:
    app.include_router(automl_router)
    app.include_router(mlops_router)
    app.include_router(production_router)
    print("✅ AutoML API v2 endpoints registered")
    print("✅ Production endpoints registered (async, validation, explainability, cost)")


# ==============================================================================
# API KEY STATUS ENDPOINT (Admin/Monitoring)
# ==============================================================================
@app.get("/api/key-status")
async def get_key_status():
    """
    Get API key configuration status.
    
    SECURITY NOTE: This endpoint NEVER returns actual API keys.
    Only shows whether keys are configured and masked previews.
    """
    if not MULTIKEY_AVAILABLE:
        return {"error": "Multi-key client not available"}
    
    try:
        from euri_client import MultiKeyEuriClient
        client = MultiKeyEuriClient(validate_on_init=False)
        return {
            "status": "ok",
            "keys": client.get_key_status(),
            "note": "Keys are masked for security. Only configuration status is shown."
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# Initialize AI agent and rule-based data cleaner
ai_agent = AIAgent()
cleaner = DataCleaning()

# ------------------------ CSV / Excel Cleaning Endpoint ------------------------

@app.post("/clean-data")
async def clean_data(file: UploadFile = File(...)):
    """Receives file from UI, cleans it using rule-based & AI methods, and returns cleaned JSON."""
    try:
        contents = await file.read()
        file_extension = file.filename.split(".")[-1]

        # Load file into Pandas DataFrame
        if file_extension == "csv":
            df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        elif file_extension == "xlsx":
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV or Excel.")

        # Step 1: Rule-Based Cleaning (always works)
        df_cleaned = cleaner.clean_data(df)
        
        ai_used = False
        df_final = df_cleaned

        # Step 2: Try AI-Powered Cleaning (optional - may fail if API unavailable)
        try:
            df_ai_cleaned = ai_agent.process_data(df_cleaned, source_type=DataSourceType.UPLOAD)
            df_final = df_ai_cleaned
            ai_used = True
        except Exception as ai_error:
            # AI failed (403, timeout, etc.) - use rule-based result
            print(f"⚠️ AI cleaning unavailable, using rule-based cleaning only: {ai_error}")
            df_final = df_cleaned
        
        # Sanitize for JSON
        df_final = sanitize_df_for_json(df_final)

        return {
            "cleaned_data": df_final.to_dict(orient="records"),
            "ai_enhanced": ai_used,
            "message": "Cleaned with AI" if ai_used else "Cleaned with rule-based methods (AI unavailable)"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# ------------------------ Database Query Cleaning Endpoint ------------------------

class DBQuery(BaseModel):
    db_url: str
    query: str

@app.post("/clean-db")
async def clean_db(query: DBQuery):
    """Fetches data from a database, cleans it using AI, and returns cleaned JSON."""
    try:
        engine = create_engine(query.db_url)
        df = pd.read_sql(query.query, engine)

        # Step 1: Rule-Based Cleaning
        df_cleaned = cleaner.clean_data(df)

        # Step 2: AI-Powered Cleaning (Database Source)
        df_ai_cleaned = ai_agent.process_data(df_cleaned, source_type=DataSourceType.DATABASE)
        
        # Sanitize for JSON
        df_final = sanitize_df_for_json(df_ai_cleaned)

        return {"cleaned_data": df_final.to_dict(orient="records")}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data from database: {str(e)}")

# ------------------------ API Data Cleaning Endpoint ------------------------

class APIRequest(BaseModel):
    api_url: str

@app.post("/clean-api")
async def clean_api(api_request: APIRequest):
    """Fetches data from an API, cleans it using AI, and returns cleaned JSON."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(api_request.api_url) as response:
                if response.status != 200:
                    raise HTTPException(status_code=400, detail="Failed to fetch data from API.")
                
                data = await response.json()
                df = pd.DataFrame(data)

                # Step 1: Rule-Based Cleaning
                df_cleaned = cleaner.clean_data(df)

                # Step 2: AI-Powered Cleaning (API Source)
                df_ai_cleaned = ai_agent.process_data(df_cleaned, source_type=DataSourceType.API)
                
                # Sanitize for JSON
                df_final = sanitize_df_for_json(df_ai_cleaned)

                return {"cleaned_data": df_final.to_dict(orient="records")}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing API data: {str(e)}")

# ------------------------ Run Server ------------------------

@app.get("/")
async def root():
    """Root endpoint - API health check"""
    endpoints = {
        "v1": {
            "clean_data": "/clean-data",
            "clean_db": "/clean-db",
            "clean_api": "/clean-api"
        }
    }
    
    if AUTOML_AVAILABLE:
        endpoints["v2"] = {
            "upload_data": "/api/v2/upload-data",
            "clean_data": "/api/v2/clean-data",
            "suggest_targets": "/api/v2/suggest-targets",
            "select_target": "/api/v2/select-target",
            "preprocess": "/api/v2/preprocess",
            "train_model": "/api/v2/train-model",
            "compare_models": "/api/v2/compare-models",
            "tune_model": "/api/v2/tune-model",
            "health": "/api/v2/health"
        }
        endpoints["mlops"] = {
            "save_model": "/api/v2/mlops/save-model",
            "list_versions": "/api/v2/mlops/versions/{project_id}",
            "export_model": "/api/v2/mlops/export-model",
            "rollback": "/api/v2/mlops/rollback",
            "retrain": "/api/v2/mlops/retrain",
            "predict": "/api/v2/mlops/predict"
        }
    
    return {
        "service": "AI Data Cleaning + AutoML Platform",
        "status": "healthy",
        "version": "2.0.0",
        "powered_by": "Euri API (27+ AI Models)",
        "automl_available": AUTOML_AVAILABLE,
        "endpoints": endpoints
    }

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("FASTAPI_HOST", "127.0.0.1")
    port = int(os.getenv("FASTAPI_PORT", 8000))
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║  🚀 AI Data Cleaning + AutoML Platform - FastAPI              ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  API URL:      http://{host}:{port}                        ║
    ║  Docs:         http://{host}:{port}/docs                   ║
    ║  ReDoc:        http://{host}:{port}/redoc                  ║
    ║  AI Engine:    Euri API (27+ Models)                          ║
    ║  AutoML:       {'Enabled' if AUTOML_AVAILABLE else 'Disabled'}                                     ║
    ║  Status:       Starting...                                     ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run("backend:app", host=host, port=port, reload=True, app_dir="scripts")

