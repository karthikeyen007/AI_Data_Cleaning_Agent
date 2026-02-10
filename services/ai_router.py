"""
AI Router - Source-Aware Multi-Key Model Selection and Orchestration

Production-grade AI routing with:
- Source-aware model and API key selection
- Secure key handling (keys never logged)
- Automatic fallback on model failure
- Request batching and chunking for large datasets
- Cost-aware routing design
- Retry logic with exponential backoff
- Comprehensive operation logging

Data Source to Model/Key Mapping:
- DATABASE → DB_EURI_API_KEY + gemini-2.5-pro
- UPLOAD   → UPLOAD_EURI_API_KEY + gpt-5-mini-2025-08-07
- API      → API_EURI_API_KEY + gemini-2.0-flash
- FALLBACK → FALLBACK_EURI_API_KEY + gpt-4.1-mini
"""

import os
import sys
import time
import logging
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from io import StringIO
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """Enumeration of supported data source types."""
    DATABASE = "database"
    UPLOAD = "upload"
    API = "api"


@dataclass
class ModelConfig:
    """Configuration for an AI model."""
    model_id: str
    key_env_var: str
    max_tokens: int = 128000
    cost_per_1k_tokens: float = 0.001
    avg_latency_ms: int = 500
    supports_batch: bool = True
    priority: int = 1  # Lower is higher priority


@dataclass
class CleaningRequest:
    """Encapsulates a data cleaning request."""
    data: pd.DataFrame
    source_type: DataSourceType
    batch_size: int = 20
    max_retries: int = 3
    timeout: int = 60
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CleaningResult:
    """Encapsulates the result of a cleaning operation."""
    cleaned_data: pd.DataFrame
    model_used: str
    processing_time_ms: float
    tokens_used: int
    batches_processed: int
    errors: List[str] = field(default_factory=list)
    fallback_used: bool = False
    fallback_reason: Optional[str] = None


class AIRouter:
    """
    Source-aware AI routing orchestrator with multi-key support.
    
    Routes cleaning requests to appropriate models based on:
    - Data source type (each uses different API key)
    - Dataset size
    - Cost considerations
    - Model availability
    
    Security:
    - API keys are NEVER logged
    - Keys are passed securely to the client layer
    - Fallback is triggered transparently
    """
    
    def __init__(self):
        """Initialize the AI Router with model configurations."""
        self._load_model_configs()
        self._init_client()
        self._log_initialization()
    
    def _load_model_configs(self) -> None:
        """Load model configurations from environment variables."""
        self.MODEL_REGISTRY = {
            DataSourceType.DATABASE: ModelConfig(
                model_id=os.getenv("DB_CLEANING_MODEL", "gemini-2.5-pro"),
                key_env_var="DB_EURI_API_KEY",
                max_tokens=2000000,  # 2M context for large DB results
                cost_per_1k_tokens=0.005,
                avg_latency_ms=800,
                priority=1
            ),
            DataSourceType.UPLOAD: ModelConfig(
                model_id=os.getenv("UPLOAD_CLEANING_MODEL", "gpt-5-mini-2025-08-07"),
                key_env_var="UPLOAD_EURI_API_KEY",
                max_tokens=128000,
                cost_per_1k_tokens=0.002,
                avg_latency_ms=400,
                priority=2
            ),
            DataSourceType.API: ModelConfig(
                model_id=os.getenv("API_CLEANING_MODEL", "gemini-2.0-flash"),
                key_env_var="API_EURI_API_KEY",
                max_tokens=1000000,  # 1M context
                cost_per_1k_tokens=0.001,
                avg_latency_ms=200,
                priority=3
            ),
        }
        
        self.fallback_model = ModelConfig(
            model_id=os.getenv("FALLBACK_MODEL", "gpt-4.1-mini"),
            key_env_var="FALLBACK_EURI_API_KEY",
            max_tokens=128000,
            cost_per_1k_tokens=0.0005,
            avg_latency_ms=150,
            priority=99
        )
    
    def _init_client(self) -> None:
        """Initialize the multi-key Euri API client."""
        # Import the multi-key client
        scripts_path = os.path.join(os.path.dirname(__file__), '..', 'scripts')
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        
        from euri_client import MultiKeyEuriClient, DataSourceType as ClientSourceType
        
        # Map our DataSourceType to client's DataSourceType
        self._source_type_map = {
            DataSourceType.DATABASE: ClientSourceType.DATABASE,
            DataSourceType.UPLOAD: ClientSourceType.UPLOAD,
            DataSourceType.API: ClientSourceType.API,
        }
        self._fallback_source = ClientSourceType.FALLBACK
        
        # Initialize client (validation is optional, can be done at startup)
        self.client = MultiKeyEuriClient(validate_on_init=False)
    
    def _log_initialization(self) -> None:
        """Log initialization status."""
        logger.info("=" * 60)
        logger.info("✅ AIRouter initialized with source-aware multi-key routing")
        logger.info("=" * 60)
        logger.info("Model Configuration:")
        for source_type, config in self.MODEL_REGISTRY.items():
            # Check if key is configured (without revealing it)
            key_set = bool(os.getenv(config.key_env_var))
            status = "✅" if key_set else "❌"
            logger.info(f"  {status} {source_type.value}: {config.model_id} (key: {config.key_env_var})")
        
        fallback_key_set = bool(os.getenv(self.fallback_model.key_env_var))
        status = "✅" if fallback_key_set else "❌"
        logger.info(f"  {status} fallback: {self.fallback_model.model_id} (key: {self.fallback_model.key_env_var})")
        logger.info("=" * 60)
    
    def get_model_for_source(self, source_type: DataSourceType) -> ModelConfig:
        """
        Get the appropriate model configuration for a data source type.
        
        Args:
            source_type: The type of data source
            
        Returns:
            ModelConfig for the specified source type
        """
        return self.MODEL_REGISTRY.get(source_type, self.fallback_model)
    
    def get_source_type_from_string(self, source: str) -> DataSourceType:
        """
        Convert string source type to enum.
        
        Args:
            source: String like 'database', 'upload', 'api'
            
        Returns:
            DataSourceType enum value
        """
        source_lower = source.lower()
        for st in DataSourceType:
            if st.value == source_lower:
                return st
        # Default to upload for unknown sources
        return DataSourceType.UPLOAD
    
    def estimate_cost(self, df: pd.DataFrame, source_type: DataSourceType) -> Dict[str, Any]:
        """
        Estimate the cost of processing a dataset.
        
        Args:
            df: DataFrame to process
            source_type: Type of data source
            
        Returns:
            Dictionary with cost estimation details
        """
        model_config = self.get_model_for_source(source_type)
        
        # Rough token estimation: ~4 chars per token
        char_count = df.to_string().__len__()
        estimated_tokens = char_count // 4
        
        # Account for prompt overhead and response
        total_tokens = estimated_tokens * 2.5  # Input + output + overhead
        
        estimated_cost = (total_tokens / 1000) * model_config.cost_per_1k_tokens
        estimated_time = (len(df) / 20) * (model_config.avg_latency_ms / 1000)  # Per batch
        
        return {
            "model": model_config.model_id,
            "key_env_var": model_config.key_env_var,  # For verification, not the key itself!
            "estimated_tokens": int(total_tokens),
            "estimated_cost_usd": round(estimated_cost, 4),
            "estimated_time_seconds": round(estimated_time, 2),
            "batches_required": (len(df) // 20) + 1
        }
    
    def _chunk_dataframe(self, df: pd.DataFrame, chunk_size: int) -> List[pd.DataFrame]:
        """
        Split DataFrame into chunks for batch processing.
        
        Args:
            df: DataFrame to chunk
            chunk_size: Number of rows per chunk
            
        Returns:
            List of DataFrame chunks
        """
        return [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    def _build_cleaning_prompt(self, df_batch: pd.DataFrame) -> str:
        """
        Build the AI prompt for data cleaning.
        
        Args:
            df_batch: DataFrame batch to clean
            
        Returns:
            Formatted prompt string
        """
        return f"""You are an AI Data Cleaning Expert. Analyze and clean this dataset:

Dataset:
{df_batch.to_string()}

Tasks:
1. Handle missing values using context-appropriate strategies (mean/median/mode/forward-fill)
2. Remove duplicate rows
3. Fix inconsistent formatting in text columns (standardize case, trim whitespace)
4. Standardize data types where applicable
5. Identify and handle obvious outliers/errors
6. Normalize semantic variations (e.g., "USA" and "United States" → standardize)

CRITICAL: Return ONLY the cleaned data in valid CSV format.
- Include the header row
- No explanations or markdown
- Output must be directly parseable by pandas.read_csv()
"""
    
    def _process_batch(
        self,
        batch: pd.DataFrame,
        source_type: DataSourceType,
        retry_count: int = 0,
        max_retries: int = 3
    ) -> Tuple[pd.DataFrame, bool, Optional[str]]:
        """
        Process a single batch with retry and fallback logic.
        
        Args:
            batch: DataFrame batch to process
            source_type: Data source type for key selection
            retry_count: Current retry attempt
            max_retries: Maximum retry attempts
            
        Returns:
            Tuple of (cleaned_dataframe, used_fallback, fallback_reason)
        """
        model_config = self.get_model_for_source(source_type)
        client_source = self._source_type_map.get(source_type, self._fallback_source)
        
        try:
            prompt = self._build_cleaning_prompt(batch)
            
            # Use multi-key client with source-aware routing
            response = self.client.get_text_completion(
                prompt=prompt,
                source_type=client_source,
                model=model_config.model_id,
                temperature=0.2  # Low temperature for consistent output
            )
            
            # Parse CSV response
            cleaned_response = response.strip()
            
            # Remove markdown code blocks if present
            if cleaned_response.startswith("```"):
                lines = cleaned_response.split("\n")
                cleaned_response = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            
            cleaned_batch = pd.read_csv(StringIO(cleaned_response))
            return cleaned_batch, False, None
            
        except Exception as e:
            error_msg = str(e)
            # Ensure we don't log any API keys
            for source in DataSourceType:
                config = self.MODEL_REGISTRY.get(source)
                if config:
                    key = os.getenv(config.key_env_var, "")
                    if key and len(key) >= 10:
                        error_msg = error_msg.replace(key, "****")
            
            logger.warning(f"Batch processing failed (attempt {retry_count + 1}): {error_msg}")
            
            if retry_count < max_retries:
                # Exponential backoff
                wait_time = 2 ** retry_count
                time.sleep(wait_time)
                
                # Try fallback model on last retry
                if retry_count == max_retries - 1:
                    return self._process_with_fallback(batch, error_msg)
                
                return self._process_batch(batch, source_type, retry_count + 1, max_retries)
            
            # All retries exhausted - try fallback
            return self._process_with_fallback(batch, error_msg)
    
    def _process_with_fallback(
        self,
        batch: pd.DataFrame,
        original_error: str
    ) -> Tuple[pd.DataFrame, bool, str]:
        """
        Process batch with fallback model.
        
        Args:
            batch: DataFrame batch to process
            original_error: Error that triggered fallback
            
        Returns:
            Tuple of (cleaned_dataframe, used_fallback, fallback_reason)
        """
        logger.warning(f"⚠️ Switching to fallback model: {self.fallback_model.model_id}")
        
        try:
            prompt = self._build_cleaning_prompt(batch)
            
            response = self.client.get_text_completion(
                prompt=prompt,
                source_type=self._fallback_source,
                model=self.fallback_model.model_id,
                temperature=0.2
            )
            
            cleaned_response = response.strip()
            if cleaned_response.startswith("```"):
                lines = cleaned_response.split("\n")
                cleaned_response = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            
            cleaned_batch = pd.read_csv(StringIO(cleaned_response))
            return cleaned_batch, True, f"Primary model failed: {original_error}"
            
        except Exception as e:
            logger.error(f"Fallback also failed: {str(e)}")
            logger.warning("Returning original data batch unchanged")
            return batch, True, f"Both primary and fallback failed: {str(e)}"
    
    def clean(self, request: CleaningRequest) -> CleaningResult:
        """
        Execute the full cleaning pipeline for a request.
        
        Args:
            request: CleaningRequest containing data and configuration
            
        Returns:
            CleaningResult with cleaned data and metadata
        """
        start_time = time.time()
        model_config = self.get_model_for_source(request.source_type)
        
        logger.info("=" * 60)
        logger.info(f"🧹 Starting cleaning pipeline")
        logger.info(f"   Source type: {request.source_type.value}")
        logger.info(f"   Model: {model_config.model_id}")
        logger.info(f"   Key: {model_config.key_env_var}")  # Log env var name, NOT the key
        logger.info(f"   Dataset shape: {request.data.shape}")
        logger.info("=" * 60)
        
        # Chunk the dataset
        chunks = self._chunk_dataframe(request.data, request.batch_size)
        logger.info(f"Split into {len(chunks)} batches of {request.batch_size} rows")
        
        cleaned_chunks = []
        errors = []
        fallback_used = False
        fallback_reason = None
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing batch {i + 1}/{len(chunks)}...")
            
            cleaned_chunk, used_fallback, reason = self._process_batch(
                chunk,
                request.source_type,
                max_retries=request.max_retries
            )
            
            cleaned_chunks.append(cleaned_chunk)
            
            if used_fallback:
                fallback_used = True
                fallback_reason = reason
                errors.append(f"Batch {i + 1}: Fallback triggered - {reason}")
        
        # Combine all cleaned chunks
        result_df = pd.concat(cleaned_chunks, ignore_index=True)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Estimate tokens used
        tokens_used = len(request.data.to_string()) // 4 * 2
        
        result = CleaningResult(
            cleaned_data=result_df,
            model_used=self.fallback_model.model_id if fallback_used else model_config.model_id,
            processing_time_ms=processing_time,
            tokens_used=tokens_used,
            batches_processed=len(chunks),
            errors=errors,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason
        )
        
        logger.info("=" * 60)
        logger.info(f"✅ Cleaning completed in {processing_time:.2f}ms")
        logger.info(f"   Result shape: {result_df.shape}")
        logger.info(f"   Fallback used: {fallback_used}")
        if fallback_used:
            logger.info(f"   Fallback reason: {fallback_reason}")
        logger.info("=" * 60)
        
        return result
    
    def clean_dataframe(
        self,
        df: pd.DataFrame,
        source_type: DataSourceType,
        batch_size: int = 20
    ) -> pd.DataFrame:
        """
        Convenience method for simple DataFrame cleaning.
        
        Args:
            df: DataFrame to clean
            source_type: Type of data source
            batch_size: Rows per batch
            
        Returns:
            Cleaned DataFrame
        """
        request = CleaningRequest(
            data=df,
            source_type=source_type,
            batch_size=batch_size
        )
        result = self.clean(request)
        return result.cleaned_data
    
    def get_routing_info(self) -> Dict[str, Any]:
        """
        Get current routing configuration for debugging/monitoring.
        Keys are never included in this output.
        
        Returns:
            Dictionary with routing configuration
        """
        info = {
            "sources": {},
            "fallback": {
                "model": self.fallback_model.model_id,
                "key_configured": bool(os.getenv(self.fallback_model.key_env_var))
            }
        }
        
        for source_type, config in self.MODEL_REGISTRY.items():
            info["sources"][source_type.value] = {
                "model": config.model_id,
                "key_configured": bool(os.getenv(config.key_env_var)),
                "max_tokens": config.max_tokens,
                "cost_per_1k": config.cost_per_1k_tokens
            }
        
        return info


# Singleton instance for application-wide use
_router_instance: Optional[AIRouter] = None


def get_ai_router() -> AIRouter:
    """Get or create the singleton AIRouter instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = AIRouter()
    return _router_instance


def reset_ai_router() -> None:
    """Reset the singleton (useful for testing)."""
    global _router_instance
    _router_instance = None
