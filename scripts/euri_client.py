"""
Euri API Client - Multi-Key Production Implementation

Production-grade client for Euron.one API with:
- Multi-key authentication for source-aware routing
- Secure key management (no logging of keys)
- Automatic fallback on failure
- Retry logic with exponential backoff
- Key validation at startup
"""

import os
import re
import time
import logging
import requests
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """Data source types for key routing."""
    DATABASE = "database"
    UPLOAD = "upload"
    API = "api"
    FALLBACK = "fallback"
    DEFAULT = "default"


@dataclass
class KeyConfig:
    """Configuration for an API key."""
    env_var: str
    key: Optional[str]
    model: str
    description: str
    is_required: bool = True


class APIKeyMasker:
    """Utility for securely masking API keys in logs and output."""
    
    @staticmethod
    def mask(key: Optional[str], visible_chars: int = 4) -> str:
        """
        Mask an API key for safe display.
        
        Args:
            key: The API key to mask
            visible_chars: Number of characters to show at start
            
        Returns:
            Masked key string (e.g., "euri-****...****")
        """
        if not key:
            return "<NOT SET>"
        if len(key) <= visible_chars * 2:
            return "*" * len(key)
        return f"{key[:visible_chars]}****...****{key[-visible_chars:]}"
    
    @staticmethod
    def is_key_in_string(text: str, key: str) -> bool:
        """Check if a key appears in a string (for log sanitization)."""
        if not key or len(key) < 10:
            return False
        return key in text
    
    @staticmethod
    def sanitize_log_message(message: str, keys: List[str]) -> str:
        """Remove API keys from log messages."""
        sanitized = message
        for key in keys:
            if key and len(key) >= 10:
                sanitized = sanitized.replace(key, APIKeyMasker.mask(key))
        return sanitized


class KeyValidator:
    """Validates API key configuration at startup."""
    
    # Required key configurations
    REQUIRED_KEYS = [
        ("DB_EURI_API_KEY", "DB_CLEANING_MODEL", "Database cleaning"),
        ("UPLOAD_EURI_API_KEY", "UPLOAD_CLEANING_MODEL", "Upload cleaning"),
        ("API_EURI_API_KEY", "API_CLEANING_MODEL", "API cleaning"),
        ("FALLBACK_EURI_API_KEY", "FALLBACK_MODEL", "Fallback cleaning"),
        ("DEFAULT_EURI_API_KEY", "DEFAULT_MODEL", "Default operations"),
    ]
    
    @classmethod
    def validate_all(cls, strict: bool = True) -> Tuple[bool, List[str]]:
        """
        Validate all required API keys are present.
        
        Args:
            strict: If True, fail on any missing key
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        for key_var, model_var, description in cls.REQUIRED_KEYS:
            key = os.getenv(key_var)
            model = os.getenv(model_var)
            
            if not key or key.strip() == "":
                issues.append(f"Missing API key: {key_var} (required for {description})")
            elif len(key) < 20:
                issues.append(f"Invalid API key format: {key_var} (too short)")
            
            if model and key:
                # Key exists, model configured - valid
                pass
            elif model and not key:
                issues.append(f"Model {model_var}={model} configured but {key_var} is missing")
        
        is_valid = len(issues) == 0 if strict else True
        
        return is_valid, issues
    
    @classmethod
    def validate_single(cls, key_var: str) -> Tuple[bool, str]:
        """Validate a single API key."""
        key = os.getenv(key_var)
        
        if not key or key.strip() == "":
            return False, f"Key not set: {key_var}"
        if len(key) < 20:
            return False, f"Key too short: {key_var}"
        
        return True, "Valid"


class MultiKeyEuriClient:
    """
    Production Euri API Client with multi-key support.
    
    Features:
    - Source-aware key routing
    - Automatic fallback on failure
    - Secure key handling (never logged)
    - Retry with exponential backoff
    """
    
    # Available Models
    MODELS = {
        # OpenAI Models
        "gpt-5-nano": "GPT 5 Nano - Fast and efficient",
        "gpt-5-mini-2025-08-07": "GPT 5 Mini - Balanced performance",
        "gpt-4.1-nano": "GPT 4.1 Nano - Cost-effective",
        "gpt-4.1-mini": "GPT 4.1 Mini - Balanced cost/performance",
        
        # Google Models
        "gemini-2.0-flash": "Gemini 2.0 Flash - Fast responses",
        "gemini-2.5-pro": "Gemini 2.5 Pro - Advanced reasoning",
        
        # Meta Models
        "llama-4-scout": "Llama 4 Scout - Open source power",
        "llama-4-maverick": "Llama 4 Maverick - Advanced capabilities",
        "llama-3.3-70b": "Llama 3.3 70B - Large model",
        
        # DeepSeek Models
        "deepseek-r1-distilled-70b": "DeepSeek R1 Distilled 70B - Reasoning model",
        
        # Alibaba Models
        "qwen-3-32b": "Qwen 3 32B - Multilingual support",
        
        # Groq Models
        "groq-compound": "Groq Compound - Ultra-fast inference",
        "groq-compound-mini": "Groq Compound Mini - Efficient processing"
    }
    
    def __init__(self, validate_on_init: bool = False):
        """
        Initialize multi-key Euri API client.
        
        Args:
            validate_on_init: If True, validate all keys on initialization
        """
        self.base_url = os.getenv("EURI_API_BASE_URL", "https://api.euron.one/api/v1/euri")
        
        # Load key configurations
        self._key_configs = self._load_key_configs()
        
        # Track which keys are available
        self._available_keys = set()
        for source_type, config in self._key_configs.items():
            if config.key:
                self._available_keys.add(source_type)
        
        # Track fallback usage for logging
        self._fallback_count = 0
        
        if validate_on_init:
            is_valid, issues = KeyValidator.validate_all(strict=True)
            if not is_valid:
                raise ValueError(f"API key validation failed:\n" + "\n".join(f"  - {i}" for i in issues))
        
        logger.info("✅ MultiKeyEuriClient initialized")
        self._log_key_status()
    
    def _load_key_configs(self) -> Dict[DataSourceType, KeyConfig]:
        """Load all key configurations from environment."""
        return {
            DataSourceType.DATABASE: KeyConfig(
                env_var="DB_EURI_API_KEY",
                key=os.getenv("DB_EURI_API_KEY"),
                model=os.getenv("DB_CLEANING_MODEL", "gemini-2.5-pro"),
                description="Database cleaning"
            ),
            DataSourceType.UPLOAD: KeyConfig(
                env_var="UPLOAD_EURI_API_KEY",
                key=os.getenv("UPLOAD_EURI_API_KEY"),
                model=os.getenv("UPLOAD_CLEANING_MODEL", "gpt-5-mini-2025-08-07"),
                description="Upload cleaning"
            ),
            DataSourceType.API: KeyConfig(
                env_var="API_EURI_API_KEY",
                key=os.getenv("API_EURI_API_KEY"),
                model=os.getenv("API_CLEANING_MODEL", "gemini-2.0-flash"),
                description="API cleaning"
            ),
            DataSourceType.FALLBACK: KeyConfig(
                env_var="FALLBACK_EURI_API_KEY",
                key=os.getenv("FALLBACK_EURI_API_KEY"),
                model=os.getenv("FALLBACK_MODEL", "gpt-4.1-mini"),
                description="Fallback"
            ),
            DataSourceType.DEFAULT: KeyConfig(
                env_var="DEFAULT_EURI_API_KEY",
                key=os.getenv("DEFAULT_EURI_API_KEY"),
                model=os.getenv("DEFAULT_MODEL", "gpt-4.1-mini"),
                description="Default"
            ),
        }
    
    def _log_key_status(self) -> None:
        """Log key configuration status (masked)."""
        logger.info("API Key Configuration:")
        for source_type, config in self._key_configs.items():
            masked = APIKeyMasker.mask(config.key)
            status = "✅" if config.key else "❌"
            logger.info(f"  {status} {source_type.value}: {masked} → {config.model}")
    
    def _get_headers(self, api_key: str) -> Dict[str, str]:
        """Get request headers with API key."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    
    def get_config_for_source(self, source_type: DataSourceType) -> KeyConfig:
        """
        Get the key configuration for a data source.
        
        Args:
            source_type: The data source type
            
        Returns:
            KeyConfig for that source
        """
        config = self._key_configs.get(source_type)
        if not config or not config.key:
            # Fall back to default
            return self._key_configs[DataSourceType.DEFAULT]
        return config
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        source_type: DataSourceType = DataSourceType.DEFAULT,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        use_fallback_on_error: bool = True
    ) -> Dict[str, Any]:
        """
        Send chat completion request with source-aware key routing.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            source_type: Data source type for key selection
            model: Override model (uses source-specific model if None)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            use_fallback_on_error: Whether to try fallback on failure
            
        Returns:
            API response dictionary
        """
        config = self.get_config_for_source(source_type)
        
        if not config.key:
            raise ValueError(f"No API key configured for source: {source_type.value}")
        
        use_model = model or config.model
        
        logger.info(f"Request: source={source_type.value}, model={use_model}")
        
        return self._execute_request(
            messages=messages,
            api_key=config.key,
            model=use_model,
            temperature=temperature,
            max_tokens=max_tokens,
            use_fallback_on_error=use_fallback_on_error
        )
    
    def _execute_request(
        self,
        messages: List[Dict[str, str]],
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        use_fallback_on_error: bool,
        retry_count: int = 0,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Execute the actual API request with retry logic."""
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "messages": messages,
            "model": model,
            "temperature": temperature
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        try:
            response = requests.post(
                url,
                headers=self._get_headers(api_key),
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            # Sanitize error message (never log API keys)
            all_keys = [c.key for c in self._key_configs.values() if c.key]
            error_msg = APIKeyMasker.sanitize_log_message(str(e), all_keys)
            
            logger.warning(f"Request failed (attempt {retry_count + 1}): {error_msg}")
            
            if retry_count < max_retries:
                # Exponential backoff
                wait_time = 2 ** retry_count
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
                
                # On last retry, try fallback if enabled
                if retry_count == max_retries - 1 and use_fallback_on_error:
                    return self._try_fallback(messages, temperature, max_tokens)
                
                return self._execute_request(
                    messages, api_key, model, temperature, max_tokens,
                    use_fallback_on_error, retry_count + 1, max_retries
                )
            
            # All retries exhausted
            if use_fallback_on_error:
                return self._try_fallback(messages, temperature, max_tokens)
            
            raise Exception(f"API request failed after {max_retries} retries: {error_msg}")
    
    def _try_fallback(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int]
    ) -> Dict[str, Any]:
        """Attempt request with fallback model/key."""
        fallback_config = self._key_configs[DataSourceType.FALLBACK]
        
        if not fallback_config.key:
            raise ValueError("Fallback API key not configured")
        
        logger.warning(f"⚠️ Switching to fallback: {fallback_config.model}")
        self._fallback_count += 1
        
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "messages": messages,
            "model": fallback_config.model,
            "temperature": temperature
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        try:
            response = requests.post(
                url,
                headers=self._get_headers(fallback_config.key),
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            result["_fallback_used"] = True
            result["_fallback_model"] = fallback_config.model
            return result
            
        except requests.exceptions.RequestException as e:
            all_keys = [c.key for c in self._key_configs.values() if c.key]
            error_msg = APIKeyMasker.sanitize_log_message(str(e), all_keys)
            raise Exception(f"Fallback request also failed: {error_msg}")
    
    def get_text_completion(
        self,
        prompt: str,
        source_type: DataSourceType = DataSourceType.DEFAULT,
        model: Optional[str] = None,
        temperature: float = 0.7
    ) -> str:
        """
        Get text completion for a single prompt.
        
        Args:
            prompt: Text prompt to send
            source_type: Data source type for key selection
            model: Override model
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        messages = [{"role": "user", "content": prompt}]
        response = self.chat_completion(
            messages, source_type, model, temperature
        )
        
        if "choices" in response and len(response["choices"]) > 0:
            return response["choices"][0]["message"]["content"]
        else:
            raise Exception("Invalid response format from Euri API")
    
    def validate_connection(self, source_type: DataSourceType = DataSourceType.DEFAULT) -> bool:
        """
        Test API connection for a specific source type.
        
        Args:
            source_type: Source type to test
            
        Returns:
            True if connection successful
        """
        try:
            test_message = [{"role": "user", "content": "Hello"}]
            response = self.chat_completion(
                test_message,
                source_type=source_type,
                temperature=0,
                use_fallback_on_error=False
            )
            return "choices" in response
        except Exception as e:
            logger.error(f"Connection validation failed for {source_type.value}: {e}")
            return False
    
    def validate_all_connections(self) -> Dict[str, bool]:
        """Test connections for all configured sources."""
        results = {}
        for source_type in DataSourceType:
            config = self._key_configs.get(source_type)
            if config and config.key:
                results[source_type.value] = self.validate_connection(source_type)
            else:
                results[source_type.value] = False
        return results
    
    def list_available_models(self) -> Dict[str, str]:
        """Get dictionary of available models."""
        return self.MODELS
    
    def get_key_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all API keys (for admin/monitoring).
        
        Returns:
            Dictionary with key status (keys are always masked)
        """
        status = {}
        for source_type, config in self._key_configs.items():
            status[source_type.value] = {
                "configured": config.key is not None,
                "key_preview": APIKeyMasker.mask(config.key),
                "model": config.model,
                "description": config.description
            }
        return status
    
    def get_fallback_count(self) -> int:
        """Get count of fallback triggers since initialization."""
        return self._fallback_count


# Convenience function for backward compatibility
def get_euri_client(validate: bool = False) -> MultiKeyEuriClient:
    """
    Get a configured multi-key Euri API client.
    
    Args:
        validate: Whether to validate all keys on init
        
    Returns:
        MultiKeyEuriClient instance
    """
    return MultiKeyEuriClient(validate_on_init=validate)


# Legacy alias for backward compatibility
EuriAPIClient = MultiKeyEuriClient


# Startup validation function
def validate_api_keys_or_fail():
    """
    Validate all required API keys are present.
    Raises SystemExit if validation fails.
    
    Call this at application startup.
    """
    is_valid, issues = KeyValidator.validate_all(strict=True)
    
    if not is_valid:
        print("\n" + "=" * 60)
        print("❌ API KEY VALIDATION FAILED")
        print("=" * 60)
        for issue in issues:
            print(f"  • {issue}")
        print("\nPlease set the required API keys in your .env file.")
        print("=" * 60 + "\n")
        raise SystemExit(1)
    
    print("✅ All API keys validated successfully")


# Test function
if __name__ == "__main__":
    print("🔍 Testing Multi-Key Euri API Client...")
    print("=" * 60)
    
    # Check key status (without validation)
    client = get_euri_client(validate=False)
    
    print("\n📋 Key Status:")
    for source, status in client.get_key_status().items():
        indicator = "✅" if status["configured"] else "❌"
        print(f"  {indicator} {source}: {status['key_preview']} → {status['model']}")
    
    print("\n⚠️ Note: Configure API keys in .env before running real requests.")
