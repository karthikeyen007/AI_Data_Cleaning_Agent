"""
Tests for Multi-Key API Routing

Tests verify:
- DB cleaning uses DB key
- Upload cleaning uses upload key
- API cleaning uses API key
- Fallback triggers when model fails
- Keys are never exposed in logs or errors
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))


class TestKeyMasking(unittest.TestCase):
    """Test API key masking utility."""
    
    def test_mask_key(self):
        """Test key masking."""
        from scripts.euri_client import APIKeyMasker
        
        key = "euri-1234567890abcdef1234567890abcdef"
        masked = APIKeyMasker.mask(key)
        
        # Should show first 4 and last 4 chars
        self.assertTrue(masked.startswith("euri"))
        self.assertIn("****", masked)
        self.assertNotIn("1234567890abcdef", masked)
    
    def test_mask_short_key(self):
        """Test masking short keys."""
        from scripts.euri_client import APIKeyMasker
        
        masked = APIKeyMasker.mask("short")
        self.assertEqual(masked, "*****")
    
    def test_mask_none_key(self):
        """Test masking None key."""
        from scripts.euri_client import APIKeyMasker
        
        masked = APIKeyMasker.mask(None)
        self.assertEqual(masked, "<NOT SET>")
    
    def test_sanitize_log_message(self):
        """Test removing keys from log messages."""
        from scripts.euri_client import APIKeyMasker
        
        key = "euri-1234567890abcdef1234567890abcdef"
        message = f"Error connecting with key {key}"
        
        sanitized = APIKeyMasker.sanitize_log_message(message, [key])
        
        self.assertNotIn(key, sanitized)
        self.assertIn("****", sanitized)


class TestKeyValidator(unittest.TestCase):
    """Test API key validation."""
    
    @patch.dict(os.environ, {
        "DB_EURI_API_KEY": "test-db-key-1234567890",
        "DB_CLEANING_MODEL": "gemini-2.5-pro",
        "UPLOAD_EURI_API_KEY": "test-upload-key-1234567890",
        "UPLOAD_CLEANING_MODEL": "gpt-5-mini-2025-08-07",
        "API_EURI_API_KEY": "test-api-key-1234567890",
        "API_CLEANING_MODEL": "gemini-2.0-flash",
        "FALLBACK_EURI_API_KEY": "test-fallback-key-1234567890",
        "FALLBACK_MODEL": "gpt-4.1-mini",
        "DEFAULT_EURI_API_KEY": "test-default-key-1234567890",
        "DEFAULT_MODEL": "gpt-4.1-mini",
    }, clear=False)
    def test_validate_all_keys_present(self):
        """Test validation passes when all keys present."""
        from scripts.euri_client import KeyValidator
        
        is_valid, issues = KeyValidator.validate_all(strict=True)
        
        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)
    
    @patch.dict(os.environ, {
        "DB_EURI_API_KEY": "",  # Missing
        "DB_CLEANING_MODEL": "gemini-2.5-pro",
    }, clear=True)
    def test_validate_missing_key(self):
        """Test validation fails when key missing."""
        from scripts.euri_client import KeyValidator
        
        is_valid, issues = KeyValidator.validate_all(strict=True)
        
        self.assertFalse(is_valid)
        self.assertTrue(any("DB_EURI_API_KEY" in issue for issue in issues))
    
    @patch.dict(os.environ, {
        "DB_EURI_API_KEY": "short",  # Too short
        "DB_CLEANING_MODEL": "gemini-2.5-pro",
    }, clear=True)
    def test_validate_short_key(self):
        """Test validation fails for short keys."""
        from scripts.euri_client import KeyValidator
        
        is_valid, issues = KeyValidator.validate_all(strict=True)
        
        self.assertFalse(is_valid)
        self.assertTrue(any("too short" in issue for issue in issues))


class TestMultiKeyClient(unittest.TestCase):
    """Test multi-key client routing."""
    
    @patch.dict(os.environ, {
        "DB_EURI_API_KEY": "test-db-key-1234567890",
        "DB_CLEANING_MODEL": "gemini-2.5-pro",
        "UPLOAD_EURI_API_KEY": "test-upload-key-1234567890",
        "UPLOAD_CLEANING_MODEL": "gpt-5-mini-2025-08-07",
        "API_EURI_API_KEY": "test-api-key-1234567890",
        "API_CLEANING_MODEL": "gemini-2.0-flash",
        "FALLBACK_EURI_API_KEY": "test-fallback-key-1234567890",
        "FALLBACK_MODEL": "gpt-4.1-mini",
        "DEFAULT_EURI_API_KEY": "test-default-key-1234567890",
        "DEFAULT_MODEL": "gpt-4.1-mini",
    }, clear=False)
    def test_client_initialization(self):
        """Test client initializes with all keys."""
        from scripts.euri_client import MultiKeyEuriClient, DataSourceType
        
        client = MultiKeyEuriClient(validate_on_init=False)
        
        # Should have all source types available
        self.assertIn(DataSourceType.DATABASE, client._available_keys)
        self.assertIn(DataSourceType.UPLOAD, client._available_keys)
        self.assertIn(DataSourceType.API, client._available_keys)
    
    @patch.dict(os.environ, {
        "DB_EURI_API_KEY": "test-db-key-1234567890",
        "DB_CLEANING_MODEL": "gemini-2.5-pro",
        "UPLOAD_EURI_API_KEY": "test-upload-key-1234567890",
        "UPLOAD_CLEANING_MODEL": "gpt-5-mini-2025-08-07",
        "API_EURI_API_KEY": "test-api-key-1234567890",
        "API_CLEANING_MODEL": "gemini-2.0-flash",
        "FALLBACK_EURI_API_KEY": "test-fallback-key-1234567890",
        "FALLBACK_MODEL": "gpt-4.1-mini",
        "DEFAULT_EURI_API_KEY": "test-default-key-1234567890",
        "DEFAULT_MODEL": "gpt-4.1-mini",
    }, clear=False)
    def test_correct_key_for_source(self):
        """Test correct key is selected for each source."""
        from scripts.euri_client import MultiKeyEuriClient, DataSourceType
        
        client = MultiKeyEuriClient(validate_on_init=False)
        
        db_config = client.get_config_for_source(DataSourceType.DATABASE)
        self.assertEqual(db_config.key, "test-db-key-1234567890")
        self.assertEqual(db_config.model, "gemini-2.5-pro")
        
        upload_config = client.get_config_for_source(DataSourceType.UPLOAD)
        self.assertEqual(upload_config.key, "test-upload-key-1234567890")
        self.assertEqual(upload_config.model, "gpt-5-mini-2025-08-07")
        
        api_config = client.get_config_for_source(DataSourceType.API)
        self.assertEqual(api_config.key, "test-api-key-1234567890")
        self.assertEqual(api_config.model, "gemini-2.0-flash")
    
    @patch.dict(os.environ, {
        "DB_EURI_API_KEY": "test-db-key-1234567890",
        "DB_CLEANING_MODEL": "gemini-2.5-pro",
        "FALLBACK_EURI_API_KEY": "test-fallback-key-1234567890",
        "FALLBACK_MODEL": "gpt-4.1-mini",
        "DEFAULT_EURI_API_KEY": "test-default-key-1234567890",
        "DEFAULT_MODEL": "gpt-4.1-mini",
    }, clear=False)
    def test_key_status_masks_keys(self):
        """Test get_key_status never reveals actual keys."""
        from scripts.euri_client import MultiKeyEuriClient
        
        client = MultiKeyEuriClient(validate_on_init=False)
        status = client.get_key_status()
        
        for source, info in status.items():
            # Key preview should be masked, not the actual key
            if info["configured"]:
                self.assertIn("****", info["key_preview"])
                self.assertNotIn("test-db-key", info["key_preview"])


class TestAIRouter(unittest.TestCase):
    """Test AI Router with multi-key routing."""
    
    @patch.dict(os.environ, {
        "DB_EURI_API_KEY": "test-db-key-1234567890",
        "DB_CLEANING_MODEL": "gemini-2.5-pro",
        "UPLOAD_EURI_API_KEY": "test-upload-key-1234567890",
        "UPLOAD_CLEANING_MODEL": "gpt-5-mini-2025-08-07",
        "API_EURI_API_KEY": "test-api-key-1234567890",
        "API_CLEANING_MODEL": "gemini-2.0-flash",
        "FALLBACK_EURI_API_KEY": "test-fallback-key-1234567890",
        "FALLBACK_MODEL": "gpt-4.1-mini",
        "DEFAULT_EURI_API_KEY": "test-default-key-1234567890",
        "DEFAULT_MODEL": "gpt-4.1-mini",
    }, clear=False)
    def test_router_initialization(self):
        """Test router initializes with model configs."""
        from services.ai_router import AIRouter, DataSourceType, reset_ai_router
        
        reset_ai_router()  # Clear singleton
        router = AIRouter()
        
        # Check model registry
        self.assertIn(DataSourceType.DATABASE, router.MODEL_REGISTRY)
        self.assertIn(DataSourceType.UPLOAD, router.MODEL_REGISTRY)
        self.assertIn(DataSourceType.API, router.MODEL_REGISTRY)
    
    @patch.dict(os.environ, {
        "DB_EURI_API_KEY": "test-db-key-1234567890",
        "DB_CLEANING_MODEL": "gemini-2.5-pro",
        "UPLOAD_EURI_API_KEY": "test-upload-key-1234567890",
        "UPLOAD_CLEANING_MODEL": "gpt-5-mini-2025-08-07",
        "API_EURI_API_KEY": "test-api-key-1234567890",
        "API_CLEANING_MODEL": "gemini-2.0-flash",
        "FALLBACK_EURI_API_KEY": "test-fallback-key-1234567890",
        "FALLBACK_MODEL": "gpt-4.1-mini",
        "DEFAULT_EURI_API_KEY": "test-default-key-1234567890",
        "DEFAULT_MODEL": "gpt-4.1-mini",
    }, clear=False)
    def test_model_selection_for_source(self):
        """Test correct model is selected for each source."""
        from services.ai_router import AIRouter, DataSourceType, reset_ai_router
        
        reset_ai_router()
        router = AIRouter()
        
        db_config = router.get_model_for_source(DataSourceType.DATABASE)
        self.assertEqual(db_config.model_id, "gemini-2.5-pro")
        self.assertEqual(db_config.key_env_var, "DB_EURI_API_KEY")
        
        upload_config = router.get_model_for_source(DataSourceType.UPLOAD)
        self.assertEqual(upload_config.model_id, "gpt-5-mini-2025-08-07")
        self.assertEqual(upload_config.key_env_var, "UPLOAD_EURI_API_KEY")
        
        api_config = router.get_model_for_source(DataSourceType.API)
        self.assertEqual(api_config.model_id, "gemini-2.0-flash")
        self.assertEqual(api_config.key_env_var, "API_EURI_API_KEY")
    
    @patch.dict(os.environ, {
        "DB_EURI_API_KEY": "test-db-key-1234567890",
        "DB_CLEANING_MODEL": "gemini-2.5-pro",
        "FALLBACK_EURI_API_KEY": "test-fallback-key-1234567890",
        "FALLBACK_MODEL": "gpt-4.1-mini",
        "DEFAULT_EURI_API_KEY": "test-default-key-1234567890",
        "DEFAULT_MODEL": "gpt-4.1-mini",
    }, clear=False)
    def test_routing_info_no_keys_exposed(self):
        """Test routing info never exposes actual keys."""
        from services.ai_router import AIRouter, reset_ai_router
        
        reset_ai_router()
        router = AIRouter()
        info = router.get_routing_info()
        
        # Convert to string and check for key presence
        info_str = str(info)
        self.assertNotIn("test-db-key", info_str)
        self.assertNotIn("test-fallback-key", info_str)


class TestFallbackBehavior(unittest.TestCase):
    """Test fallback trigger behavior."""
    
    @patch.dict(os.environ, {
        "DB_EURI_API_KEY": "test-db-key-1234567890",
        "DB_CLEANING_MODEL": "gemini-2.5-pro",
        "FALLBACK_EURI_API_KEY": "test-fallback-key-1234567890",
        "FALLBACK_MODEL": "gpt-4.1-mini",
        "DEFAULT_EURI_API_KEY": "test-default-key-1234567890",
        "DEFAULT_MODEL": "gpt-4.1-mini",
    }, clear=False)
    @patch('requests.post')
    def test_fallback_triggered_on_failure(self, mock_post):
        """Test fallback is used when primary model fails."""
        from scripts.euri_client import MultiKeyEuriClient, DataSourceType
        
        # Configure mock to fail on first call, succeed on fallback
        error_response = MagicMock()
        error_response.raise_for_status.side_effect = Exception("Model unavailable")
        
        success_response = MagicMock()
        success_response.raise_for_status.return_value = None
        success_response.json.return_value = {
            "choices": [{"message": {"content": "Cleaned data"}}]
        }
        
        # First 4 calls fail (3 retries + 1), then fallback succeeds
        mock_post.side_effect = [
            Exception("Fail 1"),
            Exception("Fail 2"),
            Exception("Fail 3"),
            success_response
        ]
        
        client = MultiKeyEuriClient(validate_on_init=False)
        
        try:
            result = client.chat_completion(
                messages=[{"role": "user", "content": "test"}],
                source_type=DataSourceType.DATABASE,
                use_fallback_on_error=True
            )
            
            # Should have used fallback
            self.assertIn("_fallback_used", result)
            self.assertTrue(result["_fallback_used"])
        except Exception:
            # Fallback count should have incremented
            self.assertGreater(client._fallback_count, 0)


class TestSecurityCompliance(unittest.TestCase):
    """Test security requirements are met."""
    
    def test_no_keys_in_logs(self):
        """Verify no API keys appear in log output."""
        import logging
        import io
        
        # Capture log output
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.DEBUG)
        
        logger = logging.getLogger("scripts.euri_client")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        try:
            os.environ["DB_EURI_API_KEY"] = "secret-key-should-never-appear"
            
            from scripts.euri_client import MultiKeyEuriClient
            client = MultiKeyEuriClient(validate_on_init=False)
            
            log_output = log_stream.getvalue()
            
            # The actual key should never appear in logs
            self.assertNotIn("secret-key-should-never-appear", log_output)
            
        finally:
            logger.removeHandler(handler)
            if "DB_EURI_API_KEY" in os.environ:
                del os.environ["DB_EURI_API_KEY"]
    
    def test_no_keys_in_error_messages(self):
        """Verify error messages don't contain keys."""
        from scripts.euri_client import APIKeyMasker
        
        key = "super-secret-api-key-12345"
        error_msg = f"Failed to connect using {key}"
        
        sanitized = APIKeyMasker.sanitize_log_message(error_msg, [key])
        
        self.assertNotIn(key, sanitized)


if __name__ == "__main__":
    unittest.main()
