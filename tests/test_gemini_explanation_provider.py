"""Minimal unit tests for Gemini explanation provider.

Tests cover basic functionality with mocks to avoid real API calls.
"""

import unittest
from unittest.mock import Mock, patch
import os
import sys


class TestGeminiExplanationProvider(unittest.TestCase):
    """Test Gemini explanation provider functionality."""

    def test_init_requires_api_key(self):
        """Should raise ValueError if GEMINI_API_KEY not set."""
        import types
        mock_package = types.ModuleType('google')
        mock_module = types.ModuleType('google.generativeai')
        # attach submodule to package for import system
        mock_package.generativeai = mock_module
        sys.modules['google'] = mock_package
        sys.modules['google.generativeai'] = mock_module
        try:
            with patch.dict(os.environ, {}, clear=True):
                from src.gemini_explanation_provider import GeminiExplanationProvider
                with self.assertRaises(ValueError):
                    GeminiExplanationProvider()
        finally:
            del sys.modules['google.generativeai']

    def test_init_with_api_key(self):
        """Should initialize successfully with API key."""
        import types
        mock_package = types.ModuleType('google')
        mock_module = types.ModuleType('google.generativeai')
        mock_module.configure = Mock()
        mock_module.GenerativeModel = Mock()
        mock_package.generativeai = mock_module
        sys.modules['google'] = mock_package
        sys.modules['google.generativeai'] = mock_module
        try:
            with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
                from src.gemini_explanation_provider import GeminiExplanationProvider
                provider = GeminiExplanationProvider()
                self.assertIsNotNone(provider)
        finally:
            del sys.modules['google.generativeai']
            del sys.modules['google']

    def test_generate_invalid_mode(self):
        """Should raise ValueError for invalid mode."""
        import types
        mock_package = types.ModuleType('google')
        mock_module = types.ModuleType('google.generativeai')
        mock_module.configure = Mock()
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "response"
        mock_model.generate_content.return_value = mock_response
        mock_module.GenerativeModel = lambda name=None: mock_model
        mock_package.generativeai = mock_module
        sys.modules['google'] = mock_package
        sys.modules['google.generativeai'] = mock_module
        try:
            with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
                from src.gemini_explanation_provider import GeminiExplanationProvider
                provider = GeminiExplanationProvider()
                with self.assertRaises(ValueError):
                    provider.generate("invalid", {})
        finally:
            del sys.modules['google.generativeai']
            del sys.modules['google']

    def test_generate_calls_api(self):
        """Should call Gemini API and return response."""
        import types
        mock_package = types.ModuleType('google')
        mock_module = types.ModuleType('google.generativeai')
        mock_module.configure = Mock()
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "AI generated response"
        mock_model.generate_content.return_value = mock_response
        mock_module.GenerativeModel = lambda name=None: mock_model
        mock_package.generativeai = mock_module
        sys.modules['google'] = mock_package
        sys.modules['google.generativeai'] = mock_module
        try:
            with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
                from src.gemini_explanation_provider import GeminiExplanationProvider
                provider = GeminiExplanationProvider()
                result = provider.generate("hint", {"user_message": "test"})
                self.assertEqual(result, "AI generated response")
                mock_model.generate_content.assert_called_once()
        finally:
            del sys.modules['google.generativeai']
            del sys.modules['google']

    def test_generate_handles_api_error(self):
        """Should raise RuntimeError on API failure."""
        import types
        mock_package = types.ModuleType('google')
        mock_module = types.ModuleType('google.generativeai')
        mock_module.configure = Mock()
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("API error")
        mock_module.GenerativeModel = lambda name=None: mock_model
        mock_package.generativeai = mock_module
        sys.modules['google'] = mock_package
        sys.modules['google.generativeai'] = mock_module
        try:
            with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
                from src.gemini_explanation_provider import GeminiExplanationProvider
                provider = GeminiExplanationProvider()
                with self.assertRaises(RuntimeError):
                    provider.generate("hint", {})
        finally:
            del sys.modules['google.generativeai']
            del sys.modules['google']

    def test_create_gemini_provider(self):
        """Should create provider with convenience function."""
        import types
        mock_package = types.ModuleType('google')
        mock_module = types.ModuleType('google.generativeai')
        mock_module.configure = Mock()
        mock_module.GenerativeModel = Mock()
        mock_package.generativeai = mock_module
        sys.modules['google'] = mock_package
        sys.modules['google.generativeai'] = mock_module
        try:
            with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
                from src.gemini_explanation_provider import create_gemini_provider
                provider = create_gemini_provider()
                self.assertIsNotNone(provider)
        finally:
            del sys.modules['google.generativeai']
            del sys.modules['google']


if __name__ == "__main__":
    unittest.main()