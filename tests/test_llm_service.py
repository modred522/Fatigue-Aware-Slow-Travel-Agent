import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from travel_agent.errors import ConfigError
from travel_agent.services import llm


class LlmServiceTests(unittest.TestCase):
    def test_get_llm_client_requires_api_key(self):
        with patch("travel_agent.services.llm.get_config", return_value=SimpleNamespace(llm_api_key="", llm_base_url="", llm_model="test-model", llm_temperature=0.3)):
            with self.assertRaises(ConfigError):
                llm.get_llm_client()

    def test_list_models_returns_sorted_ids(self):
        mock_client = Mock()
        mock_client.models.list.return_value = SimpleNamespace(
            data=[SimpleNamespace(id="z-model"), SimpleNamespace(id="a-model")]
        )
        with patch("travel_agent.services.llm.get_llm_client", return_value=mock_client):
            result = llm.list_models(api_key="key", base_url="https://example.com/v1")
        self.assertEqual(result, ["a-model", "z-model"])

    def test_is_kimi_model_identifies_kimi_models(self):
        self.assertTrue(llm._is_kimi_model("kimi-k2-0711-preview"))
        self.assertTrue(llm._is_kimi_model("kimi-latest"))
        self.assertTrue(llm._is_kimi_model("Kimi-K2"))  # case insensitive
        self.assertFalse(llm._is_kimi_model("qwen-turbo"))
        self.assertFalse(llm._is_kimi_model("gpt-4"))

    def test_get_temperature_for_kimi_models(self):
        # Kimi models return None (temperature not used with disabled thinking)
        config = SimpleNamespace(llm_model="kimi-k2", llm_temperature=0.5)
        self.assertIsNone(llm._get_temperature(config))

    def test_get_temperature_for_other_models(self):
        # Non-Kimi models should use configured temperature
        config = SimpleNamespace(llm_model="qwen-turbo", llm_temperature=0.7)
        self.assertEqual(llm._get_temperature(config), 0.7)

    def test_list_models_fallback_for_moonshot(self):
        # Moonshot API should return fallback models without calling API
        result = llm.list_models(api_key="key", base_url="https://api.moonshot.cn/v1")
        self.assertIn("kimi-k2-0711-preview", result)
        self.assertIn("kimi-latest", result)


if __name__ == "__main__":
    unittest.main()
