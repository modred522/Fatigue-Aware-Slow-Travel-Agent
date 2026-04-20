import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from travel_agent.api import api


class SettingsApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api)

    def test_update_and_get_settings_with_generic_llm_config(self):
        response = self.client.post(
            "/api/settings",
            json={
                "amap_api_key": "amap-test-key",
                "llm_api_key": "llm-test-key",
                "llm_base_url": "https://example.com/v1",
                "llm_model": "kimi-k2",
                "llm_temperature": 0.7,
            },
        )
        self.assertEqual(response.status_code, 200)

        response = self.client.get("/api/settings")
        data = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data["amap_api_key_set"])
        self.assertTrue(data["llm_api_key_set"])
        self.assertEqual(data["llm_base_url"], "https://example.com/v1")
        self.assertEqual(data["llm_model"], "kimi-k2")
        self.assertEqual(data["llm_temperature"], 0.7)

    def test_fetch_models_endpoint(self):
        with patch("travel_agent.api.list_models", return_value=["kimi-k2", "qwen-plus"]):
            response = self.client.post(
                "/api/settings/models",
                json={"llm_api_key": "abc", "llm_base_url": "https://example.com/v1"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"models": [{"id": "kimi-k2"}, {"id": "qwen-plus"}]})

    def test_fetch_map_settings_endpoint(self):
        self.client.post(
            "/api/settings",
            json={"amap_api_key": "amap-public-key"},
        )

        response = self.client.get("/api/settings/map")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"amap_api_key": "amap-public-key", "amap_api_key_set": True},
        )


if __name__ == "__main__":
    unittest.main()
