"""Tests for activities/get_weather_alerts.py."""

import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from activities.get_weather_alerts import (
    get_weather_alerts,
    GetWeatherAlertsRequest,
    WEATHER_ALERTS_TOOL_OAI,
    _alerts_url,
)


class TestAlertsUrl:
    def test_builds_correct_url(self):
        url = _alerts_url("CA")
        assert url == "https://api.weather.gov/alerts/active/area/CA"

    def test_different_state(self):
        url = _alerts_url("NY")
        assert "NY" in url


class TestWeatherAlertsToolDefinition:
    def test_has_correct_structure(self):
        assert WEATHER_ALERTS_TOOL_OAI["type"] == "function"
        assert WEATHER_ALERTS_TOOL_OAI["name"] == "get_weather_alerts"
        assert WEATHER_ALERTS_TOOL_OAI["strict"] is True
        assert "parameters" in WEATHER_ALERTS_TOOL_OAI


class TestGetWeatherAlerts:
    @pytest.mark.asyncio
    async def test_returns_json_data(self):
        mock_data = {"type": "FeatureCollection", "features": [{"properties": {"event": "Flood Watch"}}]}

        with patch(
            "activities.get_weather_alerts._make_nws_request",
            new_callable=AsyncMock,
            return_value=mock_data,
        ):
            result = await get_weather_alerts(GetWeatherAlertsRequest(state="CA"))

        data = json.loads(result)
        assert data["type"] == "FeatureCollection"
        assert "Flood Watch" in json.dumps(data)
