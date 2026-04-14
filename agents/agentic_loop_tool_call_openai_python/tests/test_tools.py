"""Tests for tools — registry, handler lookup, and individual tool functions."""

import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from tools import get_handler, get_tools
from tools.get_weather import get_weather_alerts, GetWeatherAlertsRequest
from tools.get_location import get_ip_address, get_location_info, GetLocationRequest


class TestToolRegistry:
    def test_get_handler_weather(self):
        assert get_handler("get_weather_alerts") is get_weather_alerts

    def test_get_handler_ip(self):
        assert get_handler("get_ip_address") is get_ip_address

    def test_get_handler_location(self):
        assert get_handler("get_location_info") is get_location_info

    def test_get_handler_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown tool name"):
            get_handler("nonexistent_tool")

    def test_get_tools_returns_three(self):
        tools = get_tools()
        assert len(tools) == 3
        for tool in tools:
            assert "name" in tool


class TestGetWeatherAlerts:
    @pytest.mark.asyncio
    async def test_calls_correct_url(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"type": "FeatureCollection", "features": []}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("tools.get_weather.httpx.AsyncClient", return_value=mock_client):
            result = await get_weather_alerts(GetWeatherAlertsRequest(state="NY"))

        data = json.loads(result)
        assert data["type"] == "FeatureCollection"
        url = mock_client.get.call_args[0][0]
        assert "NY" in url


class TestGetIpAddress:
    @pytest.mark.asyncio
    async def test_returns_ip(self):
        mock_response = MagicMock()
        mock_response.text = "8.8.8.8\n"
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("tools.get_location.httpx.AsyncClient", return_value=mock_client):
            result = await get_ip_address()

        assert result == "8.8.8.8"


class TestGetLocationInfo:
    @pytest.mark.asyncio
    async def test_returns_formatted_location(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "city": "New York",
            "regionName": "New York",
            "country": "United States",
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("tools.get_location.httpx.AsyncClient", return_value=mock_client):
            result = await get_location_info(GetLocationRequest(ipaddress="8.8.8.8"))

        assert result == "New York, New York, United States"
