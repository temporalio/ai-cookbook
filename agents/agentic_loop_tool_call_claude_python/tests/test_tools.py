"""Tests for tools — registry, handler lookup, and individual tool functions."""

import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from tools import get_handler, get_tools
from tools.get_weather import get_weather_alerts, GetWeatherAlertsRequest
from tools.get_location import get_ip_address, get_location_info, GetLocationRequest
from tools.random_stuff import get_random_number, generate_random_text, GenerateRandomTextRequest


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
        names = {t["name"] for t in tools}
        assert "get_weather_alerts" in names
        assert "get_location_info" in names
        assert "get_ip_address" in names


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
            result = await get_weather_alerts(GetWeatherAlertsRequest(state="CA"))

        data = json.loads(result)
        assert data["type"] == "FeatureCollection"
        mock_client.get.assert_called_once()
        url = mock_client.get.call_args[0][0]
        assert "CA" in url


class TestGetIpAddress:
    @pytest.mark.asyncio
    async def test_returns_ip(self):
        mock_response = MagicMock()
        mock_response.text = "1.2.3.4\n"
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("tools.get_location.httpx.AsyncClient", return_value=mock_client):
            result = await get_ip_address()

        assert result == "1.2.3.4"


class TestGetLocationInfo:
    @pytest.mark.asyncio
    async def test_returns_formatted_location(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "city": "San Francisco",
            "regionName": "California",
            "country": "United States",
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("tools.get_location.httpx.AsyncClient", return_value=mock_client):
            result = await get_location_info(GetLocationRequest(ipaddress="1.2.3.4"))

        assert result == "San Francisco, California, United States"


class TestRandomTools:
    @pytest.mark.asyncio
    async def test_random_number_in_range(self):
        result = await get_random_number()
        num = int(result)
        assert 1 <= num <= 100

    @pytest.mark.asyncio
    async def test_generate_random_text_word_count(self):
        result = await generate_random_text(GenerateRandomTextRequest(length=5))
        # Result ends with a period, strip it before counting words
        words = result.rstrip(".").split()
        assert len(words) == 5
