# ABOUTME: Tests for weather workflows and activities.
# Covers format_alert, make_nws_request activity, and GetAlerts/GetForecast workflows.

import pytest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from activities.weather_activities import make_nws_request
from workflows.weather_workflows import format_alert, GetAlerts, GetForecast


class TestFormatAlert:
    """Tests for the format_alert pure function."""

    def test_format_alert_complete(self):
        """All fields present are rendered correctly."""
        feature = {
            "properties": {
                "event": "Tornado Warning",
                "areaDesc": "Dallas County",
                "severity": "Extreme",
                "description": "Take shelter immediately.",
                "instruction": "Move to interior room.",
            }
        }
        result = format_alert(feature)
        assert "Tornado Warning" in result
        assert "Dallas County" in result
        assert "Extreme" in result
        assert "Take shelter immediately." in result
        assert "Move to interior room." in result

    def test_format_alert_missing_fields(self):
        """Missing keys fall back to defaults via .get()."""
        feature = {"properties": {}}
        result = format_alert(feature)
        assert "Unknown" in result
        assert "No description available" in result
        assert "No specific instructions provided" in result


class TestMakeNwsRequest:
    """Tests for the make_nws_request activity (mocked httpx)."""

    @pytest.mark.asyncio
    async def test_make_nws_request_success(self):
        """Successful API call returns parsed JSON."""
        expected = {"type": "FeatureCollection", "features": []}
        mock_response = MagicMock()
        mock_response.json.return_value = expected
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("activities.weather_activities.httpx.AsyncClient", return_value=mock_client):
            result = await make_nws_request("https://api.weather.gov/alerts")

        assert result == expected
        mock_client.get.assert_called_once()
        call_kwargs = mock_client.get.call_args
        headers = call_kwargs.kwargs["headers"]
        assert headers["User-Agent"] == "weather-app/1.0"
        assert headers["Accept"] == "application/geo+json"

    @pytest.mark.asyncio
    async def test_make_nws_request_http_error(self):
        """HTTP errors propagate as httpx.HTTPStatusError."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("activities.weather_activities.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(httpx.HTTPStatusError):
                await make_nws_request("https://api.weather.gov/alerts")


# Module-level mock activity factories to avoid Python name mangling of
# __temporal_activity_definition inside class bodies.

def _make_mock_activity(return_value=None, side_effect=None):
    """Create an @activity.defn-decorated async mock matching make_nws_request."""
    if side_effect:
        @activity.defn(name="make_nws_request")
        async def mock_make_nws_request(url: str) -> dict[str, Any] | None:
            return side_effect(url)
        # Store side_effect so the test can swap it
        mock_make_nws_request._side_effect = side_effect
    else:
        @activity.defn(name="make_nws_request")
        async def mock_make_nws_request(url: str) -> dict[str, Any] | None:
            return return_value
    return mock_make_nws_request


class TestGetAlertsWorkflow:
    """Tests for GetAlerts workflow using WorkflowEnvironment."""

    @pytest.mark.asyncio
    async def test_get_alerts_with_features(self):
        """Workflow returns formatted alerts when features exist."""
        mock_data = {
            "features": [
                {
                    "properties": {
                        "event": "Flood Watch",
                        "areaDesc": "Travis County",
                        "severity": "Moderate",
                        "description": "Flooding possible.",
                        "instruction": "Stay alert.",
                    }
                }
            ]
        }
        mock_act = _make_mock_activity(return_value=mock_data)

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="test-alerts",
                workflows=[GetAlerts],
                activities=[mock_act],
            ):
                result = await env.client.execute_workflow(
                    GetAlerts.get_alerts,
                    "TX",
                    id="test-get-alerts",
                    task_queue="test-alerts",
                )
        assert "Flood Watch" in result
        assert "Travis County" in result

    @pytest.mark.asyncio
    async def test_get_alerts_no_data(self):
        """Workflow returns fallback when activity returns None."""
        mock_act = _make_mock_activity(return_value=None)

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="test-alerts-none",
                workflows=[GetAlerts],
                activities=[mock_act],
            ):
                result = await env.client.execute_workflow(
                    GetAlerts.get_alerts,
                    "TX",
                    id="test-get-alerts-none",
                    task_queue="test-alerts-none",
                )
        assert result == "Unable to fetch alerts or no alerts found."

    @pytest.mark.asyncio
    async def test_get_alerts_no_features_key(self):
        """Workflow returns fallback when response has no 'features' key."""
        mock_act = _make_mock_activity(return_value={})

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="test-alerts-empty",
                workflows=[GetAlerts],
                activities=[mock_act],
            ):
                result = await env.client.execute_workflow(
                    GetAlerts.get_alerts,
                    "TX",
                    id="test-get-alerts-empty",
                    task_queue="test-alerts-empty",
                )
        assert result == "Unable to fetch alerts or no alerts found."


class TestGetForecastWorkflow:
    """Tests for GetForecast workflow using WorkflowEnvironment."""

    @pytest.mark.asyncio
    async def test_get_forecast_success(self):
        """Workflow returns formatted forecast from two-stage activity calls."""
        points_data = {
            "properties": {
                "forecast": "https://api.weather.gov/gridpoints/TOP/31,80/forecast"
            }
        }
        forecast_data = {
            "properties": {
                "periods": [
                    {
                        "name": "Tonight",
                        "temperature": 65,
                        "temperatureUnit": "F",
                        "windSpeed": "5 mph",
                        "windDirection": "S",
                        "detailedForecast": "Partly cloudy.",
                    }
                ]
            }
        }

        call_count = 0

        def side_effect(url: str):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return points_data
            return forecast_data

        mock_act = _make_mock_activity(side_effect=side_effect)

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="test-forecast",
                workflows=[GetForecast],
                activities=[mock_act],
            ):
                result = await env.client.execute_workflow(
                    GetForecast.get_forecast,
                    args=[40.0, -89.0],
                    id="test-get-forecast",
                    task_queue="test-forecast",
                )
        assert "Tonight" in result
        assert "65" in result
        assert "Partly cloudy." in result

    @pytest.mark.asyncio
    async def test_get_forecast_no_points_data(self):
        """Workflow returns fallback when first activity call returns None."""
        mock_act = _make_mock_activity(return_value=None)

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="test-forecast-none",
                workflows=[GetForecast],
                activities=[mock_act],
            ):
                result = await env.client.execute_workflow(
                    GetForecast.get_forecast,
                    args=[40.0, -89.0],
                    id="test-get-forecast-none",
                    task_queue="test-forecast-none",
                )
        assert result == "Unable to fetch forecast data for this location."
