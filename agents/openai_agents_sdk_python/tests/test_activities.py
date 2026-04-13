"""Tests for activities/tools.py — calculate_circle_area and get_weather."""

import math

import pytest
from temporalio.testing import ActivityEnvironment

from activities.tools import Weather, calculate_circle_area, get_weather


class TestCalculateCircleArea:
    @pytest.mark.asyncio
    async def test_radius_five(self):
        env = ActivityEnvironment()
        result = await env.run(calculate_circle_area, 5.0)
        assert result == pytest.approx(math.pi * 25)

    @pytest.mark.asyncio
    async def test_radius_zero(self):
        env = ActivityEnvironment()
        result = await env.run(calculate_circle_area, 0.0)
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_radius_one(self):
        env = ActivityEnvironment()
        result = await env.run(calculate_circle_area, 1.0)
        assert result == pytest.approx(math.pi)


class TestGetWeather:
    @pytest.mark.asyncio
    async def test_raises_exception(self):
        env = ActivityEnvironment()
        with pytest.raises(Exception, match="This is a test error"):
            await env.run(get_weather, "New York")


class TestWeatherDataclass:
    def test_construction_and_field_access(self):
        weather = Weather(
            city="London",
            temperature_range="14-20C",
            conditions="Sunny with wind.",
        )
        assert weather.city == "London"
        assert weather.temperature_range == "14-20C"
        assert weather.conditions == "Sunny with wind."
