import pytest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from activities.tool_activities import get_time_activity, get_weather_activity, list_files_activity
from models.requests import WeatherRequest


class TestGetTimeActivity:

    @pytest.mark.asyncio
    async def test_returns_formatted_time(self):
        result = await get_time_activity()
        # Should match format: YYYY-MM-DD HH:MM:SS
        assert len(result) == 19
        assert result[4] == "-"
        assert result[10] == " "
        assert result[13] == ":"


class TestGetWeatherActivity:

    @pytest.mark.asyncio
    async def test_returns_weather_for_city(self):
        mock_response = MagicMock()
        mock_response.text = "Sunny +22°C"

        with patch("activities.tool_activities.requests.get", return_value=mock_response) as mock_get:
            result = await get_weather_activity(WeatherRequest(city="London"))

            mock_get.assert_called_once()
            assert "London" in result
            assert "Sunny" in result

    @pytest.mark.asyncio
    async def test_calls_correct_api(self):
        mock_response = MagicMock()
        mock_response.text = "Cloudy +15°C"

        with patch("activities.tool_activities.requests.get", return_value=mock_response) as mock_get:
            await get_weather_activity(WeatherRequest(city="Paris"))

            call_args = mock_get.call_args
            assert "wttr.in/Paris" in call_args[0][0]


class TestListFilesActivity:

    @pytest.mark.asyncio
    async def test_returns_python_files(self):
        fake_files = ["main.py", "test.py", "readme.md", "config.json", "utils.py"]

        with patch("activities.tool_activities.os.listdir", return_value=fake_files):
            result = await list_files_activity()

            assert "Python files:" in result
            assert "main.py" in result
            assert "test.py" in result
            assert "utils.py" in result
            assert "readme.md" not in result

    @pytest.mark.asyncio
    async def test_limits_to_five_files(self):
        fake_files = [f"file{i}.py" for i in range(10)]

        with patch("activities.tool_activities.os.listdir", return_value=fake_files):
            result = await list_files_activity()

            # Should only show first 5
            assert "file0.py" in result
            assert "file4.py" in result
            assert "file5.py" not in result

    @pytest.mark.asyncio
    async def test_handles_no_python_files(self):
        fake_files = ["readme.md", "config.json"]

        with patch("activities.tool_activities.os.listdir", return_value=fake_files):
            result = await list_files_activity()

            assert "Python files:" in result
