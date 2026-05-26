from datetime import datetime
import os
from temporalio import activity
import requests

from models.requests import WeatherRequest


@activity.defn
async def get_time_activity() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


@activity.defn
async def get_weather_activity(request: WeatherRequest) -> str:
    response = requests.get(
        f"https://wttr.in/{request.city}?format=%C+%t",
        timeout=10
    )
    return f"{request.city}: {response.text.strip()}"


@activity.defn
async def list_files_activity() -> str:
    files = [f for f in os.listdir('.') if f.endswith('.py')]
    return f"Python files: {', '.join(files[:5])}"