from pydantic import BaseModel, Field
from temporalio import activity


# Each tool takes a typed request and returns a deterministic string. They do no
# network I/O so the recipe runs (and tests run) without external services. In a
# real recipe each of these would call a separate API.
class GetWeatherRequest(BaseModel):
    city: str = Field(description="City name, e.g. 'Seattle'.")


class GetTimeRequest(BaseModel):
    timezone: str = Field(description="IANA timezone name, e.g. 'America/Los_Angeles'.")


_WEATHER_BY_CITY = {
    "seattle": "rainy, 54F",
    "austin": "sunny, 88F",
    "denver": "partly cloudy, 70F",
}

_TIME_BY_ZONE = {
    "america/los_angeles": "09:00",
    "america/chicago": "11:00",
    "america/denver": "10:00",
}


@activity.defn
async def get_weather(request: GetWeatherRequest) -> str:
    return _WEATHER_BY_CITY.get(request.city.lower(), "clear, 65F")


@activity.defn
async def get_time(request: GetTimeRequest) -> str:
    return _TIME_BY_ZONE.get(request.timezone.lower(), "12:00")
