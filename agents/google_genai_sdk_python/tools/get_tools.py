from .get_location import get_location_info, get_ip_address, GET_LOCATION_TOOL_GEMINI, GET_IP_ADDRESS_TOOL_GEMINI
from .get_weather import get_weather_alerts, GET_WEATHER_ALERTS_TOOL_GEMINI
from .random_stuff import get_random_number, get_random_string, GET_RANDOM_NUMBER_TOOL_GEMINI, GET_RANDOM_STRING_TOOL_GEMINI

def get_tools():
    return [
        GET_LOCATION_TOOL_GEMINI,
        GET_IP_ADDRESS_TOOL_GEMINI,
        GET_WEATHER_ALERTS_TOOL_GEMINI,
        GET_RANDOM_NUMBER_TOOL_GEMINI,
        GET_RANDOM_STRING_TOOL_GEMINI,
    ]
