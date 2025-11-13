# Uncomment and comment out the tools you want to use

# Location and weather related tools
from . import get_location
from . import get_weather

def get_handler(tool_name: str):
    if tool_name == "get_location_info":
        return get_location.get_location_info
    if tool_name == "get_ip_address":
        return get_location.get_ip_address
    if tool_name == "get_weather_alerts":
        return get_weather.get_weather_alerts

def get_tools():
    # Return combined tool declarations for Gemini API
    # Gemini expects a list with a single dict containing all function_declarations
    all_declarations = []

    # Extract function_declarations from each tool
    for tool in [
        get_location.GET_IP_ADDRESS_TOOL_GEMINI,
        get_location.GET_LOCATION_TOOL_GEMINI,
        get_weather.GET_WEATHER_ALERTS_TOOL_GEMINI,
    ]:
        all_declarations.extend(tool["function_declarations"])

    # Return as a single dict with all function declarations
    return [{"function_declarations": all_declarations}]

# Random number tools
# from . import random_stuff
#
# def get_handler(tool_name: str):
#     if tool_name == "get_random_number":
#         return random_stuff.get_random_number
#     if tool_name == "get_random_string":
#         return random_stuff.get_random_string
#
# def get_tools():
#     return [
#         random_stuff.GET_RANDOM_NUMBER_TOOL_GEMINI,
#         random_stuff.GET_RANDOM_STRING_TOOL_GEMINI,
#     ]
