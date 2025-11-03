# Uncomment and comment out the tools you want to use

# Location and weather related tools
from .get_location import get_location_info, get_ip_address
from .get_weather import get_weather_alerts

def get_handler(tool_name: str):
    if tool_name == "get_location_info":
        return get_location_info
    if tool_name == "get_ip_address":
        return get_ip_address
    if tool_name == "get_weather_alerts":
        return get_weather_alerts

def get_tools():
    return [get_weather.WEATHER_ALERTS_TOOL_OAI, 
            get_location.GET_LOCATION_TOOL_OAI,
            get_location.GET_IP_ADDRESS_TOOL_OAI]

# Random number tool
# from .random_stuff import get_random_number, RANDOM_NUMBER_TOOL_OAI

# def get_handler(tool_name: str):
#     if tool_name == "get_random_number":
#         return get_random_number

# def get_tools():
#     return [RANDOM_NUMBER_TOOL_OAI]