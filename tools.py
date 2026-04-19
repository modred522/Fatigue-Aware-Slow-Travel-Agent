from langchain_core.tools import tool

from travel_agent.services.amap import AMapClient


@tool
def calculate_walking_distance(origin_name: str, destination_name: str, city: str = "Hangzhou") -> int:
    """Calculate the exact walking distance in meters between two locations."""
    route = AMapClient().walking_route(origin_name, destination_name, city)
    return route["distance_meters"]
