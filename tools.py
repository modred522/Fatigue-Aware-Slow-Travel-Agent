import os
import requests
from dotenv import load_dotenv, find_dotenv
from langchain_core.tools import tool

# load env variables
_ = load_dotenv(find_dotenv(), override=True)


def get_amap_key() -> str:
    """Get AMap API key, checking runtime env first."""
    key = os.getenv("AMAP_API_KEY")
    if not key:
        raise ValueError(
            "AMAP_API_KEY is not configured. "
            "Please set it in the Settings page or in your .env file."
        )
    return key


def get_coordinates(location_name: str, city: str = "杭州") -> str:
    """
    Helper function: Converts a location name into coordinates.
    Now includes a 'city' parameter to prevent finding locations in other provinces.
    """
    amap_key = get_amap_key()

    url = f"https://restapi.amap.com/v3/geocode/geo?address={location_name}&city={city}&key={amap_key}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.ConnectionError:
        raise Exception(
            f"Cannot connect to AMap API. Please check your network connection."
        )
    except requests.Timeout:
        raise Exception(
            f"AMap API request timed out while geocoding '{location_name}'."
        )
    except requests.HTTPError as e:
        raise Exception(
            f"AMap API returned HTTP {e.response.status_code} while geocoding '{location_name}'."
        )

    data = response.json()

    if data.get("status") == "0":
        info = data.get("info", "Unknown error")
        infocode = data.get("infocode", "")
        if infocode == "10001":
            raise Exception(
                f"AMap API key is invalid. Please check your AMAP_API_KEY configuration."
            )
        elif infocode == "10003":
            raise Exception(
                f"AMap API daily quota exceeded. Please try again tomorrow or use a different key."
            )
        raise Exception(
            f"AMap geocode API error: {info} (code: {infocode})"
        )

    if data.get("status") == "1" and data.get("geocodes"):
        return data["geocodes"][0]["location"]
    else:
        raise Exception(
            f"Could not find coordinates for '{location_name}' in city '{city}'. "
            f"Please check the location name is correct."
        )


@tool
def calculate_walking_distance(origin_name: str, destination_name: str, city: str = "杭州") -> int:
    """
    Calculates the exact walking distance (in meters) between two locations.
    This tool is crucial for monitoring the user's fatigue level.
    """
    print(f"🔧 [Tool Execution] Calculating walking distance from {origin_name} to {destination_name}...")

    # Step 1: Convert location names to coordinates
    origin_coords = get_coordinates(origin_name, city=city)
    dest_coords = get_coordinates(destination_name, city=city)

    # Step 2: Call Amap Walking Direction API
    amap_key = get_amap_key()
    url = f"https://restapi.amap.com/v3/direction/walking?origin={origin_coords}&destination={dest_coords}&key={amap_key}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.ConnectionError:
        raise Exception(
            f"Cannot connect to AMap API. Please check your network connection."
        )
    except requests.Timeout:
        raise Exception(
            f"AMap API request timed out while calculating walking distance."
        )
    except requests.HTTPError as e:
        raise Exception(
            f"AMap API returned HTTP {e.response.status_code} for walking direction request."
        )

    data = response.json()

    if data.get("status") == "0":
        info = data.get("info", "Unknown error")
        infocode = data.get("infocode", "")
        if infocode == "10001":
            raise Exception("AMap API key is invalid. Please check your AMAP_API_KEY configuration.")
        raise Exception(f"AMap direction API error: {info} (code: {infocode})")

    # Step 3: Parse the distance from the response
    if data.get("status") == "1" and data.get("route") and data.get("route").get("paths"):
        distance = int(data["route"]["paths"][0]["distance"])
        print(f"📍 [Tool Result] Distance is {distance} meters.")
        return distance
    else:
        raise Exception(
            f"Could not calculate walking distance from '{origin_name}' to '{destination_name}'. "
            f"The route may not be walkable."
        )

# ==========================================
# Test Execution Area
# ==========================================
if __name__ == "__main__":
    try:
        dist = calculate_walking_distance.invoke({
            "origin_name": "杭州断桥残雪",
            "destination_name": "杭州平湖秋月"
        })
        print(f"\n✅ Test Passed! The distance is {dist} meters.")
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
