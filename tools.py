import os
import requests
from dotenv import load_dotenv, find_dotenv
from langchain_core.tools import tool

# load env variables
_ = load_dotenv(find_dotenv(), override=True)

AMAP_API_KEY = os.getenv("AMAP_API_KEY")

def get_coordinates(location_name: str, city: str = "杭州") -> str:
    """
    Helper function: Converts a location name into coordinates.
    Now includes a 'city' parameter to prevent finding locations in other provinces.
    """
    if not AMAP_API_KEY:
        raise ValueError("AMAP_API_KEY is not set in the environment variables.")

    # Added &city= parameter to the API request
    url = f"https://restapi.amap.com/v3/geocode/geo?address={location_name}&city={city}&key={AMAP_API_KEY}"
    response = requests.get(url)
    data = response.json()
    
    if data.get("status") == "1" and data.get("geocodes"):
        return data["geocodes"][0]["location"]
    else:
        raise Exception(f"Failed to get coordinates for {location_name}. API Response: {data}")
@tool
def calculate_walking_distance(origin_name: str, destination_name: str) -> int:
    """
    Calculates the exact walking distance (in meters) between two locations.
    This tool is crucial for monitoring the user's fatigue level.
    """
    print(f"🔧 [Tool Execution] Calculating walking distance from {origin_name} to {destination_name}...")
    
    # Step 1: Convert location names to coordinates
    origin_coords = get_coordinates(origin_name)
    dest_coords = get_coordinates(destination_name)
    
    # Step 2: Call Amap Walking Direction API
    url = f"https://restapi.amap.com/v3/direction/walking?origin={origin_coords}&destination={dest_coords}&key={AMAP_API_KEY}"
    response = requests.get(url)
    data = response.json()
    
    # Step 3: Parse the distance from the response
    if data.get("status") == "1" and data.get("route") and data.get("route").get("paths"):
        # Extract distance and convert it to an integer (meters)
        distance = int(data["route"]["paths"][0]["distance"])
        print(f"📍 [Tool Result] Distance is {distance} meters.")
        return distance
    else:
        raise Exception(f"Failed to calculate walking distance. API Response: {data}")

# ==========================================
# Test Execution Area
# ==========================================
if __name__ == "__main__":
    try:
        # Let's test the distance between two spots in Hangzhou
        dist = calculate_walking_distance.invoke({
            "origin_name": "杭州断桥残雪", 
            "destination_name": "杭州平湖秋月"
        })
        print(f"\n✅ Test Passed! The distance is {dist} meters.")
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")