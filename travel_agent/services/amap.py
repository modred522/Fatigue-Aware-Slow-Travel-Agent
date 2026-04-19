from __future__ import annotations

from typing import Literal

import requests

from travel_agent.config import get_config
from travel_agent.errors import ConfigError, ExternalServiceError
from travel_agent.logger import log_geocode, log_route

TransportMode = Literal["walking", "transit", "driving"]


class AMapClient:
    BASE_URL = "https://restapi.amap.com/v3"

    def __init__(self) -> None:
        config = get_config()
        if not config.amap_api_key:
            raise ConfigError(
                "AMap API key is not configured. Please set it in Settings.",
                "amap_api_key",
            )
        self.api_key = config.amap_api_key

    def geocode(self, address: str, city: str | None = None) -> str:
        params = {"address": address, "key": self.api_key}
        if city:
            params["city"] = city
            # Limit search to specified city only to avoid matching distant locations with same name
            params["citylimit"] = "true"
        try:
            data = self._get("/geocode/geo", params)
            geocodes = data.get("geocodes") or []
            if not geocodes:
                log_geocode(address, city, None, "No geocodes found")
                raise ExternalServiceError(
                    f"Could not find coordinates for '{address}' in '{city}'. "
                    f"Please check the spelling or try a different location."
                )
            result = geocodes[0]["location"]
            log_geocode(address, city, result)
            return result
        except ExternalServiceError:
            raise

    def validate_location(self, city: str, location: str) -> dict:
        """Validate that a location exists within the specified city.

        Returns a dict with validation results including:
        - valid: bool
        - formatted_address: str | None
        - city_match: bool
        """
        params = {
            "address": location,
            "key": self.api_key,
            "city": city,
            "citylimit": "true",
        }
        try:
            data = self._get("/geocode/geo", params)
            geocodes = data.get("geocodes") or []

            if not geocodes:
                return {
                    "valid": False,
                    "formatted_address": None,
                    "city_match": False,
                    "message": f"Could not find '{location}' in '{city}'. Please check the spelling."
                }

            # Check if the returned location is actually in the specified city
            first_result = geocodes[0]
            formatted_address = first_result.get("formatted_address", "")
            province = first_result.get("province", "")
            result_city = first_result.get("city", "")
            district = first_result.get("district", "")

            # Check if result city matches requested city (handle cases like "杭州市" vs "杭州")
            city_match = (
                city in result_city or
                result_city in city or
                city in district or
                city in formatted_address
            )

            if not city_match:
                return {
                    "valid": False,
                    "formatted_address": formatted_address,
                    "city_match": False,
                    "message": f"'{location}' was found in '{result_city}', not in '{city}'. Please check your input."
                }

            return {
                "valid": True,
                "formatted_address": formatted_address,
                "city_match": True,
                "location": first_result.get("location"),
                "message": None
            }

        except ExternalServiceError as e:
            return {
                "valid": False,
                "formatted_address": None,
                "city_match": False,
                "message": str(e)
            }

    def walking_route(self, origin_name: str, destination_name: str, city: str) -> dict:
        """Legacy method for backward compatibility."""
        return self.route(origin_name, destination_name, city, mode="walking")

    def route(
        self,
        origin_name: str,
        destination_name: str,
        city: str,
        mode: TransportMode = "walking",
    ) -> dict:
        """Calculate route using specified transport mode.

        Supports: walking, transit, driving
        Reference: https://lbs.amap.com/api/webservice/guide/api/direction/

        If walking mode fails due to distance limit, automatically falls back to driving mode.
        """
        origin_coords = self.geocode(origin_name, city)
        destination_coords = self.geocode(destination_name, city)

        result = self._route_with_coords(
            origin_coords, destination_coords, city, mode
        )

        # If walking failed with distance limit, try driving as fallback
        if mode == "walking" and result.get("_fallback_to_driving"):
            del result["_fallback_to_driving"]
            try:
                driving_result = self._route_with_coords(
                    origin_coords, destination_coords, city, "driving"
                )
                driving_result["mode"] = "walking"  # Keep original mode
                driving_result["_note"] = "Distance calculated using driving route (walking limit exceeded)"
                log_route(origin_name, destination_name, city, mode, driving_result.get("distance_meters"), fallback_used=True)
                return driving_result
            except ExternalServiceError as e:
                log_route(origin_name, destination_name, city, mode, error=str(e), fallback_used=True)
                raise  # If driving also fails, raise original error

        log_route(origin_name, destination_name, city, mode, result.get("distance_meters"))
        return result

    def _route_with_coords(
        self,
        origin_coords: str,
        destination_coords: str,
        city: str,
        mode: TransportMode,
    ) -> dict:
        """Internal method to calculate route with pre-resolved coordinates."""
        endpoint_map = {
            "walking": "/direction/walking",
            "transit": "/direction/transit/integrated",
            "driving": "/direction/driving",
        }

        endpoint = endpoint_map.get(mode, "/direction/walking")
        params = {
            "origin": origin_coords,
            "destination": destination_coords,
            "key": self.api_key,
        }

        # Transit mode requires additional city parameters
        if mode == "transit":
            params["city"] = city
            params["cityd"] = city

        data = self._get(endpoint, params, allow_walking_fallback=(mode == "walking"))

        # Check for fallback marker (walking distance limit exceeded)
        if data.get("_fallback_to_driving"):
            return data

        # Parse response based on transport mode
        route_data = data.get("route") or {}

        if mode == "transit":
            # Transit returns transits array instead of paths
            paths = route_data.get("transits") or []
            if not paths:
                raise ExternalServiceError(
                    f"Could not calculate a transit route."
                )
            # Transit includes cost, duration, and walking distance
            first_route = paths[0]
            return {
                "distance_meters": int(first_route.get("walking_distance", 0)),
                "duration_seconds": int(first_route.get("duration", 0)),
                "cost": float(first_route.get("cost", 0)),
                "origin_coords": origin_coords,
                "destination_coords": destination_coords,
                "mode": mode,
            }
        else:
            # Other modes use paths array
            paths = route_data.get("paths") or []
            if not paths:
                raise ExternalServiceError(
                    f"Could not calculate a {mode} route."
                )
            first_path = paths[0]
            return {
                "distance_meters": int(first_path.get("distance", 0)),
                "duration_seconds": int(first_path.get("duration", 0)),
                "origin_coords": origin_coords,
                "destination_coords": destination_coords,
                "mode": mode,
            }

    def _get(self, path: str, params: dict, allow_walking_fallback: bool = False) -> dict:
        try:
            response = requests.get(f"{self.BASE_URL}{path}", params=params, timeout=10)
            response.raise_for_status()
        except requests.Timeout as exc:
            raise ExternalServiceError("AMap request timed out.") from exc
        except requests.ConnectionError as exc:
            raise ExternalServiceError("Cannot connect to AMap API. Check your network.") from exc
        except requests.HTTPError as exc:
            raise ExternalServiceError(
                f"AMap returned HTTP {exc.response.status_code}."
            ) from exc

        data = response.json()
        if data.get("status") == "0":
            infocode = data.get("infocode", "")
            info = data.get("info", "Unknown AMap error")
            if infocode == "10001":
                raise ConfigError(
                    "AMap API key is invalid. Please check your AMap settings.",
                    "amap_api_key",
                )
            if infocode == "20803":
                if allow_walking_fallback:
                    # Return marker for caller to handle fallback
                    return {"_fallback_to_driving": True}
                raise ExternalServiceError(
                    f"Route distance exceeds limit for this transport mode. "
                    f"Try using driving mode for longer distances. (AMap error: {info})"
                )
            raise ExternalServiceError(f"AMap error: {info} (code: {infocode})")
        return data
