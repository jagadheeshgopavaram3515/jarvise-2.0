"""
Weather Tool for JARVIS (Phase 1.5).

Provides real-world weather conditions, temperature, humidity, wind, and
precipitation using a clean WeatherProvider interface backed by Open-Meteo.
Supports input via explicit coordinates (from location tool chaining) or city names.

Never hallucinates weather data; returns structured errors when offline.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from assistant.core.log import get
from assistant.tools.schemas import fail, ok

log = get("tools.weather")

# WMO Weather interpretation codes (WW)
_WMO_CODE_MAP = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    62: "Moderate rain",
    63: "Moderate rain",
    64: "Heavy rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


class WeatherProvider(ABC):
    """Abstract interface for fetching weather data."""

    @abstractmethod
    def get_weather(
        self,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        city: str = "",
    ) -> Dict[str, Any]:
        """Fetch current weather for coordinates or city."""
        raise NotImplementedError


class OpenMeteoWeatherProvider(WeatherProvider):
    """Fetches real weather using free, open, keyless Open-Meteo APIs with strict timeouts."""

    def __init__(self, timeout_s: float = 6.0):
        self.timeout_s = timeout_s

    def get_weather(
        self,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        city: str = "",
    ) -> Dict[str, Any]:
        target_lat = latitude
        target_lon = longitude
        location_label = city.strip() if city else ""

        # 1. Geocode city if coordinates not given
        if (target_lat is None or target_lon is None) and city:
            geo = self._geocode_city(city)
            if geo:
                target_lat = geo["latitude"]
                target_lon = geo["longitude"]
                location_label = geo["name"]

        if target_lat is None or target_lon is None:
            return {
                "status": "ERROR",
                "error": f"Could not determine geographic coordinates for '{city}'." if city else "Missing coordinates or city name.",
            }

        # 2. Fetch current weather from Open-Meteo
        try:
            url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={target_lat:.4f}&longitude={target_lon:.4f}&"
                f"current=temperature_2m,relative_humidity_2m,apparent_temperature,"
                f"precipitation,weather_code,wind_speed_10m&timezone=auto"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "JARVIS-Weather/1.5"})
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                if resp.status != 200:
                    return {"status": "ERROR", "error": f"Weather API returned HTTP {resp.status}"}
                payload = json.loads(resp.read().decode("utf-8"))

            current = payload.get("current", {})
            code = current.get("weather_code", 0)
            condition = _WMO_CODE_MAP.get(code, "Clear")

            if not location_label:
                location_label = f"{target_lat:.2f}°N, {target_lon:.2f}°E"

            return {
                "status": "SUCCESS",
                "location": location_label,
                "latitude": target_lat,
                "longitude": target_lon,
                "temperature_c": current.get("temperature_2m"),
                "apparent_temperature_c": current.get("apparent_temperature"),
                "humidity_percent": current.get("relative_humidity_2m"),
                "wind_speed_kmh": current.get("wind_speed_10m"),
                "precipitation_mm": current.get("precipitation", 0.0),
                "condition": condition,
                "source": "open-meteo",
                "timestamp": current.get("time"),
            }

        except urllib.error.URLError as e:
            log.warning("[WEATHER] Network error fetching weather: %s", e)
            return {"status": "ERROR", "error": f"Network error contacting weather service: {e}"}
        except Exception as e:
            log.warning("[WEATHER] Error fetching weather: %s", e)
            return {"status": "ERROR", "error": str(e)}

    def _geocode_city(self, city_name: str) -> Optional[Dict[str, Any]]:
        """Geocode city name using Open-Meteo geocoding API."""
        try:
            encoded = urllib.parse.quote(city_name.strip())
            url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded}&count=1&language=en&format=json"
            req = urllib.request.Request(url, headers={"User-Agent": "JARVIS-Weather/1.5"})
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    results = data.get("results")
                    if results and len(results) > 0:
                        first = results[0]
                        name_parts = [first.get("name"), first.get("admin1"), first.get("country")]
                        full_name = ", ".join(p for p in name_parts if p)
                        return {
                            "latitude": float(first["latitude"]),
                            "longitude": float(first["longitude"]),
                            "name": full_name,
                        }
        except Exception as e:
            log.warning("[WEATHER] Geocoding failed for '%s': %s", city_name, e)
        return None


# Global active weather provider
_ACTIVE_WEATHER_PROVIDER: WeatherProvider = OpenMeteoWeatherProvider()


def set_weather_provider(provider: WeatherProvider) -> None:
    """Inject a custom or mock weather provider."""
    global _ACTIVE_WEATHER_PROVIDER
    _ACTIVE_WEATHER_PROVIDER = provider


def get_weather_provider() -> WeatherProvider:
    return _ACTIVE_WEATHER_PROVIDER


def get_weather(
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    city: str = "",
) -> dict:
    """Tool handler for ToolRegistry: fetches real weather for coordinates or city."""
    # Convert string lat/lon if passed by LLM as strings
    try:
        lat = float(latitude) if latitude is not None and str(latitude).strip() else None
    except (ValueError, TypeError):
        lat = None

    try:
        lon = float(longitude) if longitude is not None and str(longitude).strip() else None
    except (ValueError, TypeError):
        lon = None

    provider = get_weather_provider()
    res = provider.get_weather(latitude=lat, longitude=lon, city=city or "")

    if res.get("status") != "SUCCESS":
        return fail(
            f"I could not retrieve the weather for that location, sir.",
            error=res.get("error", "weather_unavailable"),
            data=res,
        )

    loc_name = res.get("location", "the requested area")
    temp = res.get("temperature_c")
    cond = res.get("condition", "Clear")
    humid = res.get("humidity_percent")
    wind = res.get("wind_speed_kmh")
    rain = res.get("precipitation_mm", 0.0)

    msg = f"The current weather in {loc_name} is {temp}°C and {cond}."
    if humid is not None:
        msg += f" Humidity is {humid}%."
    if wind is not None:
        msg += f" Wind speed is {wind} km/h."
    if rain and rain > 0:
        msg += f" Precipitation is {rain} mm."

    return ok(msg, data=res)

