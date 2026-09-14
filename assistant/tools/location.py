"""
Location Tool for JARVIS (Phase 1.5).

Provides structured, real-world location information through a clean provider
boundary. Designed to support IP geolocation and future Windows device GPS
without faking coordinates or inventing data.

Returns canonical ToolResult dictionary with structured location payload.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from assistant.core.log import get
from assistant.tools.schemas import fail, ok

log = get("tools.location")


class LocationProvider(ABC):
    """Abstract interface for obtaining geographic location."""

    @abstractmethod
    def get_location(self, purpose: str = "") -> Dict[str, Any]:
        """Fetch current location.

        Returns dict containing at minimum:
            status: "SUCCESS" | "UNAVAILABLE" | "PERMISSION_DENIED" | "ERROR"
            city: str
            region: str
            country: str
            latitude: Optional[float]
            longitude: Optional[float]
            accuracy: str ("city" | "device")
            source: str
        """
        raise NotImplementedError


class IPLocationProvider(LocationProvider):
    """Fetches real-world approximate location using IP geolocation with strict timeouts."""

    def __init__(self, timeout_s: float = 5.0):
        self.timeout_s = timeout_s

    def get_location(self, purpose: str = "") -> Dict[str, Any]:
        # Primary: ip-api.com (reliable, JSON, free for non-commercial)
        endpoints = [
            ("http://ip-api.com/json/?fields=status,message,country,regionName,city,lat,lon,query", self._parse_ip_api),
            ("https://ipapi.co/json/", self._parse_ipapi_co),
        ]

        for url, parser in endpoints:
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "JARVIS-Location/1.5"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    if resp.status == 200:
                        raw = json.loads(resp.read().decode("utf-8"))
                        parsed = parser(raw)
                        if parsed and parsed.get("status") == "SUCCESS":
                            return parsed
            except urllib.error.URLError as e:
                log.warning("[LOCATION] Endpoint %s failed: %s", url, e)
            except Exception as e:
                log.warning("[LOCATION] Unexpected error querying %s: %s", url, e)

        return {
            "status": "UNAVAILABLE",
            "city": None,
            "region": None,
            "country": None,
            "latitude": None,
            "longitude": None,
            "accuracy": None,
            "source": "none",
            "error": "Unable to determine location from network providers.",
        }

    @staticmethod
    def _parse_ip_api(data: dict) -> Optional[Dict[str, Any]]:
        if data.get("status") == "success":
            return {
                "status": "SUCCESS",
                "city": data.get("city") or "Unknown",
                "region": data.get("regionName") or "Unknown",
                "country": data.get("country") or "Unknown",
                "latitude": float(data["lat"]) if "lat" in data else None,
                "longitude": float(data["lon"]) if "lon" in data else None,
                "accuracy": "city",
                "source": "ip_geolocation",
            }
        return None

    @staticmethod
    def _parse_ipapi_co(data: dict) -> Optional[Dict[str, Any]]:
        if "city" in data and not data.get("error"):
            return {
                "status": "SUCCESS",
                "city": data.get("city") or "Unknown",
                "region": data.get("region") or "Unknown",
                "country": data.get("country_name") or "Unknown",
                "latitude": float(data["latitude"]) if "latitude" in data else None,
                "longitude": float(data["longitude"]) if "longitude" in data else None,
                "accuracy": "city",
                "source": "ip_geolocation",
            }
        return None


# Global active provider instance (can be swapped in testing or when Windows GPS is available)
_ACTIVE_PROVIDER: LocationProvider = IPLocationProvider()


def set_location_provider(provider: LocationProvider) -> None:
    """Inject a custom or mock location provider."""
    global _ACTIVE_PROVIDER
    _ACTIVE_PROVIDER = provider


def get_location_provider() -> LocationProvider:
    return _ACTIVE_PROVIDER


def get_location(purpose: str = "") -> dict:
    """Tool handler for ToolRegistry: returns the user's current city, region, and coordinates."""
    provider = get_location_provider()
    loc = provider.get_location(purpose=purpose)

    status = loc.get("status", "ERROR")
    if status != "SUCCESS" or loc.get("latitude") is None:
        return fail(
            "I could not determine your current location, sir.",
            error="location_unavailable",
            data=loc,
        )

    city = loc.get("city")
    region = loc.get("region")
    country = loc.get("country")
    place = ", ".join(part for part in [city, region, country] if part and part != "Unknown")

    return ok(
        f"Current location is {place}.",
        data=loc,
    )

