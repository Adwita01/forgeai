"""Keyless weather via open-meteo.com (geocoding + forecast)."""
from __future__ import annotations

import requests

_GEO = "https://geocoding-api.open-meteo.com/v1/search"
_FC = "https://api.open-meteo.com/v1/forecast"


def run(args: dict, ctx=None) -> dict:
    lat, lon, name = args.get("latitude"), args.get("longitude"), args.get("location") or args.get("city")
    if (lat is None or lon is None) and name:
        g = requests.get(_GEO, params={"name": name, "count": 1}, timeout=10)
        g.raise_for_status()
        res = (g.json().get("results") or [])
        if not res:
            raise ValueError(f"location not found: {name}")
        lat, lon, name = res[0]["latitude"], res[0]["longitude"], f"{res[0]['name']}, {res[0].get('country', '')}"
    if lat is None or lon is None:
        raise ValueError("need 'location' or 'latitude'+'longitude'")
    r = requests.get(
        _FC,
        params={
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,wind_speed_10m,wind_gusts_10m,precipitation,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
            "forecast_days": 3, "timezone": "auto",
        },
        timeout=10,
    )
    r.raise_for_status()
    d = r.json()
    return {"location": name or f"{lat},{lon}", "current": d.get("current"), "daily": d.get("daily"), "units": d.get("current_units")}
