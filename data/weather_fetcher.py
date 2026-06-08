"""
data/weather_fetcher.py
-----------------------
Fetches weather conditions at NFL game time via OpenWeatherMap (free tier).
Requires OPENWEATHER_API_KEY in .env.

Returns None for dome/retractable-roof stadiums (weather irrelevant).
Returns None if the API key is missing or the call fails — bot degrades gracefully.
"""

import logging
import os
from datetime import datetime

import requests

logger = logging.getLogger(__name__)

# (lat, lon, is_dome)
# is_dome=True for fully enclosed and retractable-roof stadiums that routinely close.
_NFL_STADIUMS: dict[str, tuple[float, float, bool]] = {
    # AFC East
    "Buffalo Bills":          (42.774, -78.787, False),
    "Miami Dolphins":         (25.958, -80.239, False),
    "New England Patriots":   (42.091, -71.264, False),
    "New York Jets":          (40.814, -74.075, False),
    # AFC North
    "Baltimore Ravens":       (39.278, -76.623, False),
    "Cincinnati Bengals":     (39.096, -84.516, False),
    "Cleveland Browns":       (41.506, -81.700, False),
    "Pittsburgh Steelers":    (40.447, -80.016, False),
    # AFC South
    "Houston Texans":         (29.685, -95.411, True),   # NRG Stadium – retractable
    "Indianapolis Colts":     (39.760, -86.164, True),   # Lucas Oil Stadium
    "Jacksonville Jaguars":   (30.324, -81.638, False),
    "Tennessee Titans":       (36.166, -86.771, False),
    # AFC West
    "Denver Broncos":         (39.744, -105.020, False),
    "Kansas City Chiefs":     (39.049, -94.484, False),
    "Las Vegas Raiders":      (36.091, -115.184, True),  # Allegiant Stadium
    "Los Angeles Chargers":   (33.953, -118.339, False), # SoFi – open-air canopy
    # NFC East
    "Dallas Cowboys":         (32.748, -97.092, True),   # AT&T Stadium – retractable
    "New York Giants":        (40.814, -74.075, False),
    "Philadelphia Eagles":    (39.901, -75.168, False),
    "Washington Commanders":  (38.908, -76.864, False),
    # NFC North
    "Chicago Bears":          (41.862, -87.617, False),
    "Detroit Lions":          (42.340, -83.045, True),   # Ford Field
    "Green Bay Packers":      (44.501, -88.062, False),
    "Minnesota Vikings":      (44.974, -93.258, True),   # U.S. Bank Stadium
    # NFC South
    "Atlanta Falcons":        (33.755, -84.401, True),   # Mercedes-Benz Stadium – retractable
    "Carolina Panthers":      (35.226, -80.853, False),
    "New Orleans Saints":     (29.951, -90.081, True),   # Caesars Superdome
    "Tampa Bay Buccaneers":   (27.976, -82.503, False),
    # NFC West
    "Arizona Cardinals":      (33.528, -112.263, True),  # State Farm Stadium – retractable
    "Los Angeles Rams":       (33.953, -118.339, False), # SoFi – open-air canopy
    "San Francisco 49ers":    (37.403, -121.970, False),
    "Seattle Seahawks":       (47.595, -122.332, False),
}


class WeatherFetcher:
    _FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"

    def __init__(self):
        self.api_key = os.getenv("OPENWEATHER_API_KEY", "")
        self.session = requests.Session()

    def get_game_weather(self, home_team: str, game_time_utc: datetime) -> dict | None:
        """
        Returns a weather dict for an NFL game, or None if unavailable.

        Dict keys:
          is_dome      — True if venue is enclosed (weather irrelevant)
          temp         — temperature in °F
          wind_speed   — wind speed in mph
          description  — short condition string ("light rain", "clear sky", etc.)
          precip_pct   — probability of precipitation 0–100
        """
        stadium = _NFL_STADIUMS.get(home_team)
        if not stadium:
            # Try nickname fallback ("Chiefs" → "Kansas City Chiefs")
            nick = home_team.split()[-1]
            stadium = next(
                (v for k, v in _NFL_STADIUMS.items() if k.split()[-1] == nick),
                None,
            )
        if not stadium:
            logger.debug(f"No stadium data for {home_team}")
            return None

        lat, lon, is_dome = stadium
        if is_dome:
            return {"is_dome": True}

        if not self.api_key:
            logger.debug("OPENWEATHER_API_KEY not set — skipping weather fetch")
            return None

        try:
            r = self.session.get(
                self._FORECAST_URL,
                params={
                    "lat":   lat,
                    "lon":   lon,
                    "appid": self.api_key,
                    "units": "imperial",
                    "cnt":   8,   # 24 hours of 3-hour intervals
                },
                timeout=10,
            )
            r.raise_for_status()
            forecasts = r.json().get("list", [])
            if not forecasts:
                return None

            game_ts = game_time_utc.timestamp()
            closest = min(forecasts, key=lambda f: abs(f["dt"] - game_ts))
            return {
                "is_dome":     False,
                "temp":        round(closest["main"]["temp"]),
                "wind_speed":  round(closest["wind"]["speed"]),
                "description": (closest["weather"][0]["description"]
                                if closest.get("weather") else "unknown"),
                "precip_pct":  round(closest.get("pop", 0) * 100),
            }
        except Exception as e:
            logger.warning(f"Weather fetch failed for {home_team}: {e}")
            return None
