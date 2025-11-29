# src/settings.py

from datetime import datetime

APP_TITLE = "Satellite Time-lapse Studio"
APP_PAGE_ICON = "🛰️"

DEFAULT_START_DATE = datetime(2022, 1, 1)
DEFAULT_END_DATE = datetime(2022, 3, 1)

SENTINEL_COLLECTION = "sentinel-2-l2a"
PLANETARY_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Approximate 10m resolution in degrees at the equator
TARGET_RESOLUTION_DEG = 10.0 / 111_320.0

BAND_OPTIONS = [
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B11", "B12"
]

VISUAL_MODES = [
    "Natural Color (B4, B3, B2)",
    "Color Infrared (B8, B4, B3)",
    "Agriculture (B11, B8, B2)",
    "Healthy Vegetation (B8, B11, B2)",
    "NDVI (B8, B4)",
    "Custom RGB"
]

BAND_COMBINATIONS = {
    "Natural Color (B4, B3, B2)": ["B04", "B03", "B02"],
    "Color Infrared (B8, B4, B3)": ["B08", "B04", "B03"],
    "Agriculture (B11, B8, B2)": ["B11", "B08", "B02"],
    "Healthy Vegetation (B8, B11, B2)": ["B08", "B11", "B02"],
    # NDVI uses B08 and B04, handled specially
}

DEFAULT_FPS = 2
MAX_FPS = 10
