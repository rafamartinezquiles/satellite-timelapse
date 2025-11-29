from typing import List, Optional, Tuple

import planetary_computer
import pystac_client
from odc.stac import stac_load
import xarray as xr

import settings


def query_items(
    bbox: Tuple[float, float, float, float],
    date_range: str,
    max_cloud_cover: Optional[int],
    ignore_cloud_filter: bool,
) -> List:
    client = pystac_client.Client.open(settings.PLANETARY_STAC_URL)

    search_kwargs = {
        "bbox": bbox,
        "datetime": date_range,
        "collections": [settings.SENTINEL_COLLECTION],
    }

    if not ignore_cloud_filter and max_cloud_cover is not None:
        search_kwargs["query"] = {
            "eo:cloud_cover": {"lt": max_cloud_cover}
        }

    search = client.search(**search_kwargs)
    items = list(search.items())
    return items


def load_data_array(
    items: List,
    bbox: Tuple[float, float, float, float],
    bands: List[str],
) -> xr.Dataset:
    if not items:
        raise ValueError("No STAC items to load.")

    signed_items = [planetary_computer.sign(item) for item in items]

    dataset = stac_load(
        signed_items,
        bands=bands,
        crs="EPSG:4326",
        resolution=settings.TARGET_RESOLUTION_DEG,
        bbox=bbox,
        chunks={"x": 2048, "y": 2048},
        patch_url=planetary_computer.sign,
    )

    return dataset
