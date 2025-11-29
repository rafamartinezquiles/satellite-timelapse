import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Streamlit

import streamlit as st

from datetime import datetime

from . import layout
from . import map_view
from . import stac_search
from . import timelapse
from . import analytics
from . import settings


def main() -> None:
    layout.configure_page()
    layout.render_header()

    controls = layout.sidebar_controls()

    # --- Map & AOI ---
    left_col, right_col = layout.two_column_layout()

    with left_col:
        map_state = map_view.create_map()

    bbox = map_view.extract_bbox(map_state)

    with right_col:
        st.subheader("Status")
        if bbox is None:
            st.info("Draw a polygon on the map to define the area of interest.")
            return

        st.success("Area of interest detected.")
        st.write(
            f"**Bounding box** (lon/lat): "
            f"{bbox[0]:.4f}, {bbox[1]:.4f} → {bbox[2]:.4f}, {bbox[3]:.4f}"
        )

        st.markdown("---")
        if st.button("Generate time-lapse", use_container_width=True):
            run_pipeline(bbox, controls)


def run_pipeline(bbox, controls) -> None:
    start = controls["start_date"]
    end = controls["end_date"]

    if start > end:
        st.error("Start date must be earlier than end date.")
        return

    date_range = f"{start.isoformat()}/{end.isoformat()}"
    st.write(f"**Date range**: {date_range}")

    # --- STAC query ---
    with st.spinner("Querying Sentinel-2 catalog..."):
        items = stac_search.query_items(
            bbox=bbox,
            date_range=date_range,
            max_cloud_cover=controls["max_cloud_cover"],
            ignore_cloud_filter=controls["ignore_cloud_filter"],
        )

    if not items:
        st.error("No matching scenes found. Try expanding the date range or changing the cloud tolerance.")
        return

    st.success(f"Found {len(items)} scenes.")

    # --- Data load ---
    visual_mode = controls["mode"]
    custom_bands = controls["custom_bands"]

    is_ndvi_mode = visual_mode == "NDVI (B8, B4)"

    if is_ndvi_mode:
        bands_to_load = ["B04", "B08"]
    elif visual_mode == "Custom RGB" and custom_bands:
        bands_to_load = list(custom_bands)
    else:
        bands_to_load = settings.BAND_COMBINATIONS[visual_mode]

    st.markdown("---")
    st.subheader("Data loading")

    with st.spinner("Loading imagery from Planetary Computer..."):
        dataset = stac_search.load_data_array(
            items=items,
            bbox=bbox,
            bands=bands_to_load,
        )

    st.success("Imagery loaded.")

    # --- Preview / CV ---
    if controls["apply_cv"]:
        try:
            ndvi_cube = analytics.compute_ndvi_cube(dataset)
            analytics.summarize_ndvi_per_frame(ndvi_cube)
        except Exception as exc:
            st.warning(f"NDVI analytics unavailable: {exc}")

    st.markdown("---")
    st.subheader("Time-lapse preview")

    # Preview first frame quickly
    if is_ndvi_mode:
        try:
            ndvi_cube = analytics.compute_ndvi_cube(dataset)
            frames, timestamps = timelapse.render_frames_ndvi(ndvi_cube, mode_label="NDVI")
        except Exception as exc:
            st.error(f"Could not render NDVI frames: {exc}")
            return
    else:
        rgb_stack = timelapse.prepare_rgb_stack(dataset, bands_to_load)
        try:
            frames, timestamps = timelapse.render_frames_rgb(rgb_stack, mode_label=visual_mode)
        except Exception as exc:
            st.error(f"Could not render RGB frames: {exc}")
            return

    # --- Video export ---
    fps = controls["fps"]
    st.info(f"Rendering video at {fps} fps with {len(frames)} frames.")

    with st.spinner("Exporting MP4 time-lapse..."):
        video_path = timelapse.export_video(frames, fps=fps)

    st.success("Time-lapse ready.")

    st.video(video_path)

    with open(video_path, "rb") as f:
        st.download_button(
            label="Download MP4",
            data=f,
            file_name="sentinel_timelapse.mp4",
            mime="video/mp4",
        )


if __name__ == "__main__":
    main()
