# Design and Development of a Modular Satellite-Imagery Time-Lapse Generator Using Planetary Computer and Streamlit
This project implements a fully modular and extensible system for retrieving, processing, and visualizing multispectral Sentinel-2 satellite imagery. Built with Streamlit, Folium, the Microsoft Planetary Computer STAC API, and modern geospatial libraries such as ODC-STAC, Xarray, and NumPy, the application allows users to interactively define a geographic region, select temporal ranges, control cloud-cover filtering, and choose spectral combinations—including custom band composites and NDVI. The pipeline loads and harmonizes remote-sensing data, generates RGB or NDVI frames, applies optional vegetation-index analytics, and exports a high-quality MP4 time-lapse video. 

![](images/system.png)

## Overview and Background
Generating meaningful visualizations from satellite data requires a complete pipeline for acquiring, filtering, processing, and compositing multispectral imagery into user-friendly products. In this project, geospatial data from the Sentinel-2 L2A collection is retrieved directly from the Microsoft Planetary Computer using its STAC API. The user defines an area of interest interactively by drawing a polygon on a Folium-powered map, and selects a date range, cloud-tolerance settings, and a spectral visualization mode. These parameters are then used to query the STAC catalog and load only the relevant scenes into an optimized Xarray dataset via ODC-STAC.

Each frame of the time-lapse is produced by combining the selected spectral bands, such as natural color (B04, B03, B02), color-infrared (B08, B04, B03), or vegetation-focused composites, and applying robust normalization to produce clear RGB output. For analytical modes like NDVI, the system computes vegetation index values and maps them to a perceptual colormap, producing scientifically meaningful visual transitions over time. Optional analytics compute NDVI statistics per frame, generating an accompanying temporal profile useful for environmental inspection and remote-sensing analysis.

A modular architecture separates the application into clean units: a Streamlit UI for configuration, a Folium map component for spatial input, a STAC query engine, a data-loading and harmonization module, and a dedicated video-generation subsystem. This design ensures smooth integration, low coupling, and high maintainability. The final result is an exported MP4 time-lapse video that illustrates landscape changes, vegetation dynamics, cloud-free mosaics, or seasonal patterns across the selected region—making the tool accessible for environmental monitoring, agricultural studies, research workflows, and educational exploration.

## Table of Contents
```
satellite-timelapse
|__ images
|   |__ system.png
|   |__ banana.png
|__ src
    |__ analytics.py
    |__ app.py
    |__ layout.py
    |__ map_view.py
    |__ settings.py
    |__ stac_search.py
    |__ timelapse.py
.gitignore
README.md
requirements.txt
LICENSE
```

## Getting started

### Resources used
A high-performance Acer Nitro 5 laptop, powered by an Intel Core i7 processor and an NVIDIA GeForce GTX 1650 GPU (4 GB VRAM), was used for model training and evaluation. Due to the large size of the dataset, the training process was computationally demanding and prolonged. Nevertheless, this hardware configuration provided a stable and efficient environment, enabling consistent experimentation and reliable validation of the gesture-recognition models.

### Installing
The project is deployed in a local machine, so you need to install the next software and dependencies to start working:

1. Create and activate the new virtual environment for the project

```bash
conda create --name satellite-timelapse python=3.11
conda activate satellite-timelapse
```

2. Clone repository

```bash
git clone https://github.com/rafamartinezquiles/satellite-timelapse.git
```

3. In the same folder that the requirements are, install the necessary requirements

```bash
cd satellite-timelapse
pip install -r requirements.txt
```