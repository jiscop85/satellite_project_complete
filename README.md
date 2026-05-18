# Satellite Intelligence Project

A complete, multi-file baseline project for satellite data processing, land-cover classification, and change detection.

## What is included

- Synthetic satellite data generator
- Raster I/O utilities
- Spectral index engineering
- Texture features
- Strong pixel-wise classifier
- Tiled inference for large rasters
- Spectral change detection
- Semantic change detection
- Area summary reports
- Optional Streamlit dashboard

## Why this project works without external data

The project can generate a realistic synthetic dataset that mimics multi-band satellite imagery:
- several land-cover classes
- spatial patterns such as water, vegetation, urban areas, and bare soil
- noisy multi-band reflectance values
- before/after scenes for change detection

That means the whole pipeline runs end-to-end immediately.

## Suggested folder structure

```text
satellite_project/
├── main.py
├── requirements.txt
├── README.md
├── data/
├── models/
├── outputs/
└── src/
    ├── __init__.py
    ├── app_streamlit.py
    ├── change_detection.py
    ├── cli.py
    ├── config.py
    ├── features.py
    ├── inference.py
    ├── indices.py
    ├── io_raster.py
    ├── reporting.py
    ├── synthetic.py
    ├── training.py
    └── visualization.py
```

## Quick start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Generate demo data

```bash
python main.py demo-data
```

### 3) Train the model

```bash
python main.py train --image data/demo/train/image.tif --labels data/demo/train/labels.tif --model-out models/model.joblib
```

### 4) Predict on an image

```bash
python main.py predict --image data/demo/infer/image.tif --model models/model.joblib --out outputs/prediction.tif
```

### 5) Detect change

```bash
python main.py change --before data/demo/change/before.tif --after data/demo/change/after.tif --out outputs/change_map.tif
```

### 6) Launch the dashboard

```bash
streamlit run main.py
```

## Class mapping

- 0 = background / nodata
- 1 = vegetation
- 2 = water
- 3 = built_up
- 4 = bare_soil

## Notes

- The default sensor preset is Sentinel-2-like.
- You can switch presets in the CLI if needed.
- The model bundle stores both the pipeline and configuration.
- The code is intentionally written so it can be split into a larger repository later.
