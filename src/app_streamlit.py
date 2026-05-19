from __future__ import annotations

from pathlib import Path

import streamlit as st

from .change_detection import semantic_change_detection, spectral_change_detection
from .config import ProjectConfig
from .inference import predict_tiled
from .reporting import summarize_prediction_map
from .synthetic import create_demo_dataset
from .training import train_model


def run_app():
    st.set_page_config(page_title="Satellite Intelligence", layout="wide")
    st.title("Satellite Intelligence Project")
    st.caption("An end-to-end satellite processing dashboard.")

    cfg = ProjectConfig()
    mode = st.sidebar.selectbox("Mode", ["Create Demo Data", "Train", "Predict", "Change", "Semantic Change", "Summarize"])
    model_path = st.sidebar.text_input("Model bundle", "models/model.joblib")

    if mode == "Create Demo Data":
        st.subheader("Demo data generator")
        if st.button("Create demo dataset"):
            paths = create_demo_dataset()
            st.success("Demo dataset created")
            st.json(paths)

    elif mode == "Train":
        st.subheader("Training")
        img_file = st.file_uploader("Training image", type=["tif", "tiff"], key="train_img")
        lbl_file = st.file_uploader("Label raster", type=["tif", "tiff"], key="train_lbl")
        model_out = st.text_input("Model output", "models/model.joblib")
        metrics_out = st.text_input("Metrics output", "outputs/train_metrics.json")
        if st.button("Start training"):
            if img_file is None or lbl_file is None:
                st.error("Upload both files.")
            else:
                tmp = Path(".tmp_uploads")
                tmp.mkdir(parents=True, exist_ok=True)
                img_path = tmp / "train_image.tif"
                lbl_path = tmp / "train_labels.tif"
                img_path.write_bytes(img_file.getbuffer())
                lbl_path.write_bytes(lbl_file.getbuffer())
                with st.spinner("Training..."):
                    bundle = train_model(img_path, lbl_path, model_out, metrics_out, cfg)
                st.success("Training completed")
                st.json(bundle.metrics)

    elif mode == "Predict":
        st.subheader("Prediction")
        img_file = st.file_uploader("Input image", type=["tif", "tiff"], key="pred_img")
        out_path = st.text_input("Prediction output", "outputs/prediction.tif")
        proba_out = st.text_input("Confidence output (optional)", "")
        if st.button("Run prediction"):
            if img_file is None:
                st.error("Upload an image.")
            else:
                tmp = Path(".tmp_uploads")
                tmp.mkdir(parents=True, exist_ok=True)
                img_path = tmp / "infer_image.tif"
                img_path.write_bytes(img_file.getbuffer())
                with st.spinner("Predicting..."):
                    pred, _ = predict_tiled(img_path, model_path, out_path, proba_out=proba_out or None)
                st.success("Prediction completed")
                st.image(pred, caption="Prediction map", use_container_width=True)
