from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .app_streamlit import run_app
from .change_detection import semantic_change_detection, spectral_change_detection
from .config import ProjectConfig
from .inference import predict_tiled
from .reporting import summarize_prediction_map
from .synthetic import SyntheticSceneSpec, create_demo_dataset
from .training import train_model

LOGGER = logging.getLogger("satellite_project")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Satellite Intelligence Project",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    p_demo = sub.add_parser("demo-data", help="Generate synthetic demo data")
    p_demo.add_argument("--out", default="data/demo")
    p_demo.add_argument("--width", type=int, default=512)
    p_demo.add_argument("--height", type=int, default=512)
    p_demo.add_argument("--bands", type=int, default=12)
    p_demo.add_argument("--seed", type=int, default=42)

    p_train = sub.add_parser("train", help="Train classifier")
    p_train.add_argument("--image", required=True)
    p_train.add_argument("--labels", required=True)
    p_train.add_argument("--model-out", default="models/model.joblib")
    p_train.add_argument("--metrics-out", default="outputs/train_metrics.json")
    p_train.add_argument("--sensor", default="sentinel2", choices=["sentinel2", "landsat8", "landsat9", "generic"])

    p_predict = sub.add_parser("predict", help="Predict class map")
    p_predict.add_argument("--image", required=True)
    p_predict.add_argument("--model", required=True)
    p_predict.add_argument("--out", default="outputs/prediction.tif")
    p_predict.add_argument("--proba-out", default="")
    p_predict.add_argument("--no-smooth", action="store_true")

    p_change = sub.add_parser("change", help="Spectral change detection")
    p_change.add_argument("--before", required=True)
    p_change.add_argument("--after", required=True)
    p_change.add_argument("--out", default="outputs/change_map.tif")
    p_change.add_argument("--sensor", default="sentinel2", choices=["sentinel2", "landsat8", "landsat9", "generic"])

    p_sem = sub.add_parser("semantic-change", help="Semantic change detection")
    p_sem.add_argument("--before", required=True)
    p_sem.add_argument("--after", required=True)
    p_sem.add_argument("--model", required=True)
    p_sem.add_argument("--out", default="outputs/semantic_change.tif")

    p_sum = sub.add_parser("summarize", help="Summarize prediction areas")
    p_sum.add_argument("--prediction", required=True)
    p_sum.add_argument("--out", default="outputs/area_stats.csv")

    sub.add_parser("app", help="Run Streamlit dashboard")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "demo-data":
        spec = SyntheticSceneSpec(width=args.width, height=args.height, bands=args.bands, seed=args.seed)
        paths = create_demo_dataset(args.out, spec)
        LOGGER.info("Demo data created: %s", paths)

    elif args.command == "train":
        cfg = ProjectConfig(sensor=args.sensor)
        bundle = train_model(args.image, args.labels, args.model_out, args.metrics_out, cfg)
        LOGGER.info("Training complete. Metrics: %s", bundle.metrics)

    elif args.command == "predict":
        predict_tiled(args.image, args.model, args.out, proba_out=args.proba_out or None, apply_smoothing=not args.no_smooth)

    elif args.command == "change":
        cfg = ProjectConfig(sensor=args.sensor)
        spectral_change_detection(args.before, args.after, args.out, cfg)

    elif args.command == "semantic-change":
        result = semantic_change_detection(args.before, args.after, args.model, args.out)
        LOGGER.info("Semantic change result: %s", result)

    elif args.command == "summarize":
        rows = summarize_prediction_map(args.prediction, args.out)
        LOGGER.info("Area summary: %s", rows)

    elif args.command == "app":
        run_app()


if __name__ == "__main__":
    main()
