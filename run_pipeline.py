import argparse
import os
import sys

# Fix for Windows Microsoft Store Python - ZenML path junction bug
# AppData\LocalCache\Roaming is a junction of AppData\Roaming; ZenML's
# _validate_path uses Path.resolve() which traverses the junction and
# produces a path that no longer starts with the artifact store root.
_roaming = os.path.join(os.environ.get("APPDATA", ""), "zenml", "local_stores")
os.environ.setdefault("ZENML_LOCAL_STORES_PATH", _roaming)

from datetime import datetime
from pathlib import Path
from pipelines.training_pipeline import pipeline_dev
from zenml.client import Client


OUTPUT_DIR = Path("outputs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle Detection - YOLO Pipeline")
    parser.add_argument(
        "--image",
        type=str,
        default="1.jpg",
    )
    args = parser.parse_args()

    # Auto-create outputs/ folder
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Build a timestamped output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = str(OUTPUT_DIR / f"output_{timestamp}.jpg")

    _et = Client().active_stack.experiment_tracker
    if _et:
        print(_et.get_tracking_uri())
    else:
        print("No experiment tracker configured — running without MLflow.")
    print(f"Running detection on : {args.image}")
    print(f"Output will be saved : {output_path}")

    pipeline_dev(
        data_path="traffic-project-2/data.yaml",
        model_path="pt_model/yolov8n.pt",
        image_path=args.image,
        output_path=output_path
    )

    print(f"\n✅ Done! Result saved to: {output_path}")
