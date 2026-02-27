from ultralytics import YOLO
import os

def download_model():
    models = ["yolo26l.pt", "yolo26l-pose.pt"]
    for model_name in models:
        print(f"Downloading {model_name}...")
        try:
            model = YOLO(model_name)
            print(f"Success! {model_name} is ready.")
        except Exception as e:
            print(f"Error downloading {model_name}: {e}")

if __name__ == "__main__":
    download_model()
