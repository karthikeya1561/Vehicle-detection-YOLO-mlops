from zenml import step
from ultralytics import YOLO
from models.training_modeldev import yolo_detector
import mlflow

from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker
tracker_name = experiment_tracker.name if experiment_tracker else None

@step(experiment_tracker=tracker_name)
def trained_model(data_path: str) -> YOLO:
     if tracker_name:
         mlflow.sklearn.autolog()
     obj = yolo_detector()
     model = obj.train(data_path = data_path)
     return model