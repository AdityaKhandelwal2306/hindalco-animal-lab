"""Shared model, preprocessing, saving, and prediction helpers."""
import json
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_model(num_classes, pretrained=True, freeze=True):
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    if freeze:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    return model


def save_model(model, classes, model_dir):
    path = Path(model_dir)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    (path / "classes.json").write_text(json.dumps(list(classes), indent=2), encoding="utf-8")


def load_model(model_dir):
    root = Path(model_dir)
    model_file = next(root.rglob("model.pt"), None)
    classes_file = next(root.rglob("classes.json"), None)
    if model_file is None or classes_file is None:
        raise FileNotFoundError(f"Could not find model.pt and classes.json below {root}")
    classes = json.loads(classes_file.read_text(encoding="utf-8"))
    model = build_model(len(classes), pretrained=False, freeze=False)
    model.load_state_dict(torch.load(model_file, map_location="cpu"))
    model.eval()
    return model, classes


def predict_image(model, classes, pil_image):
    image = pil_image.convert("RGB")
    with torch.no_grad():
        scores = torch.softmax(model(get_transform()(image).unsqueeze(0)), dim=1)[0]
    index = int(torch.argmax(scores))
    return {"animal": classes[index], "confidence": float(scores[index]),
            "all_scores": {name: float(value) for name, value in zip(classes, scores)}}


def log_metric(name, value, step=None):
    uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if uri.startswith("azureml"):
        import mlflow
        mlflow.log_metric(name, value, step=step)