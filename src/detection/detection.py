from pathlib import Path
from typing import Union

from src.detection.yolov8_model import load_model


class PersonDetector:
    """Person detector using the trained Milestone 1 model."""

    def __init__(
        self,
        model_path: Union[str, Path] = "models/detection/best.pt",
        confidence: float = 0.30,
        image_size: int = 640,
        device: str = "cpu",
    ):
        self.model = load_model(str(model_path))
        self.confidence = confidence
        self.image_size = image_size
        self.device = device

    def detect(self, source):
        """
        Detect persons in an image or video frame.

        Returns:
            List of dictionaries containing:
            bbox, confidence, class_id
        """

        results = self.model.predict(
            source=source,
            imgsz=self.image_size,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for i in range(len(result.boxes)):
                class_id = int(result.boxes.cls[i].item())

                # Our dataset contains only one class: person
                if class_id != 0:
                    continue

                confidence = float(result.boxes.conf[i].item())
                x1, y1, x2, y2 = result.boxes.xyxy[i].tolist()

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "class_id": class_id,
                })

        return detections