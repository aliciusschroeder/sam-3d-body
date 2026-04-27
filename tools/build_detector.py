# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
from PIL import Image


def _scale_boxes(boxes, scale):
    enlarged_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = (x2 - x1) * scale
        h = (y2 - y1) * scale
        new_x1 = max(cx - w / 2, 0)
        new_y1 = max(cy - h / 2, 0)
        new_x2 = cx + w / 2
        new_y2 = cy + h / 2
        enlarged_boxes.append([new_x1, new_y1, new_x2, new_y2])
    return enlarged_boxes


class HumanDetector:
    def __init__(self, name="yolo26x", device="cuda", **kwargs):
        self.device = device

        if name == "yolo26x":
            from ultralytics import YOLO

            print("########### Using human detector: YOLO26X...")
            model_path = kwargs.get("path", "yolo26x.pt")
            self.detector = YOLO(model_path)
            self.detector_func = lambda detector, img, **kwargs: self.yolo_run(
                img, **kwargs
            )
        elif name == "sam3":
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model

            self.detector = build_sam3_image_model()
            self.processor = Sam3Processor(self.detector)
            self.detector_func = lambda detector, img, **kwargs: self.sam3_run(
                img, **kwargs
            )
        else:
            raise NotImplementedError

    def yolo_run(
        self,
        img,
        det_cat_id: int = 0,
        bbox_thr: float = 0.5,
        scale_bbox: float = 1.2,
        **kwargs,
    ):
        results = self.detector.predict(source=img, device=self.device, verbose=False)
        result = results[0].boxes
        if result is None:
            raise RuntimeError
        result = result.cpu().numpy()

        boxes = result.xyxy
        classes = result.cls
        scores = result.conf

        valid_idx = (classes == det_cat_id) & (scores > bbox_thr)
        boxes = boxes[valid_idx]

        enlarged_boxes = _scale_boxes(boxes, scale_bbox)
        return np.array(enlarged_boxes)

    def sam3_run(
        self,
        img,
        det_cat_id: int = 0,
        bbox_thr: float = 0.5,
        scale_bbox: float = 1.2,
        **kwargs,
    ):
        # switch bgr to rgb
        img = img[:, :, ::-1].copy()
        img = Image.fromarray(img.astype("uint8"), "RGB")
        inference_state = self.processor.set_image(img)
        # Prompt the model with text
        output = self.processor.set_text_prompt(state=inference_state, prompt="person")

        # Get the masks, bounding boxes, and scores
        _masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

        confident_idx = scores > bbox_thr
        boxes = boxes[confident_idx].cpu().numpy()

        enlarged_boxes = _scale_boxes(boxes, scale_bbox)
        return np.array(enlarged_boxes)

    def run_human_detection(self, img, **kwargs):
        return self.detector_func(self.detector, img, **kwargs)
