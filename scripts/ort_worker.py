import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def clip_box(left: int, top: int, right: int, bottom: int, width: int, height: int) -> list[int]:
    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(0, min(right, width - 1))
    bottom = max(0, min(bottom, height - 1))
    return [left, top, right, bottom]


class PackageSession:
    def __init__(self, package_dir: Path, conf_threshold: float, nms_threshold: float) -> None:
        self.package_dir = package_dir.resolve()
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        manifest_path = self.package_dir / "model-manifest.json"
        labels_path = self.package_dir / "labels.txt"

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        model_name = manifest["files"]["model"]
        self.model_path = self.package_dir / model_name
        self.input_name = manifest["preprocessing"]["input_tensor_name"]
        self.output_name = manifest["output_format"]["raw_output_name"]
        self.input_width = int(manifest["preprocessing"]["input_size"][0])
        self.input_height = int(manifest["preprocessing"]["input_size"][1])
        expected_classes = int(manifest["model"]["class_count"])

        if len(self.labels) != expected_classes:
            raise RuntimeError("labels.txt sinif sayisi manifest ile uyusmuyor.")

        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

        ort_input = self.session.get_inputs()[0]
        ort_output = self.session.get_outputs()[0]

        if ort_input.name != self.input_name:
            raise RuntimeError(f"Manifest input adi uyusmuyor: {ort_input.name} != {self.input_name}")
        if ort_output.name != self.output_name:
            raise RuntimeError(f"Manifest output adi uyusmuyor: {ort_output.name} != {self.output_name}")

    def infer_frame(self, frame: np.ndarray) -> dict:
        preprocess_start = time.perf_counter()
        original_height, original_width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        tensor = resized.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

        inference_start = time.perf_counter()
        output = self.session.run([self.output_name], {self.input_name: tensor})[0]
        inference_ms = (time.perf_counter() - inference_start) * 1000.0

        postprocess_start = time.perf_counter()
        rows = np.transpose(output[0], (1, 0))
        x_factor = original_width / float(self.input_width)
        y_factor = original_height / float(self.input_height)

        boxes: list[list[int]] = []
        confidences: list[float] = []
        class_ids: list[int] = []

        for row in rows:
            scores = row[4:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence < self.conf_threshold:
                continue

            cx, cy, width, height = row[:4]
            left = int((cx - 0.5 * width) * x_factor)
            top = int((cy - 0.5 * height) * y_factor)
            box_width = int(width * x_factor)
            box_height = int(height * y_factor)

            if box_width <= 1 or box_height <= 1:
                continue

            boxes.append([left, top, box_width, box_height])
            confidences.append(confidence)
            class_ids.append(class_id)

        detections: list[dict] = []
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
            if len(indices) > 0:
                flat_indices = np.array(indices).reshape(-1).tolist()
                for index in flat_indices:
                    left, top, box_width, box_height = boxes[index]
                    bbox = clip_box(
                        left,
                        top,
                        left + box_width,
                        top + box_height,
                        original_width,
                        original_height,
                    )
                    detections.append(
                        {
                            "class_id": int(class_ids[index]),
                            "label": self.labels[class_ids[index]],
                            "confidence": round(float(confidences[index]), 6),
                            "bbox_xyxy": bbox,
                        }
                    )

        postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0
        worker_total_ms = preprocess_ms + inference_ms + postprocess_ms

        return {
            "detections": detections,
            "timings_ms": {
                "preprocess": round(preprocess_ms, 3),
                "inference": round(inference_ms, 3),
                "postprocess": round(postprocess_ms, 3),
                "worker_total": round(worker_total_ms, 3),
            },
        }


def write_line(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--conf-threshold", type=float, default=0.15)
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    args = parser.parse_args()

    session = PackageSession(Path(args.package), args.conf_threshold, args.nms_threshold)
    sys.stdout.write("READY\n")
    sys.stdout.flush()

    stdin = sys.stdin.buffer

    while True:
        header = stdin.readline()
        if not header:
            break

        command = header.decode("utf-8", errors="replace").strip()
        if command == "QUIT":
            break

        if not command.startswith("FRAME "):
            write_line({"detections": [], "error": f"Bilinmeyen komut: {command}"})
            continue

        try:
            payload_size = int(command.split(" ", 1)[1])
        except ValueError:
            write_line({"detections": [], "error": f"Gecersiz boyut: {command}"})
            continue

        payload = stdin.read(payload_size)
        if len(payload) != payload_size:
            write_line({"detections": [], "error": "Eksik frame verisi alindi."})
            continue

        try:
            worker_total_start = time.perf_counter()
            encoded = np.frombuffer(payload, dtype=np.uint8)
            decode_start = time.perf_counter()
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            decode_ms = (time.perf_counter() - decode_start) * 1000.0
            if frame is None:
                raise RuntimeError("JPEG decode basarisiz.")

            response = session.infer_frame(frame)
            response.setdefault("timings_ms", {})
            response["timings_ms"]["decode"] = round(decode_ms, 3)
            response["timings_ms"]["worker_total"] = round((time.perf_counter() - worker_total_start) * 1000.0, 3)
            write_line(response)
        except Exception as exc:
            write_line({"detections": [], "error": str(exc)})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
