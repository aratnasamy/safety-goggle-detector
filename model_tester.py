"""
goggles_check.py

Usage:
  python goggles_check.py --image /path/to/face.jpg --model /path/to/best.pt
  python goggles_check.py --image face.jpg --model https://.../best.pt

Optional:
  --goggles-names goggles "safety glasses" eyewear eye_protection
  --conf 0.25
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


DEFAULT_GOGGLES_NAMES = [
    "goggles",
    "safety goggles",
    "safety glasses",
    "safety-glasses",
    "eye protection",
    "eye_protection",
    "eyewear",
    "protective eyewear",
    "protective_eyewear",
]


def norm(s: str) -> str:
    return s.strip().lower().replace("-", " ").replace("_", " ")


def parse_args():
    p = argparse.ArgumentParser(description="Detect safety goggles in a face image using YOLO.")
    p.add_argument("--image", required=True, help="Path to an input image (face photo).")
    p.add_argument("--model", required=True, help="Path/URL to a YOLO .pt model (Ultralytics).")
    p.add_argument("--conf", type=float, default=0.6, help="Confidence threshold for detections.")
    p.add_argument(
        "--goggles-names",
        nargs="*",
        default=DEFAULT_GOGGLES_NAMES,
        help="Class name(s) that should count as goggles/eye protection.",
    )
    p.add_argument("--verbose", action="store_true", help="Print detections and class map.")
    return p.parse_args()


def main():
    args = parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        sys.exit(2)

    # Load model (Ultralytics YOLO)
    model = YOLO(args.model)

    # Run inference on a single image
    results = model.predict(
        source=str(image_path),
        conf=args.conf,
        verbose=False,
    )

    if not results:
        print("NO")  # no result object (unexpected but safe)
        sys.exit(0)

    r = results[0]

    # Map class ids -> names (Ultralytics stores this on the model)
    names_map = getattr(model, "names", None) or {}
    goggles_name_set = {norm(x) for x in args.goggles_names}

    found = False
    found_best = None  # (name, conf, cls_id, box_xyxy)

    if r.boxes is not None and len(r.boxes) > 0:
        for b in r.boxes:
            cls_id = int(b.cls.item())
            conf = float(b.conf.item())
            class_name = names_map.get(cls_id, str(cls_id))
            if norm(class_name) in goggles_name_set:
                found = True
                xyxy = [float(x) for x in b.xyxy[0].tolist()]
                if found_best is None or conf > found_best[1]:
                    found_best = (class_name, conf, cls_id, xyxy)

    if args.verbose:
        print("Model classes:", names_map)
        if r.boxes is None or len(r.boxes) == 0:
            print("Detections: (none)")
        else:
            dets = []
            for b in r.boxes:
                cls_id = int(b.cls.item())
                conf = float(b.conf.item())
                dets.append((names_map.get(cls_id, str(cls_id)), conf))
            print("Detections:", dets)
        if found_best:
            print("Best goggles match:", found_best)

    print("YES" if found else "NO")


if __name__ == "__main__":
    main()
