#!/usr/bin/env python3
"""Load one batch from each split with the pinned YOLOv5 dataloader."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_yaml", type=Path)
    parser.add_argument("--yolov5", type=Path, default=Path(__file__).parents[1] / "yolov5")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    yolov5 = args.yolov5.resolve()
    sys.path.insert(0, str(yolov5))
    from utils.dataloaders import create_dataloader  # noqa: PLC0415
    from utils.general import check_dataset  # noqa: PLC0415

    data = check_dataset(str(args.data_yaml.resolve()), autodownload=False)
    result: dict[str, object] = {
        "status": "passed",
        "data_yaml": str(args.data_yaml.resolve()),
        "names": data["names"],
        "nc": data["nc"],
        "splits": {},
    }
    for split, key in (("train", "train"), ("valid", "val"), ("test", "test")):
        loader, dataset = create_dataloader(
            data[key],
            imgsz=args.img_size,
            batch_size=args.batch_size,
            stride=32,
            augment=False,
            cache=False,
            rect=False,
            workers=0,
            shuffle=False,
            prefix=f"sanity-{split}: ",
            seed=0,
        )
        images, targets, paths, shapes = next(iter(loader))
        class_ids = sorted({int(value) for value in targets[:, 1].tolist()}) if len(targets) else []
        if images.ndim != 4 or images.shape[1] != 3 or images.shape[2:] != (args.img_size, args.img_size):
            raise SystemExit(f"Unexpected image batch shape for {split}: {tuple(images.shape)}")
        if any(class_id < 0 or class_id >= data["nc"] for class_id in class_ids):
            raise SystemExit(f"Class ID outside 0..{data['nc'] - 1} in {split}: {class_ids}")
        result["splits"][split] = {
            "dataset_images": len(dataset),
            "batch_shape": list(images.shape),
            "batch_targets": int(len(targets)),
            "batch_class_ids": class_ids,
            "sample_paths": [str(Path(path).resolve()) for path in paths],
            "shapes_metadata_present": shapes is not None,
        }

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
