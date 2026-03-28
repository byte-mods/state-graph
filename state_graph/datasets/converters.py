"""Format converters — YOLO↔COCO, CSV↔JSONL, Alpaca↔ShareGPT, etc."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def csv_to_jsonl(csv_path: str, output_path: str | None = None) -> dict:
    """Convert CSV to JSONL."""
    p = Path(csv_path)
    out = Path(output_path) if output_path else p.with_suffix(".jsonl")

    rows = 0
    with open(p, "r", encoding="utf-8") as fin, open(out, "w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        for row in reader:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1

    return {"status": "converted", "format": "jsonl", "path": str(out), "rows": rows}


def jsonl_to_csv(jsonl_path: str, output_path: str | None = None) -> dict:
    """Convert JSONL to CSV."""
    p = Path(jsonl_path)
    out = Path(output_path) if output_path else p.with_suffix(".csv")

    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        return {"status": "error", "message": "Empty file"}

    fields = list(rows[0].keys())
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})

    return {"status": "converted", "format": "csv", "path": str(out), "rows": len(rows)}


def yolo_to_coco(
    images_dir: str,
    labels_dir: str,
    classes: list[str],
    output_path: str | None = None,
) -> dict:
    """Convert YOLO format to COCO JSON."""
    img_dir = Path(images_dir)
    lbl_dir = Path(labels_dir)
    out = Path(output_path) if output_path else lbl_dir.parent / "coco_annotations.json"

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(classes)],
    }

    ann_id = 0
    for img_id, img_file in enumerate(sorted(img_dir.glob("*"))):
        if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue

        # Try to get image dimensions
        w, h = 0, 0
        try:
            from PIL import Image
            with Image.open(img_file) as im:
                w, h = im.size
        except Exception:
            pass

        coco["images"].append({
            "id": img_id,
            "file_name": img_file.name,
            "width": w,
            "height": h,
        })

        # Read YOLO labels
        lbl_file = lbl_dir / (img_file.stem + ".txt")
        if not lbl_file.exists():
            continue

        for line in lbl_file.read_text().strip().split("\n"):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            # Convert YOLO (center, w, h normalized) to COCO (x, y, w, h pixels)
            x = (xc - bw / 2) * w
            y = (yc - bh / 2) * h
            box_w = bw * w
            box_h = bh * h

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id,
                "bbox": [round(x, 1), round(y, 1), round(box_w, 1), round(box_h, 1)],
                "area": round(box_w * box_h, 1),
                "iscrowd": 0,
            })
            ann_id += 1

    out.write_text(json.dumps(coco, indent=2))
    return {
        "status": "converted",
        "format": "coco",
        "path": str(out),
        "images": len(coco["images"]),
        "annotations": len(coco["annotations"]),
    }


def coco_to_yolo(
    coco_json_path: str,
    output_dir: str | None = None,
) -> dict:
    """Convert COCO JSON to YOLO format."""
    p = Path(coco_json_path)
    out_dir = Path(output_dir) if output_dir else p.parent / "yolo_labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = json.loads(p.read_text())
    images = {img["id"]: img for img in coco["images"]}
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

    # Group annotations by image
    by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        by_image.setdefault(img_id, []).append(ann)

    file_count = 0
    for img_id, img_info in images.items():
        w, h = img_info.get("width", 1), img_info.get("height", 1)
        if w == 0 or h == 0:
            continue

        anns = by_image.get(img_id, [])
        fname = Path(img_info["file_name"]).stem + ".txt"

        lines = []
        for ann in anns:
            bx, by_, bw, bh = ann["bbox"]
            # COCO (x, y, w, h) pixels → YOLO (center_x, center_y, w, h) normalized
            xc = (bx + bw / 2) / w
            yc = (by_ + bh / 2) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"{ann['category_id']} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        (out_dir / fname).write_text("\n".join(lines))
        file_count += 1

    # Write classes.txt
    classes = [categories[i] for i in sorted(categories.keys())]
    (out_dir / "classes.txt").write_text("\n".join(classes))

    return {
        "status": "converted",
        "format": "yolo",
        "path": str(out_dir),
        "files": file_count,
    }


def alpaca_to_sharegpt(alpaca_path: str, output_path: str | None = None) -> dict:
    """Convert Alpaca format to ShareGPT format."""
    p = Path(alpaca_path)
    out = Path(output_path) if output_path else p.with_name(p.stem + "_sharegpt.json")

    data = json.loads(p.read_text())
    sharegpt = []
    for item in data:
        user_msg = item.get("instruction", "")
        if item.get("input"):
            user_msg += f"\n\n{item['input']}"
        sharegpt.append({
            "conversations": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": item.get("output", "")},
            ]
        })

    out.write_text(json.dumps(sharegpt, indent=2, ensure_ascii=False))
    return {"status": "converted", "format": "sharegpt", "path": str(out), "count": len(sharegpt)}


def sharegpt_to_alpaca(sharegpt_path: str, output_path: str | None = None) -> dict:
    """Convert ShareGPT format to Alpaca format."""
    p = Path(sharegpt_path)
    out = Path(output_path) if output_path else p.with_name(p.stem + "_alpaca.json")

    data = json.loads(p.read_text())
    alpaca = []
    for item in data:
        convs = item.get("conversations", [])
        if len(convs) >= 2:
            alpaca.append({
                "instruction": convs[0].get("content", ""),
                "input": "",
                "output": convs[1].get("content", ""),
            })

    out.write_text(json.dumps(alpaca, indent=2, ensure_ascii=False))
    return {"status": "converted", "format": "alpaca", "path": str(out), "count": len(alpaca)}
