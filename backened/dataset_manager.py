"""
backend/dataset_manager.py

Detects whether the permitted research datasets referenced in the SIH PPT
are physically present on disk, WITHOUT fabricating any statistics.

Permitted datasets ONLY:
    - AI4Shipwrecks   -> data/ai4shipwrecks/
    - GhostVision (Ghost Pot) -> data/ghostvision/
    - SubPipe          -> data/subpipe/

No other dataset (Kaggle, COCO, ImageNet, random GitHub sonar sets, etc.)
is referenced anywhere in this application.
"""

from __future__ import annotations

from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
ANNOTATION_EXTS = {".json", ".xml", ".txt", ".csv"}

DATASETS = [
    {
        "key": "ai4shipwrecks",
        "name": "AI4Shipwrecks",
        "purpose": "Primary sonar baseline (shipwreck detection/segmentation)",
        "type": "Side-Scan Sonar imagery + segmentation annotations",
        "source": "University of Michigan / Field Robotics Group (research reference)",
        "role": "Primary",
    },
    {
        "key": "ghostvision",
        "name": "Ghost Pot / GhostVision",
        "purpose": "Derelict fishing gear / crab-pot validation",
        "type": "Side-Scan Sonar imagery",
        "source": "Research reference dataset",
        "role": "Secondary",
    },
    {
        "key": "subpipe",
        "name": "SubPipe",
        "purpose": "Submarine pipeline / cable inspection validation",
        "type": "Side-Scan Sonar imagery",
        "source": "Research reference dataset",
        "role": "Secondary",
    },
]


def _scan_folder(folder: Path) -> dict:
    if not folder.exists():
        return {"installed": False, "num_images": 0, "num_annotations": 0}
    images = [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    annots = [p for p in folder.rglob("*") if p.suffix.lower() in ANNOTATION_EXTS]
    return {
        "installed": len(images) > 0,
        "num_images": len(images),
        "num_annotations": len(annots),
    }


def get_dataset_status() -> list[dict]:
    status = []
    for ds in DATASETS:
        folder = DATA_ROOT / ds["key"]
        scan = _scan_folder(folder)
        entry = dict(ds)
        entry["local_path"] = str(folder)
        entry.update(scan)
        entry["status_label"] = (
            f"Installed - {scan['num_images']} image(s) found"
            if scan["installed"]
            else "Research dataset not installed."
        )
        status.append(entry)
    return status
