"""Connected component analysis and bounding box extraction."""
from typing import List
import numpy as np
from app.schemas.common import BoundingBox, Confidence

try:
    from skimage.measure import label, regionprops
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


def extract_bounding_boxes(
    mask: np.ndarray,
    min_area_px: int = 25,
    max_boxes: int = 20,
    base_confidence: float = 0.85
) -> List[BoundingBox]:
    """
    Identifies connected clusters of changed pixels and returns bounding boxes.
    """
    boxes: List[BoundingBox] = []
    if np.count_nonzero(mask) == 0:
        return boxes

    if HAS_SKIMAGE:
        labeled_mask = label(mask)
        props = regionprops(labeled_mask)

        # Sort by area descending
        props = sorted(props, key=lambda p: p.area, reverse=True)

        for p in props[:max_boxes]:
            if p.area < min_area_px:
                continue

            minr, minc, maxr, maxc = p.bbox
            width = maxc - minc
            height = maxr - minr

            # Scale confidence slightly by area prominence
            area_weight = min(float(p.area) / 1000.0, 0.15)
            box_conf = min(base_confidence + area_weight, 0.98)

            boxes.append(BoundingBox(
                x=int(minc),
                y=int(minr),
                width=int(width),
                height=int(height),
                label="surface_change",
                confidence=Confidence(
                    score=round(box_conf, 2),
                    method="connected_component_area",
                    calibrated=False
                )
            ))
    else:
        # Fallback bounding box around all non-zero pixels
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if np.any(rows) and np.any(cols):
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            boxes.append(BoundingBox(
                x=int(xmin),
                y=int(ymin),
                width=int(xmax - xmin + 1),
                height=int(ymax - ymin + 1),
                label="change_region",
                confidence=Confidence(
                    score=round(base_confidence, 2),
                    method="extent_bbox",
                    calibrated=False
                )
            ))

    return boxes
