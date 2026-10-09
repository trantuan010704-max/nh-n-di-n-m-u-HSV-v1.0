import cv2
import numpy as np
from config import MORPH_KERNEL_SIZE, MIN_AREA, MAX_AREA

def morphology_process(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return opening, closing

def connected_components_filter(mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    objects = []
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if not MIN_AREA <= area <= MAX_AREA:
            continue
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[label]
        filtered[labels == label] = 255
        objects.append({"label": label, "area": area, "x": x, "y": y, "w": w, "h": h, "cx": cx, "cy": cy})
    return filtered, objects

def find_object_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if MIN_AREA <= cv2.contourArea(c) <= MAX_AREA]
