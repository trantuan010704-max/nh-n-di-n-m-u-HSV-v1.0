import cv2
import numpy as np
import math

def extract_shape_features(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4.0 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h) if h > 0 else 0.0
    bbox_area = w * h
    extent = area / float(bbox_area) if bbox_area > 0 else 0.0
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / float(hull_area) if hull_area > 0 else 0.0
    return {"area": area, "perimeter": perimeter, "circularity": circularity,
            "aspect_ratio": aspect_ratio, "extent": extent, "solidity": solidity}

def extract_color_features(hsv, contour):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    pixels = hsv[mask == 255]
    if len(pixels) == 0:
        return {"h_mean": 0, "h_std": 0, "s_mean": 0, "s_std": 0, "v_mean": 0, "v_std": 0}
    h, s, v = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    return {"h_mean": float(np.mean(h)), "h_std": float(np.std(h)),
            "s_mean": float(np.mean(s)), "s_std": float(np.std(s)),
            "v_mean": float(np.mean(v)), "v_std": float(np.std(v))}

def calculate_entropy(gray_values):
    if len(gray_values) == 0:
        return 0.0
    histogram = np.bincount(gray_values, minlength=256)
    probabilities = histogram / float(len(gray_values))
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))

def extract_texture_features(gray, contour):
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    values = gray[mask == 255]
    if len(values) == 0:
        return {"gray_mean": 0, "gray_std": 0, "entropy": 0}
    return {"gray_mean": float(np.mean(values)), "gray_std": float(np.std(values)),
            "entropy": calculate_entropy(values)}

def create_feature_vector(shape, color, texture):
    return np.array([
        shape["area"], shape["perimeter"], shape["circularity"], shape["aspect_ratio"],
        shape["extent"], shape["solidity"], color["h_mean"], color["h_std"],
        color["s_mean"], color["s_std"], color["v_mean"], color["v_std"],
        texture["gray_mean"], texture["gray_std"], texture["entropy"]
    ], dtype=np.float32)
