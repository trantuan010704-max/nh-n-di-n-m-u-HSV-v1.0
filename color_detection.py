import cv2
import numpy as np
from config import MIN_SATURATION, MIN_VALUE, MIN_COLOR_PIXELS, DEFAULT_HUE_RADIUS, MIN_PEAK_DISTANCE

def calculate_hue_histogram(hsv, valid_mask):
    return cv2.calcHist([hsv[:, :, 0]], [0], valid_mask, [180], [0, 180]).flatten()

def find_hue_peaks(hist):
    if hist.max() <= 0:
        return []
    smooth = np.convolve(hist, np.ones(9) / 9.0, mode="same")
    if smooth.max() <= 0:
        return []
    threshold = smooth.max() * 0.15
    candidates = []
    for h in range(180):
        left, center, right = smooth[(h - 1) % 180], smooth[h], smooth[(h + 1) % 180]
        if center >= left and center >= right and center >= threshold:
            candidates.append((h, center))
    candidates.sort(key=lambda item: item[1], reverse=True)
    peaks = []
    for hue, _ in candidates:
        if all(min(abs(hue - p), 180 - abs(hue - p)) >= MIN_PEAK_DISTANCE for p in peaks):
            peaks.append(hue)
    return sorted(peaks)

def create_color_mask(hsv, peak_hue, hue_radius=DEFAULT_HUE_RADIUS):
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    lower_h, upper_h = peak_hue - hue_radius, peak_hue + hue_radius
    if lower_h < 0:
        mask_h = (h >= lower_h + 180) | (h <= upper_h)
    elif upper_h >= 180:
        mask_h = (h >= lower_h) | (h <= upper_h - 180)
    else:
        mask_h = (h >= lower_h) & (h <= upper_h)
    mask_sv = (s >= MIN_SATURATION) & (v >= MIN_VALUE)
    return (mask_h & mask_sv).astype(np.uint8) * 255

def detect_color_peaks(hsv, valid_mask):
    histogram = calculate_hue_histogram(hsv, valid_mask)
    peaks = find_hue_peaks(histogram)
    valid_peaks = []
    for peak in peaks:
        mask = create_color_mask(hsv, peak)
        if cv2.countNonZero(mask) >= MIN_COLOR_PIXELS:
            valid_peaks.append(peak)
    return valid_peaks, histogram

def hue_to_color_name(hue):
    if hue < 8 or hue >= 170: return "RED"
    if hue < 20: return "ORANGE"
    if hue < 35: return "YELLOW"
    if hue < 85: return "GREEN"
    if hue < 100: return "CYAN"
    if hue < 130: return "BLUE"
    if hue < 160: return "PURPLE"
    if hue < 170: return "PINK"
    return "UNKNOWN"

def get_object_color_name(features):
    h, s, v = features["h_mean"], features["s_mean"], features["v_mean"]
    if s < MIN_SATURATION:
        if v < 60: return "BLACK"
        if v > 200: return "WHITE"
        return "GRAY"
    return hue_to_color_name(h)
