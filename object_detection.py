import cv2
from color_detection import detect_color_peaks, create_color_mask, get_object_color_name
from segmentation import morphology_process, connected_components_filter, find_object_contours
from feature_extraction import extract_shape_features, extract_color_features, extract_texture_features, create_feature_vector
from depth_processing import get_3d_position
from config import MIN_SATURATION, MIN_VALUE, ROI_X1, ROI_Y1

def detect_objects(blur, hsv, gray, depth_frame, depth_intrinsics):
    saturation, value = hsv[:, :, 1], hsv[:, :, 2]
    valid_mask = (((saturation >= MIN_SATURATION) & (value >= MIN_VALUE)).astype("uint8") * 255)
    valid_peaks, _ = detect_color_peaks(hsv, valid_mask)
    all_objects = []
    for color_id, peak in enumerate(valid_peaks, start=1):
        mask = create_color_mask(hsv, peak)
        _, closing = morphology_process(mask)
        filtered_mask, _ = connected_components_filter(closing)
        for contour in find_object_contours(filtered_mask):
            shape = extract_shape_features(contour)
            color = extract_color_features(hsv, contour)
            color_name = get_object_color_name(color)
            texture = extract_texture_features(gray, contour)
            feature_vector = create_feature_vector(shape, color, texture)
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            position = get_3d_position(depth_frame, depth_intrinsics, ROI_X1 + cx, ROI_Y1 + cy)
            all_objects.append({
                "color_id": color_id, "color_name": color_name, "hue_peak": peak,
                "contour": contour, "shape": shape, "color": color, "texture": texture,
                "feature_vector": feature_vector, "cx": cx, "cy": cy, "position": position
            })
    return all_objects, valid_peaks, valid_mask
