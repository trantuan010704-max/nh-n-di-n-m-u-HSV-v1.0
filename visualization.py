import cv2

def draw_object(frame, contour, shape, color_features, color_name, position, object_id):
    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        cx, cy = int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])
    else:
        x, y, w, h = cv2.boundingRect(contour)
        cx, cy = x + w // 2, y + h // 2
    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
    x, y, w, h = cv2.boundingRect(contour)
    tx, ty = max(5, x), max(20, y - 10)
    lines = [
        (f"Object {object_id}", (0, 255, 0), 0.55),
        (f"Color: {color_name}", (0, 255, 255), 0.55),
        (f"Area: {shape['area']:.0f}", (255, 255, 255), 0.45),
        (f"Circularity: {shape['circularity']:.2f}", (255, 255, 255), 0.45),
        (f"H: {color_features['h_mean']:.1f}", (255, 255, 255), 0.45),
        (f"S: {color_features['s_mean']:.1f}", (255, 255, 255), 0.45),
        (f"V: {color_features['v_mean']:.1f}", (255, 255, 255), 0.45),
    ]
    if position is not None:
        lines.append((f"X:{position['X']:.3f} Y:{position['Y']:.3f} Z:{position['Z']:.3f}", (0, 255, 255), 0.42))
    for i, (text, color, scale) in enumerate(lines):
        cv2.putText(frame, text, (tx, ty + i * 20), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2 if i < 2 else 1)
