import pyrealsense2 as rs
import numpy as np
from config import DEPTH_KERNEL

def get_3d_position(depth_frame, depth_intrinsics, cx, cy):
    x, y = int(round(cx)), int(round(cy))
    width, height = depth_frame.get_width(), depth_frame.get_height()
    if not (0 <= x < width and 0 <= y < height):
        return None
    half = DEPTH_KERNEL // 2
    distances = []
    for yy in range(max(0, y-half), min(height, y+half+1)):
        for xx in range(max(0, x-half), min(width, x+half+1)):
            distance = depth_frame.get_distance(xx, yy)
            if distance > 0:
                distances.append(distance)
    if not distances:
        return None
    depth = float(np.median(distances))
    point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [float(x), float(y)], depth)
    return {"X": point[0], "Y": point[1], "Z": point[2]}
