import pyrealsense2 as rs
from config import COLOR_WIDTH, COLOR_HEIGHT, FPS

def create_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, COLOR_WIDTH, COLOR_HEIGHT, rs.format.z16, FPS)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_sensor = profile.get_device().first_depth_sensor()
    print("Depth scale:", depth_sensor.get_depth_scale())
    return pipeline, align

def get_aligned_frames(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    return aligned.get_color_frame(), aligned.get_depth_frame()
