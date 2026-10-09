import cv2
import numpy as np
import pyrealsense2 as rs

from config import ROI_X1, ROI_Y1, ROI_X2, ROI_Y2, GAUSSIAN_KERNEL
from camera import create_realsense, get_aligned_frames
from color_detection import create_color_mask
from object_detection import detect_objects
from visualization import draw_object

def main():
    pipeline = None
    try:
        pipeline, align = create_realsense()
        print("=" * 60)
        print("RealSense started")
        print("Press Q to quit")
        print("=" * 60)

        while True:
            color_frame, depth_frame = get_aligned_frames(pipeline, align)
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            roi = color_image[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2].copy()
            blur = cv2.GaussianBlur(roi, GAUSSIAN_KERNEL, 0)
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

            depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            objects, valid_peaks, valid_mask = detect_objects(blur, hsv, gray, depth_frame, depth_intrinsics)

            camera_display = color_image.copy()
            cv2.rectangle(camera_display, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (255, 255, 0), 2)
            cv2.putText(camera_display, f"Detected color groups: {len(valid_peaks)}",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            for object_id, obj in enumerate(objects, start=1):
                draw_object(roi, obj["contour"], obj["shape"], obj["color"],
                            obj["color_name"], obj["position"], object_id)

            combined_mask = np.zeros_like(valid_mask)
            for peak in valid_peaks:
                combined_mask = cv2.bitwise_or(combined_mask, create_color_mask(hsv, peak))

            cv2.imshow("01 - CAMERA", camera_display)
            cv2.imshow("02 - ROI / RESULT", roi)
            cv2.imshow("03 - VALID COLOR MASK", valid_mask)
            cv2.imshow("04 - ADAPTIVE HSV MASK", combined_mask)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception as exc:
        print("\\nERROR:", exc)
    finally:
        if pipeline is not None:
            pipeline.stop()
        cv2.destroyAllWindows()
        print("Program stopped.")

if __name__ == "__main__":
    main()
