# Vision Pipeline

## Cài thư viện
```bash
pip install numpy opencv-python pyrealsense2
```

## Chạy chương trình
Mở terminal tại thư mục `vision_pipeline`, sau đó chạy:
```bash
python main.py
```

## Các module
- `config.py`: thông số cấu hình
- `camera.py`: RealSense và đồng bộ RGB/depth
- `color_detection.py`: histogram Hue, tạo mask và gọi tên màu
- `segmentation.py`: morphology, connected components, contour
- `feature_extraction.py`: đặc trưng hình dạng, màu, texture
- `depth_processing.py`: tọa độ 3D
- `visualization.py`: vẽ thông tin lên ảnh
- `object_detection.py`: điều phối phát hiện vật thể
- `main.py`: vòng lặp chính và hiển thị

Lưu ý: tọa độ X/Y/Z hiện là tọa độ camera (mét), chưa phải tọa độ robot. Cần hiệu chuẩn hệ camera-robot trước khi gửi lệnh gắp.
