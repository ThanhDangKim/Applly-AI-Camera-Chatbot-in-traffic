import cv2
import os
import argparse
import numpy as np
from infer_faster import TensorRTInfer
from visualize_faster import visualize_detections
from image_batcher import ImageBatcher

def preprocess_frame(frame, input_shape):
    # Resize và chuẩn hóa khung hình theo định dạng đầu vào của model
    _, c, h, w = input_shape
    resized = cv2.resize(frame, (w, h))
    img = resized.astype(np.float32)
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Thêm batch dim
    return img, resized.shape[1] / w  # Trả về ảnh và tỉ lệ scale


def draw_detections_on_frame(frame, detections, labels):
    # Lưu frame tạm để xử lý trực quan hóa
    tmp_path = "_tmp_frame.jpg"
    frame = cv2.resize(frame, (640, 640))
    cv2.imwrite(tmp_path, frame)
    visualize_detections(tmp_path, tmp_path, detections, labels)
    return cv2.imread(tmp_path)


def main():
    engine_path = "D:/colab-env/DATN/BE/model.trt"
    input_video_path = "D:/colab-env/DATN/BE/videos/video2_11h_13h.mp4"
    det2_config_path = "D:/colab-env/DATN/BE/config.yaml"

    labels = [
        "objects", "bike", "bus", "bus-l-", "bus-s-", "car", "extra large truck",
        "green-light", "large bus", "medium truck", "motorbike", "red-light",
        "small bus", "small truck", "truck", "truck-l-", "truck-m-",
        "truck-s-", "truck-xl-", "yellow-light"
    ]

    # Khởi tạo inference engine
    trt_infer = TensorRTInfer(engine_path)
    input_shape, _ = trt_infer.input_spec()

    # Cài đặt video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError("Không thể mở file video: {}".format(input_video_path))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # print(f"Đang xử lý frame {frame_count}", end="\r")
        batch, scale = preprocess_frame(frame, input_shape)
        detections = trt_infer.infer(batch, scales=[scale], nms_threshold=0.3)
        vis_frame = draw_detections_on_frame(frame, detections[0], labels)
        
        # Phát trực tiếp
        cv2.imshow("Phát hiện đối tượng - Live", vis_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nDừng sớm theo yêu cầu người dùng.")
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("\nHoàn tất xử lý video trực tiếp.")


if __name__ == "__main__":
    main()
