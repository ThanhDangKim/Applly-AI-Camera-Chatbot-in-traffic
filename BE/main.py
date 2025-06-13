from flask import Flask, Response
import cv2
import os
import torch
import json
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import sys, distutils.core
dist = distutils.core.run_setup(r"C:\Users\ADMIN\OneDrive\Documents\Btap_Code\VisualStudioCode\Python\SecurityCamera\detectron2\setup.py")
sys.path.insert(0, os.path.abspath(r"C:\Users\ADMIN\OneDrive\Documents\Btap_Code\VisualStudioCode\Python\SecurityCamera\detectron2"))
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from deep_sort_realtime.deepsort_tracker import DeepSort
import supervision as sv
import psycopg2
import threading
import library_cam as cv
import time
from collections import defaultdict, deque
import math
from API.data_store import *
from API.api import api_bp
import signal
import sys

def preprocess_frame(frame, apply_canny=False):
    """
    Làm mịn + tăng tương phản (CLAHE trên YCrCb.Y) nhưng vẫn giữ màu gốc cho Detectron2.
    """
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)

    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_clahe = clahe.apply(y)

    enhanced = cv2.merge((y_clahe, cr, cb))
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_YCrCb2BGR)

    return enhanced_bgr


app = Flask(__name__)
app.register_blueprint(api_bp)

VIDEO_PATHS = [f"./videos/video{i}_11h_13h.mp4" for i in range(6)]
CAM_CONFIGS = {
    0: cv.Cam0Config(VIDEO_PATHS[0]),
    1: cv.Cam1Config(VIDEO_PATHS[1]),
    2: cv.Cam2Config(VIDEO_PATHS[2]),
    3: cv.Cam3Config(VIDEO_PATHS[3]),
    4: cv.Cam4Config(VIDEO_PATHS[4]),
    5: cv.Cam5Config(VIDEO_PATHS[5])
}

with open("./output_vehicle_faster_rcnn_R_50_FPN_1x_13nd/thing_classes.json", "r") as f:
    CLASS_NAMES = json.load(f)

PREDICTORS = []
for _ in range(len(VIDEO_PATHS)):
    cfg = get_cfg()
    OUTPUT_DIR = "./output_vehicle_faster_rcnn_R_50_FPN_1x_13nd"
    cfg.merge_from_file(os.path.join(OUTPUT_DIR, "config.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.WEIGHTS = os.path.join(OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.RETINANET.NUM_CLASSES = len(CLASS_NAMES)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = DefaultPredictor(cfg)
    PREDICTORS.append(predictor)

def create_tracker():
    return DeepSort(
        max_age=20,
        embedder="torchreid",
        embedder_model_name="osnet_x1_0",
        half=True,
        bgr=True,
        embedder_gpu=True,
    )

TRACKERS = [create_tracker() for _ in range(len(VIDEO_PATHS))]
CLASS_COUNTS = [defaultdict(int) for _ in range(len(VIDEO_PATHS))]
START_TIMES = [datetime.now() for _ in range(len(VIDEO_PATHS))]
COUNTED_IDS = [set() for _ in range(len(VIDEO_PATHS))]

VIDEO_CAPS = [cv2.VideoCapture(path) for path in VIDEO_PATHS]
VIDEO_LOCKS = [threading.Lock() for _ in VIDEO_PATHS]

box_annotator = sv.BoxAnnotator(thickness=1)
label_annotator = sv.LabelAnnotator(
    text_thickness=0,
    text_scale=0.2,
    text_padding=0,
    color=sv.ColorPalette.DEFAULT,
    text_color=sv.Color.WHITE
)

def save_to_db(stream_index):
    try:
        conn = psycopg2.connect(
            host="",
            port="",
            database="",
            user="",
            password=""
        )
        cursor = conn.cursor()

        now = datetime.now()
        date = now.date()  
        time_slot = (now.hour * 60 + now.minute) // 30
        camera_id = stream_index + 1

        # === INSERT INTO vehicle_stats ===
        print("1")
        direction_class_counts = CAM_CONFIGS[stream_index].direction_class_counts
        print(direction_class_counts)
        for direction, class_counts in direction_class_counts.items():
            for vehicle_type, count in class_counts.items():
                if count > 0:
                    cursor.execute("""
                        INSERT INTO vehicle_stats (camera_id, date, time_slot, direction, vehicle_type, vehicle_count)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (camera_id, date, time_slot, direction, vehicle_type)
                        DO UPDATE SET vehicle_count = vehicle_stats.vehicle_count + EXCLUDED.vehicle_count;
                    """, (camera_id, date, time_slot, direction, vehicle_type, count))
        # === INSERT INTO avg_speeds ===
        print("2")
        if CAM_CONFIGS[stream_index].count_speed > 0:
            average_speed = CAM_CONFIGS[stream_index].total_speed / CAM_CONFIGS[stream_index].count_speed
            average_speed = round(float(average_speed), 2)
        else:
            average_speed = 0.0

        cursor.execute("""
            INSERT INTO avg_speeds (camera_id, date, time_slot, average_speed)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (camera_id, date, time_slot)
            DO UPDATE SET average_speed = EXCLUDED.average_speed;
        """, (camera_id, date, time_slot, average_speed))

        # === INSERT INTO daily_traffic_summary (tùy chọn nếu theo ngày) ===
        print("3")
        # Tổng số lượng xe
        total_vehicle_count = sum(
            sum(counts.values()) for counts in direction_class_counts.values()
        )

        # Tìm slot cao điểm (tạm thời dùng slot hiện tại), hướng có nhiều xe nhất
        direction_totals = {d: sum(v.values()) for d, v in direction_class_counts.items()}
        if direction_totals:
            direction_with_most_traffic = max(direction_totals, key=direction_totals.get)
        else:
            direction_with_most_traffic = "unknown"

        cursor.execute("""
            INSERT INTO daily_traffic_summary (camera_id, date, total_vehicle_count, avg_speed, peak_time_slot, direction_with_most_traffic)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (camera_id, date)
            DO UPDATE SET
                total_vehicle_count = daily_traffic_summary.total_vehicle_count + EXCLUDED.total_vehicle_count,
                avg_speed = EXCLUDED.avg_speed,
                peak_time_slot = EXCLUDED.peak_time_slot,
                direction_with_most_traffic = EXCLUDED.direction_with_most_traffic;
        """, (camera_id, date, total_vehicle_count, average_speed, time_slot, direction_with_most_traffic))

        conn.commit()
        cursor.close()
        conn.close()
        print(f"[DB] ✅ Saved data for camera {camera_id} at {now} (slot {time_slot})")

        # Reset sau khi lưu
        CAM_CONFIGS[stream_index].direction_class_counts.clear()
        CAM_CONFIGS[stream_index].total_speed = 0
        CAM_CONFIGS[stream_index].count_speed = 0

    except Exception as e:
        print(f"[DB ERROR] ❌ {e}")


def async_save_to_db(stream_index):
    threading.Thread(target=save_to_db, args=(stream_index,), daemon=True).start()

def generate_stream(cap, lock, predictor, tracker, class_counts, start_time, stream_index):
    target_size = (320, 320)
    MAX_AREA = target_size[0] * target_size[1] * 0.25
    MAX_BBOX_HEIGHT_RATIO = 0.30
    detection_interval = 1
    frame_id = 0
    last_detections = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")
    input_prev_time = time.time()
    input_fps = 0

    while True:
        with lock:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        # Tính FPS đầu vào (chỉ thời gian đọc frame)
        input_curr_time = time.time()
        input_fps = 1 / (input_curr_time - input_prev_time)
        input_prev_time = input_curr_time

        frame_id += 1
        tracks = []
        # frame = preprocess_frame(frame, apply_canny=False)
        frame = cv2.resize(frame, target_size)
        ai_input_frame = preprocess_frame(frame.copy(), apply_canny=False)
        if frame_id % detection_interval == 0:
            # frame = cv2.resize(frame, target_size)
            # outputs = predictor(frame)
            # ai_input_frame = cv2.resize(ai_input_frame, target_size)
            outputs = predictor(ai_input_frame)
            instances = outputs["instances"]
            boxes = instances.pred_boxes.tensor.cpu().numpy()
            scores = instances.scores.cpu().numpy()
            class_ids = instances.pred_classes.cpu().numpy().astype(int)

            last_detections = [
                ([int(x1), int(y1), int(x2 - x1), int(y2 - y1)], float(score), int(cls))
                for (x1, y1, x2, y2), score, cls in zip(boxes, scores, class_ids)
                if (x2 - x1) * (y2 - y1) < MAX_AREA
            ]
            # tracks = tracker.update_tracks(last_detections, frame=ai_input_frame)
        else:
            # frame = cv2.resize(frame, target_size)
            last_detections = []
            # tracks = tracker.update_tracks(last_detections, frame=None)
        # if frame_id % detection_interval == 0:
        if last_detections:
            # print(f"LD: {last_detections}")
            tracks = tracker.update_tracks(last_detections, frame=ai_input_frame)
        else:
            tracks = tracker.update_tracks([], frame=ai_input_frame)

        det_xyxy, det_conf, det_class, det_labels, track_ids = [], [], [], [], []
        for track in tracks:
            if not track.is_confirmed() :
                continue
            
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            bbox_height = y2 - y1
            frame_height = frame.shape[0]
            frame_width = frame.shape[1]

            if bbox_height > MAX_BBOX_HEIGHT_RATIO * frame_height:
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            # class_id = getattr(track, "det_class", None)  # dùng getattr để tránh lỗi nếu det_class chưa tồn tại
            # if class_id is None or class_id not in CLASS_NAMES:
            #     continue  # bỏ qua nếu không hợp lệ
            class_id = track.det_class
            class_name = CLASS_NAMES[class_id]

            if class_name != "light" and track_id not in COUNTED_IDS[stream_index]:
                class_counts[class_name] += 1
                COUNTED_IDS[stream_index].add(track_id)

            det_xyxy.append([x1, y1, x2, y2])
            det_conf.append(1.0)
            det_class.append(class_id)
            # det_labels.append(class_name)
            det_labels.append(f"{class_name} #{track_id}")
            track_ids.append(track_id)   

        if det_xyxy:
            detections_sv = sv.Detections(
                xyxy=np.array(det_xyxy),
                confidence=np.array(det_conf),
                class_id=np.array(det_class),
                data={"label": det_labels},
                tracker_id=np.array(track_ids)
            )
            
            zone = CAM_CONFIGS[stream_index].zone
            zone_annotator = CAM_CONFIGS[stream_index].zone_annotator
            in_zone = zone.trigger(detections=detections_sv)
            # CAM_CONFIGS[stream_index].draw_direction_zones(ai_input_frame, CAM_CONFIGS[stream_index].direction_zones)
            for j, inside in enumerate(in_zone):
                tid = detections_sv.tracker_id[j]
                box = detections_sv.xyxy[j]
                center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                
                if inside:
                    CAM_CONFIGS[stream_index].track_coords_history[tid].appendleft(center)
                    CAM_CONFIGS[stream_index].track_bbox_history[tid].appendleft(box)

            frame = box_annotator.annotate(scene=frame, detections=detections_sv)
            # frame = zone_annotator.annotate(scene=frame)
            frame = label_annotator.annotate(scene=frame, detections=detections_sv, labels=det_labels)
            # frame = CAM_CONFIGS[stream_index].draw_zones(frame)

            # Tính vận tốc
            flag_direction = False
            for j, inside in enumerate(in_zone):
                tid = detections_sv.tracker_id[j]
                box = detections_sv.xyxy[j]
                if tid in CAM_CONFIGS[stream_index].track_coords_history and tid in COUNTED_IDS[stream_index]:
                    # === Gán hướng vào thời điểm đếm ===
                    if len(CAM_CONFIGS[stream_index].track_coords_history[tid]) >= 10 and tid not in CAM_CONFIGS[stream_index].track_directions:
                        flag_direction = True
                        direction = CAM_CONFIGS[stream_index].get_adjusted_direction(CAM_CONFIGS[stream_index].track_coords_history[tid], CAM_CONFIGS[stream_index].direction_zones)
                        # ✅ Đếm xe theo hướng
                        CAM_CONFIGS[stream_index].direction_counts[direction] += 1
                        CAM_CONFIGS[stream_index].track_directions[tid] = direction # Lưu lại hướng cho xe này
                        
                        # Cập nhật số lượng xe theo hướng và loại xe
                        CAM_CONFIGS[stream_index].direction_class_counts[direction][class_name] += 1
                        CAM_CONFIGS[stream_index].logger.info(f"TID: {tid} - direction: {direction}\n")

                        # ✅ Thêm timestamp vào recent_counts deque
                        now = time.time()
                        CAM_CONFIGS[stream_index].recent_counts[direction].append(now)

                    if len(CAM_CONFIGS[stream_index].track_coords_history[tid]) >= 6:
                        bbox = CAM_CONFIGS[stream_index].track_bbox_history[tid][0]
                        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                        if inside:
                            if CAM_CONFIGS[stream_index].check_zone(center):
                            # if is_in_center_zone(center, frame_width, frame_height):
                                # set_trace()
                                mpp = CAM_CONFIGS[stream_index].mpp
                                M = CAM_CONFIGS[stream_index].M
                                speed = CAM_CONFIGS[stream_index].calculate_speed_MPP(CAM_CONFIGS[stream_index].track_coords_history[tid], CAM_CONFIGS[stream_index].track_bbox_history[tid], 
                                                                                        fps=fps, frame_width=frame_width, frame_height=frame_height, 
                                                                                        mpp=mpp, matrix=M)
                                CAM_CONFIGS[stream_index].logger.info(f"Center: {center} - Speed: {speed} - TID: {tid}")

                                # === Áp dụng làm mượt với speed_history ===
                                if tid not in CAM_CONFIGS[stream_index].speed_history:
                                    CAM_CONFIGS[stream_index].speed_history[tid] = deque(maxlen=CAM_CONFIGS[stream_index].max_history_length)
                                CAM_CONFIGS[stream_index].speed_history[tid].append(speed)
                            
                                if len(CAM_CONFIGS[stream_index].speed_history[tid]) >= 3:
                                    avg_speed = sum(CAM_CONFIGS[stream_index].speed_history[tid]) / len(CAM_CONFIGS[stream_index].speed_history[tid])
                                else:
                                    avg_speed = speed

                                if avg_speed > 1 and avg_speed < 80:
                                    CAM_CONFIGS[stream_index].total_speed += avg_speed
                                    CAM_CONFIGS[stream_index].count_speed += 1
                                    
                                    # === Xác định hướng hiện tại và lưu tốc độ ===
                                    if tid in CAM_CONFIGS[stream_index].track_directions:
                                        direction = CAM_CONFIGS[stream_index].track_directions[tid]
                                        CAM_CONFIGS[stream_index].direction_speeds[direction].append(avg_speed)
                                        # Cập nhật danh sách tốc độ theo hướng và loại xe
                                        CAM_CONFIGS[stream_index].direction_class_speeds[direction][class_name].append(avg_speed)   

                                if avg_speed > 8 and avg_speed < 70:
                                    x1, y1, x2, y2 = map(int, box)
                                    label = f"{math.floor(avg_speed)} km/h"
                                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
            #Update giá trị
            update_cam_configs(CAM_CONFIGS[stream_index], stream_index)

            # Thêm bản sao direction_counts vào lịch sử
            if flag_direction == True:
                CAM_CONFIGS[stream_index].direction_counts_history.append(CAM_CONFIGS[stream_index].direction_counts.copy())
                flag_direction = False

            # Khi đủ 10 frame, kiểm tra chu kỳ
            if len(CAM_CONFIGS[stream_index].direction_counts_history) >= 9:
                cycle_ended, current_cycle_direction = CAM_CONFIGS[stream_index].detect_cycle_direction(CAM_CONFIGS[stream_index].direction_counts_history)
                # cv2.putText(frame, f"Current Cycle: {current_cycle_direction}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                if cycle_ended:
                    update_light_time(stream_index, CAM_CONFIGS[stream_index].previous_cycle_direction, CAM_CONFIGS[stream_index].green_time, CAM_CONFIGS[stream_index].yellow_time, CAM_CONFIGS[stream_index].red_time_opposite)
                    # Reset direction_counts để bắt đầu chu kỳ mới
                    CAM_CONFIGS[stream_index].direction_counts = CAM_CONFIGS[stream_index].direction_counts_ori
                    CAM_CONFIGS[stream_index].direction_speeds = CAM_CONFIGS[stream_index].direction_speeds_ori
                    CAM_CONFIGS[stream_index].track_directions = {}  # Reset hướng xe đã đếm
                    
                # Giữ lại 10 frame gần nhất
                CAM_CONFIGS[stream_index].direction_counts_history = CAM_CONFIGS[stream_index].direction_counts_history[-3:]

        # Cập nhật biến lưu trữ dùng chung cho API

        if (datetime.now() - start_time[0]) >= timedelta(minutes=30):
            async_save_to_db(stream_index)
            COUNTED_IDS[stream_index].clear()
            start_time[0] = datetime.now()

        _, buffer = cv2.imencode('.jpg', frame)
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
        )

def create_video_route(index):
    def video():
        return Response(generate_stream(
            VIDEO_CAPS[index],
            VIDEO_LOCKS[index],
            PREDICTORS[index],
            TRACKERS[index],
            CLASS_COUNTS[index],
            [START_TIMES[index]],
            index
        ), mimetype='multipart/x-mixed-replace; boundary=frame')
    return video

for i in range(len(VIDEO_PATHS)):
    app.add_url_rule(f"/video{i}", f"video{i}", create_video_route(i))

def run_server():
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

if __name__ == '__main__':
    run_server()

