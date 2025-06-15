import numpy as np
import supervision as sv
import cv2
from collections import defaultdict, deque
from deep_sort_realtime.deepsort_tracker import DeepSort
import logging
import math

def write_logger():
    # Xóa toàn bộ handler khỏi root logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Tạo logger riêng
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Xóa tất cả handler khỏi logger riêng
    logger.handlers.clear()

    # Tạo file handler
    file_handler = logging.FileHandler(
        r"log/log_4.txt",
        encoding='utf-8',
        mode='a'
    )
    file_handler.setLevel(logging.INFO)

    # Tạo formatter và gắn vào file handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Gắn file handler vào logger
    logger.addHandler(file_handler)

    return logger

class Cam4Config:
    def __init__(self, source_path="./Data_video/Cam_4/test_faster.mp4"):
        # === DeepSORT tracker ===
        self.tracker = DeepSort(max_age=25)

        # === Video info ===
        self.source_path = source_path
        self.video_info = sv.VideoInfo.from_video_path(self.source_path)
        self.frame_width, self.frame_height = 320, 320  # Resize target

        # === Zone setup ===
        center_x, center_y = self.frame_width // 2, self.frame_height // 2
        rect_w, rect_h = int(self.frame_width * 1.0), int(self.frame_height * 0.90)
        self.polygon = np.array([
            [center_x - rect_w // 2, center_y - rect_h // 2],
            [center_x + rect_w // 2, center_y - rect_h // 2],
            [center_x + rect_w // 2, center_y + rect_h // 2],
            [center_x - rect_w // 2, center_y + rect_h // 2]
        ])
        self.zone = sv.PolygonZone(polygon=self.polygon)
        self.label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.8)
        self.zone_annotator = sv.PolygonZoneAnnotator(zone=self.zone)
        self.box_annotator = sv.BoxAnnotator()
        self.zones = self.get_zones()

        # === Height data & transformation ===
        self.REAL_HEIGHTS = {
            "car": 1.5,
            "bus": 3.2,
            "truck": 3.5,
            "motorcycle": 1.2
        }
        self.TARGET_WIDTH = 25  # meters
        self.TARGET_HEIGHT = 40  # meters

        self.src_pts = np.float32(self.polygon)
        self.dst_pts = np.float32([
            [0, 0],
            [self.TARGET_WIDTH, 0],
            [self.TARGET_WIDTH, self.TARGET_HEIGHT],
            [0, self.TARGET_HEIGHT]
        ])
        self.M = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)

        # === Directional zones ===
        self.direction_zones = {
            "bottom": [
                (int(self.frame_width * 0.24), int(self.frame_height * 0.45)),
                (int(self.frame_width * 0.68), int(self.frame_height * 0.42)),
                (int(self.frame_width * 0.99), int(self.frame_height * 0.78)),
                (int(self.frame_width * 0.17), int(self.frame_height * 0.85))
            ],
            "top": [
                (int(self.frame_width * 0.0), int(self.frame_height * 0.00)),
                (int(self.frame_width * 0.0), int(self.frame_height * 0.00)),
                (int(self.frame_width * 0.0), int(self.frame_height * 0.00)),
                (int(self.frame_width * 0.0), int(self.frame_height * 0.00))
            ],
            "right": [
                (int(self.frame_width * 0.00), int(self.frame_height * 0.00)),
                (int(self.frame_width * 0.00), int(self.frame_height * 0.00)),
                (int(self.frame_width * 0.00), int(self.frame_height * 0.00)),
                (int(self.frame_width * 0.00), int(self.frame_height * 0.00))
            ]
        }
        self.tb = 40
        self.lr = 40

        # === Counters and Histories ===
        self.counted_ids = set()
        self.total_count = 0
        self.total_speed = 0
        self.count_speed = 0
        self.track_coords_history = defaultdict(lambda: deque(maxlen=20))
        self.track_bbox_history = defaultdict(lambda: deque(maxlen=20))
        self.mpp = 0.035  # meters per pixel

        self.speed_history = {}  # Last 5 speeds per track
        self.max_history_length = 5

        # === Direction tracking ===
        self.direction_counts_ori = {"top": 0, "bottom": 0, "right": 0}
        self.direction_speeds_ori = {"top": [], "bottom": [], "right": []}
        self.direction_counts = {"top": 0, "bottom": 0, "right": 0}
        self.direction_speeds = {"top": [], "bottom": [], "right": []}
        self.track_directions = {}
        # Lưu số lượng xe theo hướng + loại xe
        self.direction_class_counts = defaultdict(lambda: defaultdict(int))  # {direction: {class_name: count}}
        # Lưu danh sách tốc độ theo hướng + loại xe
        self.direction_class_speeds = defaultdict(lambda: defaultdict(list))  # {direction: {class_name: [speed1, speed2, ...]}}

        # Giá trị reset nếu chỉ có 1 chu kỳ
        self.recent_counts = {
            "top": deque(),
            "bottom": deque(),
            "right": deque()
        }
        self.window_seconds = 30

        self.previous_cycle_direction = None
        self.current_cycle_direction = None
        self.direction_counts_history = []

        self.change_confirm_count = 0
        self.confirm_threshold = 2

        # === Traffic signal parameters ===
        self.VTw_dict = {"top": 3.2, "bottom": 3.8, "right": 4.1}  # Green time
        self.VTL_dict = {"top": 3.4, "bottom": 3.1, "right": 4.2}  # Red time
        self.green_time = 0 
        self.yellow_time = 0
        self.red_time_opposite = 0

        # === Write logger ===
        self.logger = write_logger()

    def get_zones(self):
        zones = [
            ("Lane1", 
            (int(self.frame_width * 0.24), int(self.frame_height * 0.55)),
            (int(self.frame_width * 0.42), int(self.frame_height * 0.55)),
            (int(self.frame_width * 0.17), int(self.frame_height * 0.85)),
            (int(self.frame_width * 0.58), int(self.frame_height * 0.85)),
            (128, 255, 255)),  # Yellow

            ("Lane2", 
            (int(self.frame_width * 0.50), int(self.frame_height * 0.01)),
            (int(self.frame_width * 0.63), int(self.frame_height * 0.01)),
            (int(self.frame_width * 0.52), int(self.frame_height * 0.53)),
            (int(self.frame_width * 0.99), int(self.frame_height * 0.53)),
            (255, 255, 0)),  # Cyan

            ("Lane3", 
            (int(self.frame_width * 0.52), int(self.frame_height * 0.53)),
            (int(self.frame_width * 0.63), int(self.frame_height * 0.53)),
            (int(self.frame_width * 0.52), int(self.frame_height * 0.63)),
            (int(self.frame_width * 0.99), int(self.frame_height * 0.63)),
            (128, 0, 128)),  # Purple

            ("Lane4", 
            (int(self.frame_width * 0.61), int(self.frame_height * 0.64)),
            (int(self.frame_width * 0.99), int(self.frame_height * 0.64)),
            (int(self.frame_width * 0.65), int(self.frame_height * 0.87)),
            (int(self.frame_width * 0.99), int(self.frame_height * 0.87)),
            (0, 255, 0)),  # Green

            ("High_Lane1", 
            (int(self.frame_width * 0.33), int(self.frame_height * 0.55)),
            (int(self.frame_width * 0.58), int(self.frame_height * 0.55)),
            (int(self.frame_width * 0.33), int(self.frame_height * 0.64)),
            (int(self.frame_width * 0.58), int(self.frame_height * 0.64)),
            (128, 255, 255)),  # Yellow

            ("High_Lane2", 
            (int(self.frame_width * 0.626), int(self.frame_height * 0.384)),
            (int(self.frame_width * 0.746), int(self.frame_height * 0.384)),
            (int(self.frame_width * 0.626), int(self.frame_height * 0.525)),
            (int(self.frame_width * 0.746), int(self.frame_height * 0.525)),
            (255, 255, 0)),  # Cyan

            ("High_Lane3", 
            (int(self.frame_width * 0.627), int(self.frame_height * 0.53)),
            (int(self.frame_width * 0.812), int(self.frame_height * 0.53)),
            (int(self.frame_width * 0.627), int(self.frame_height * 0.63)),
            (int(self.frame_width * 0.812), int(self.frame_height * 0.63)),
            (128, 0, 128)),  # Purple

            ("High_Lane4", 
            (int(self.frame_width * 0.61), int(self.frame_height * 0.64)),
            (int(self.frame_width * 0.812), int(self.frame_height * 0.64)),
            (int(self.frame_width * 0.61), int(self.frame_height * 0.812)),
            (int(self.frame_width * 0.812), int(self.frame_height * 0.812)),
            (0, 255, 0)),  # Green
        ]
        return zones
    
    def draw_zones(self, frame):
        # Các vùng cần vẽ, mỗi vùng là một tuple: (tên vùng, topleft, topright, bottomleft, bottomright, màu)
        for label, tl, tr, bl, br, color in self.zones:
            # Vẽ đa giác 4 điểm
            points = np.array([tl, tr, br, bl], np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)
            # Ghi nhãn tại góc trên bên trái
            cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        return frame

    def is_in_lane1_zone(self, center, zone):
        x, y = center
        _, tl, tr, bl, br, _ = zone

        # đơn giản hóa: kiểm tra theo hình chữ nhật nằm ngang
        if bl[0] <= x <= br[0] and tl[1] <= y <= bl[1]:
            self.logger.info("1. Lane1")
            return True
        return False

    def is_in_lane2_zone(self, center, zone):
        x, y = center
        _, tl, tr, bl, br, _ = zone

        # đơn giản hóa: kiểm tra theo hình chữ nhật nằm ngang
        if tr[0] <= x <= br[0] and tr[1] <= y <= bl[1]:
            self.logger.info("2. Lane2")
            return True
        return False

    def is_in_lane3_zone(self, center, zone):
        x, y = center
        _, tl, tr, bl, br, _ = zone

        # đơn giản hóa: kiểm tra theo hình chữ nhật nằm ngang
        if tr[0] <= x <= br[0] and tl[1] <= y <= bl[1]:
            self.logger.info("3. Lane3")
            return True
        return False

    def is_in_lane4_zone(self, center, zone):
        x, y = center
        _, tl, tr, bl, br, _ = zone

        # đơn giản hóa: kiểm tra theo hình chữ nhật nằm ngang
        if tl[0] <= x <= br[0] and tr[1] <= y <= bl[1]:
            self.logger.info("4. Lane4")
            return True
        return False
    
    def is_in_high_lane1_zone(self, center, zone):
        x, y = center
        _, tl, tr, bl, br, _ = zone

        # đơn giản hóa: kiểm tra theo hình chữ nhật nằm ngang
        if bl[0] <= x <= br[0] and tl[1] <= y <= bl[1]:
            self.logger.info("1. high_Lane1")
            return True
        return False

    def is_in_high_lane2_zone(self, center, zone):
        x, y = center
        _, tl, tr, bl, br, _ = zone

        # đơn giản hóa: kiểm tra theo hình chữ nhật nằm ngang
        if tl[0] <= x <= br[0] and tr[1] <= y <= bl[1]:
            self.logger.info("2. high_Lane2")
            return True
        return False

    def is_in_high_lane3_zone(self, center, zone):
        x, y = center
        _, tl, tr, bl, br, _ = zone

        # đơn giản hóa: kiểm tra theo hình chữ nhật nằm ngang
        if tl[0] <= x <= br[0] and tl[1] <= y <= bl[1]:
            self.logger.info("3. high_Lane3")
            return True
        return False

    def is_in_high_lane4_zone(self, center, zone):
        x, y = center
        _, tl, tr, bl, br, _ = zone

        # đơn giản hóa: kiểm tra theo hình chữ nhật nằm ngang
        if tl[0] <= x <= br[0] and tr[1] <= y <= bl[1]:
            self.logger.info("4. high_Lane4")
            return True
        return False
    
    def transform_to_ground_plane(self, point, matrix):
        point_array = np.array([[point]], dtype='float32')  # shape (1, 1, 2)
        transformed = cv2.perspectiveTransform(point_array, matrix)
        return transformed[0][0]  # trả về (x, y) trên mặt phẳng thực
    
    def calculate_filtered_distance_pixels(self, transformed_coords):
        """
        Tính khoảng cách đã lọc outlier giữa các frame (median + IQR).
        Trả về tổng khoảng cách (đơn vị: pixel).
        """
        distances = []
        for i in range(1, 6):
            dx = transformed_coords[i][0] - transformed_coords[i-1][0]
            dy = transformed_coords[i][1] - transformed_coords[i-1][1]
            if abs(dx) <= 0.015 and abs(dy) <= 0.015:
                return 0, 0 
            dist = (dx**2 + dy**2)**0.5
            distances.append(dist)

        if not distances:
            return 0, 0

        q1 = sorted(distances)[len(distances)//4]
        q3 = sorted(distances)[3*len(distances)//4]
        iqr = q3 - q1
        filtered = [d for d in distances if (q1 - 1.5*iqr) <= d <= (q3 + 1.5*iqr)]

        if not filtered:
            filtered = distances  # fallback nếu loại hết

        avg_distance = sum(filtered) / len(filtered)
        return avg_distance, len(filtered)
    
    def adjust_speed_by_position(self, y_center, frame_height, alpha, power=1.5):
        """
        y_center: tọa độ y tâm bbox
        frame_height: chiều cao khung hình
        """
        distance_factor = (y_center) / frame_height  # gần 0: gần camera, gần 1: xa camera
        correction = 1 + alpha * ((1 - distance_factor) ** power)  # alpha là hệ số điều chỉnh
        return correction

    def capped_log_ratio(self, standard_coords, end, max_value=12):
        # Tính khoảng cách Euclidean
        distance = math.hypot(standard_coords[0] - end[0], standard_coords[1] - end[1])
        # Áp dụng logarit cơ số 10 và cộng 1 để tránh log(0)
        log_distance = math.log(distance + 1) / math.log(1.3)
        # Giới hạn giá trị không vượt quá max_value
        return min(log_distance, max_value)
    
    def calculate_speed_MPP(self, coordinates, bbox_list, fps, frame_width, frame_height, mpp, matrix):
        """
        Tính tốc độ theo công thức 2 (quy đổi pixel sang mét).
        Trả về tốc độ tính bằng km/h.
        """
        if len(coordinates) < 2:
            return 0
        transformed_coords = [self.transform_to_ground_plane(p, matrix) for p in coordinates]
        start = transformed_coords[-1]
        end = transformed_coords[0]
        distance_pixels, len_dp = self.calculate_filtered_distance_pixels(transformed_coords)
        self.logger.info(f"Distance pixels filtered: {distance_pixels}")

        # Xác định vùng để scale `mpp`
        bbox = bbox_list[0]  # lấy bbox mới nhất
        bbox_height = bbox[3] - bbox[1]
        # bbox_height_mean = sum([bbox[3] - bbox[1] for bbox in bbox_list]) / len(bbox_list)
        # ratio_height = bbox_height / bbox_height_mean

        p_standard = (frame_width / 2, frame_height / 2)
        standard_coords = self.transform_to_ground_plane(p_standard, matrix)
        ratio = self.capped_log_ratio(standard_coords, end, 12)
        self.logger.info(f"Ratio: {ratio}")
        
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        mpp_scaled = 0
        # Scale theo vùng
        if self.is_in_lane1_zone(center, self.zones[0]) == True:
            block_parameter = self.adjust_speed_by_position(center[1], frame_height, 33.5, 2)
            self.logger.info(f"BP: {block_parameter}")    
            # scale theo chiều cao bbox
            real_height = 2.5
            mpp = 0.63
            scale_factor = real_height / (bbox_height + 1e-6)
            if self.is_in_high_lane1_zone(center, self.zones[4]) == True:
                mpp = mpp * 3.66
                mpp_scaled = mpp * scale_factor * ratio * block_parameter
            else:
                mpp = mpp * 3.81
                mpp_scaled = mpp * scale_factor * ratio * block_parameter

        elif (self.is_in_lane4_zone(center, self.zones[3]) == True):
            block_parameter = self.adjust_speed_by_position(center[1], frame_height, 51.5, 1)
            self.logger.info(f"BP: {block_parameter}")

            real_height = 1.8
            mpp = 0.7
            scale_factor = real_height / (bbox_height * mpp + 1e-6)  # tránh chia 0
            if self.is_in_high_lane4_zone(center, self.zones[7]) == True:
                mpp = mpp * 0.403
                mpp_scaled = mpp * scale_factor * ratio * block_parameter
            else:
                mpp = mpp * 0.716
                mpp_scaled = mpp * scale_factor * ratio * block_parameter

        elif (self.is_in_lane3_zone(center, self.zones[2]) == True):
            block_parameter = self.adjust_speed_by_position(center[1], frame_height, 77.4, 2)
            self.logger.info(f"BP: {block_parameter}")

            # scale theo chiều cao bbox
            real_height = 1.8
            mpp = 1.88
            scale_factor = real_height / (bbox_height * mpp + 1e-6)  # tránh chia 0
            if self.is_in_high_lane3_zone(center, self.zones[6]) == True:
                mpp = mpp * 0.94
                mpp_scaled = mpp * scale_factor * ratio * block_parameter
            else:
                mpp = mpp * 1.79
                mpp_scaled = mpp * scale_factor * ratio * block_parameter

        elif (self.is_in_lane2_zone(center, self.zones[1]) == True):
            block_parameter = self.adjust_speed_by_position(center[1], frame_height, 47.1, 0.9)
            self.logger.info(f"BP: {block_parameter}")

            real_height = 1.8
            mpp = 1.41
            scale_factor = real_height / (bbox_height * mpp + 1e-6)  # tránh chia 0
            if self.is_in_high_lane2_zone(center, self.zones[5]) == True:
                mpp = mpp * 0.95
                mpp_scaled = mpp * scale_factor * ratio * block_parameter
            else:
                mpp = mpp * 1.21
                mpp_scaled = mpp * scale_factor * ratio * block_parameter
            
        # mpp_scaled = mpp
        if mpp_scaled == 0:
            return 0
        distance_meters = distance_pixels * mpp_scaled
        if len_dp != 0:
            time = len_dp / fps
        else: 
            return 0
        speed_kmh = (distance_meters / time) * 3.6
        self.logger.info(f"distance meters: {distance_meters}")
            
        return speed_kmh
    
    def draw_direction_zones(self, frame, direction_zones):
        colors = {
            "top": (255, 0, 0),
            "bottom": (0, 255, 0),
            "right": (0, 255, 255)
        }

        for dir_name, points in direction_zones.items():
            # Vẽ polygon (zone là hình 4 cạnh)
            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=colors[dir_name], thickness=2)
            # Đổ màu mờ nếu cần (tuỳ chọn)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], color=colors[dir_name])
            alpha = 0.2  # độ mờ
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


    def get_direction(self, center, direction_zones):
        closest_zone = None
        min_distance = float('inf')

        for dir_name, zone_points in direction_zones.items():
            pts = np.array(zone_points, np.int32)

            # Kiểm tra nếu center nằm trong polygon
            is_inside = cv2.pointPolygonTest(pts, center, False)
            if is_inside >= 0:
                return dir_name  # Trả về ngay nếu nằm trong zone

            # Nếu không nằm trong polygon, tính khoảng cách đến polygon
            distance = abs(cv2.pointPolygonTest(pts, center, True))  # Đo khoảng cách thực sự
            if distance < min_distance:
                min_distance = distance
                closest_zone = dir_name

        return closest_zone  # Trả về zone gần nhất nếu không nằm trong zone nào
    
    def get_direction_from_vector(self, history, threshold=0.05):
        if len(history) < 2:
            return "unknown"

        dx = history[0][0] - history[-1][0]
        dy = history[0][1] - history[-1][1]
        self.logger.info(f"dx: {dx} - dy: {dy}")

        if dx < 0 and dx < -threshold:
            if dy < 0 and dy < -threshold:
                return "left-top"
            elif dy > 0 and dy > threshold:
                return "left-bottom"
            else:
                return "left"
            
        elif dx > 0 and dx > threshold:
            if dy < 0 and dy < -threshold:
                return "right-top"
            elif dy > 0 and dy > threshold:
                return "right-bottom"
            else:
                return "right"
            
        else:
            if dy < 0 and dy < -threshold:
                return "top"
            elif dy > 0 and dy > threshold:
                return "bottom"
            else:
                return "unknown"


    def get_adjusted_direction(self, history, direction_zones, threshold=0.05):
        zone = self.get_direction(history[-1], direction_zones)  # Gần zone nào nhất
        vector_dir = self.get_direction_from_vector(history, threshold)  # Hướng di chuyển
        self.logger.info(f"Location 1st: {history[-1]} - Zone: {zone} - Vector: {vector_dir}")
        # Chuẩn hóa hướng vector thành hướng tổng quát (vd: bottom-left, top-right...)
        if (zone == "top" and vector_dir == "left-bottom") or (zone == "top" and vector_dir == "bottom") or (zone == "top" and vector_dir == "right-bottom"):
            return "top"
        if (zone == "right" and vector_dir == "right-bottom") or (zone == "top" and vector_dir == "right-bottom"):
            return "right"
        if (zone == "bottom" and vector_dir == "left-top") or (zone == "bottom" and vector_dir == "right-top") or (zone == "bottom" and vector_dir == "top"):
            return "bottom"
        else:
            return zone  # fallback theo vector nếu không thỏa

    ''' #### Density directions  

    - Mật độ = Số lượng xe / Diện tích đoạn đường (hoặc chiều dài đoạn vào)
    - Thời gian đèn xanh (giây) = hệ số * mật độ xe + hệ số * vận tốc di chuyển qua giao lộ

    Công thức cải tiến tín hiệu đèn:
    - VTL = TD / N

        - VTL: Vehicle Time Lost (trung bình thời gian mất mát, ví dụ: đèn xanh nhưng xe không đi).
        - TD: Tổng thời gian mất mát vì tín hiệu kém tối ưu (xe không qua dù đèn xanh).
        -  N: Tổng số phương tiện đã qua nút giao.

    ➡️ VTL phản ánh hiệu suất tín hiệu. Càng nhỏ thì hệ thống càng tối ưu.


    - VTw = VQL / N

        - VTw: Vehicle Time Waiting (thời gian trung bình chờ đèn xanh).
        - VQL: Tổng độ dài hàng đợi (tổng số phương tiện xếp hàng chờ đèn).
        - N: Tổng số phương tiện.
        
    ➡️ VTw phản ánh mức độ tắc nghẽn (nhiều phương tiện chờ = nhiều tắc).
    '''
    def classify_traffic(self, density, speed):
        if density > 1.2:
            return "tắc nghẽn"
        elif speed < 15:
            return "chậm"
        elif speed < 40:
            return "bình thường"
        else:
            return "nhanh"

    def get_coefficients(self, traffic_level):
        if traffic_level == "tắc nghẽn":
            return 3.5, 0.5
        elif traffic_level == "chậm":
            return 2.5, 1.2
        elif traffic_level == "bình thường":
            return 2.0, 1.5
        else:  # nhanh
            return 1.5, 2.0

    def calculate_light_times(self, direction_counts_snapshot, direction_speeds_snapshot, 
                            direction, area_or_length, VTw_dict, VTL_dict):
        if direction == 'tb':
            keys = ["top", "bottom"]
        else:
            keys = ["right"]

        green_times = []
        for key in keys:
            counts = direction_counts_snapshot.get(key, 0)
            speeds = direction_speeds_snapshot.get(key, [])
            density = counts / area_or_length if area_or_length > 0 else 0
            avg_speed = sum(speeds) / len(speeds) if speeds else 0
            traffic_level = self.classify_traffic(density, avg_speed)
            print(f"Traffic level: {traffic_level}")
            coef_d, coef_s = self.get_coefficients(traffic_level)
            base_green_time = coef_d * density + coef_s * avg_speed

            # Thêm ảnh hưởng từ VTw
            VTw = VTw_dict.get(key, 0)
            green_time = base_green_time + 0.3 * VTw  # Hệ số điều chỉnh 
            green_times.append(green_time)

        green_light_time = sum(green_times) / 2
        green_diff = abs(green_times[0] - green_times[1])

        # Đèn vàng vẫn giữ logic cũ
        combined_speeds = direction_speeds_snapshot.get(keys[0], []) + direction_speeds_snapshot.get(keys[1], [])
        avg_speed = sum(combined_speeds) / len(combined_speeds) if combined_speeds else 0
        yellow_light_time = max(3, min(5, avg_speed / 10))

        # Tính VTL trung bình của 2 hướng kia để tăng đèn đỏ
        opposite_keys = ["right"] if direction == "tb" else ["top", "bottom"]
        VTL_opp_avg = sum([VTL_dict.get(k, 0) for k in opposite_keys]) / 2

        red_light_time = green_light_time + yellow_light_time + 0.7 * VTL_opp_avg

        return round(green_light_time, 0), round(yellow_light_time, 0), round(red_light_time, 0)
    
    def log_green_light_time(self, cycle_direction, green_time, yellow_time, red_time, log_file='green_light_log.txt'):
        """
        Ghi log thời gian đèn xanh ra file.

        cycle_direction: 'tb' hoặc 'lr'
        green_time: thời gian đèn xanh tính được
        log_file: tên file log
        """
        from datetime import datetime
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f'''{now} | Previous Cycle: {cycle_direction} | 
                -> Green Light Time: {green_time} seconds
                -> Yellow Light Time: {yellow_time} seconds
                -> Red Light Time Opposite: {red_time} seconds\n''')
            
    def ema(self, values, window=5):
        """Tính exponential moving average (EMA)."""
        weights = np.exp(np.linspace(-1., 0., window))
        weights /= weights.sum()
        return np.convolve(values, weights, mode='valid')

    def get_average_growth_rate(self, series, window_size=10):
        """Tính tốc độ tăng trung bình (slope) trong một cửa sổ."""
        if len(series) < window_size:
            return 0
        recent = series[-window_size:]
        x = np.arange(len(recent))
        slope, _ = np.polyfit(x, recent, 1)
        return slope

    def detect_cycle_direction(self, direction_counts_history):
        tb_list = []
        lr_list = []
        for counts in direction_counts_history:
            tb = counts.get('top', 0) + counts.get('bottom', 0)
            lr = counts.get('right', 0)
            tb_list.append(tb)
            lr_list.append(lr)

        # Tính EMA để làm mượt
        tb_ema = self.ema(tb_list)
        lr_ema = self.ema(lr_list)
        
        min_len = min(len(tb_ema), len(lr_ema))
        tb_growth = self.get_average_growth_rate(tb_ema, window_size=min_len)
        lr_growth = self.get_average_growth_rate(lr_ema, window_size=min_len)

        growth_diff = lr_growth - tb_growth

        # Ngưỡng
        growth_threshold = 0.05
        difference_margin = 0.15

        # Xác định hướng luồng chính mới
        if growth_diff > difference_margin and lr_growth > growth_threshold:
            new_direction = "lr"
        elif -growth_diff > difference_margin and tb_growth > growth_threshold:
            new_direction = "tb"
        else:
            new_direction = self.current_cycle_direction

        # Nếu chưa có hướng nào thì khởi tạo
        if self.current_cycle_direction is None and self.previous_cycle_direction is None:
            self.current_cycle_direction = new_direction
            print(f"=> Chu kỳ khởi tạo: {self.previous_cycle_direction} -> {self.current_cycle_direction}")
            self.previous_cycle_direction = new_direction
            return False, self.current_cycle_direction

        # Kiểm tra chuyển chu kỳ
        if new_direction != self.current_cycle_direction:
            self.change_confirm_count += 1
        else:
            self.change_confirm_count = 0

        if self.change_confirm_count >= self.confirm_threshold:
            # Ghi log chu kỳ cũ
            if self.current_cycle_direction is not None:
                snapshot_counts = self.direction_counts.copy()
                snapshot_speeds = self.direction_speeds.copy()
                area_or_length = self.tb if self.previous_cycle_direction == "tb" else self.lr

                self.green_time, self.yellow_time, self.red_time_opposite = self.calculate_light_times(
                    direction_counts_snapshot=snapshot_counts,
                    direction_speeds_snapshot=snapshot_speeds,
                    direction=self.previous_cycle_direction,
                    area_or_length=area_or_length,
                    VTw_dict=self.VTw_dict,
                    VTL_dict=self.VTL_dict
                )

                print(f'''[OK] Chu kỳ {self.previous_cycle_direction} kết thúc 
                    -> thời gian đèn xanh gợi ý: {self.green_time} giây
                    -> thời gian đèn vàng gợi ý: {self.yellow_time} giây
                    -> thời gian đèn đỏ bên kia gợi ý: {self.red_time_opposite} giây''')

                log_file_light = r"log/light_time.txt"
                self.log_green_light_time(self.previous_cycle_direction, self.green_time, self.yellow_time, self.red_time_opposite, log_file_light)

            self.previous_cycle_direction = self.current_cycle_direction
            self.current_cycle_direction = new_direction
            self.change_confirm_count = 0
            print(f"=> Chu kỳ thay đổi: {self.previous_cycle_direction} -> {self.current_cycle_direction}")

            return True, self.current_cycle_direction

        return False, None
    
    def check_zone(self, center):
        if (self.is_in_lane1_zone(center, self.zones[0]) == True) or (self.is_in_lane2_zone(center, self.zones[1]) == True) or (self.is_in_lane3_zone(center, self.zones[2]) == True) or (self.is_in_lane4_zone(center, self.zones[3]) == True):
            return True
        else:
            return False
        