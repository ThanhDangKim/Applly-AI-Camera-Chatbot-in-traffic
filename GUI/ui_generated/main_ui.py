from PyQt6.QtWidgets import QWidget, QGridLayout, QVBoxLayout, QLabel, QHBoxLayout, QFrame, QPushButton
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QUrl, Qt, QTimer
from PyQt6.QtGui import QIcon, QCursor
import os
import requests
from datetime import datetime
from PyQt6.QtWidgets import QSizePolicy
import webbrowser

class VideoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("6 Video Stream Viewer")
        # self.setFixedSize(1530, 800)  # Tăng chiều cao để đủ chỗ
        self.showMaximized()
        self.setWindowIcon(QIcon(os.path.join(os.path.dirname(__file__), "..", "Resources", "traffic.ico")))

        # Main layout bao ngoài
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        self.setLayout(main_layout)

        # Layout dạng lưới cho 6 khung camera
        grid_layout = QGridLayout()
        grid_layout.setSpacing(10)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addLayout(grid_layout)

        self.time_timer = QTimer()
        self.time_timer.timeout.connect(self.update_timestamp_labels)
        self.time_timer.start(1000)

        self.info_labels = []

        camera_info = [
            "Camera 1: Xa lộ Hà Nội - Đường D1",
            "Camera 2: Xa lộ Hà Nội - Đường D400",
            "Camera 3: Xa lộ Hà Nội - Đường Lê Văn Việt",
            "Camera 4: Xa lộ Hà Nội - Cầu ngã tư Thủ Đức",
            "Camera 5: Xa lộ Hà Nội - Thảo Điền",
            "Camera 6: Xa lộ Hà Nội - Đường 120",
        ]

        for i in range(6):
            frame = QFrame()
            frame.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
            frame.setLineWidth(2)
            frame.setFixedSize(498, 380)
            outer_layout = QVBoxLayout()
            outer_layout.setContentsMargins(10, 10, 10, 10)
            outer_layout.setSpacing(10)
            frame.setLayout(outer_layout)

            title_label = QLabel(camera_info[i])
            title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
            outer_layout.addWidget(title_label)

            content_layout = QHBoxLayout()
            content_layout.setSpacing(15)

            view = QWebEngineView()
            view.setUrl(QUrl(f"http://127.0.0.1:5000/video{i}"))
            view.setFixedSize(320, 320)
            content_layout.addWidget(view)

            info_layout = QVBoxLayout()
            cam_labels = []

            flow_label = QLabel("Trạng thái: Tốt")
            avg_speed_label = QLabel("Tốc độ trung bình: Đang cập nhật")
            density_label = QLabel("")
            light_label = QLabel("")
            timestamp_label = QLabel("Thời gian:")

            for label in [flow_label, avg_speed_label, density_label, light_label]:
                label.setAlignment(Qt.AlignmentFlag.AlignLeft)
                label.setWordWrap(True)
                label.setStyleSheet("font-size: 12px;")
                label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
                info_layout.addWidget(label)
                cam_labels.append(label)

            info_layout.addStretch()

            timestamp_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
            timestamp_label.setWordWrap(True)
            timestamp_label.setStyleSheet("font-size: 12px;")
            timestamp_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            info_layout.addWidget(timestamp_label)
            cam_labels.append(timestamp_label)

            content_layout.addLayout(info_layout)
            outer_layout.addLayout(content_layout)
            grid_layout.addWidget(frame, i // 3, i % 3)
            self.info_labels.append(cam_labels)

        # Nút mở chatbot (dưới grid_layout)
        chatbot_button = QPushButton("💬 Mở Trợ lý Chatbot")
        chatbot_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        chatbot_button.setFixedSize(200, 38)
        chatbot_button.setStyleSheet("""
            QPushButton {
                color: white;
                background-color: #444;
                border: 2px solid #00BFFF;
                border-radius: 8px;
                font-weight: bold;
                font-size: 15px;
            }
            QPushButton:hover {
                background-color: #2893ff;
                color: black;
            }
        """)
        chatbot_button.clicked.connect(self.open_chatbot_url)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(chatbot_button)
        button_layout.addStretch()

        main_layout.addLayout(button_layout)

        self.traffic_timer = QTimer()
        self.traffic_timer.timeout.connect(self.update_traffic_data_from_api)
        self.traffic_timer.start(1000)

        self.light_timer = QTimer()
        self.light_timer.timeout.connect(self.update_light_data_from_api)
        self.light_timer.start(3000)


    def update_traffic_data_from_api(self):
        try:
            response = requests.get("http://127.0.0.1:5000/traffic_data")
            if response.status_code == 200:
                response_data = response.json()
                cam_index = response_data["cam_index"]
                direction_data = response_data["data"]
                current_cycle = response_data["cycle"]
                avg_speeds = []
                for d in direction_data:
                    direction_info = direction_data.get(d, {})
                    avg_speed = direction_info.get("avg_speed", 0)
                    if avg_speed > 0:
                        avg_speeds.append(avg_speed)

                if avg_speeds:
                    overall_avg_speed = round(sum(avg_speeds) / len(avg_speeds), 2)
                else:
                    overall_avg_speed = 0

                if 0 <= cam_index < len(self.info_labels):
                    if current_cycle == None:
                        flow_text = "Trạng thái: Tốt"
                    else:
                        if current_cycle == "tb":
                            flow_text = "Luồng phương tiện: Lên - Xuống"
                        else:
                            flow_text = "Luồng phương tiện: Trái - Phải"

                    self.info_labels[cam_index][0].setText(flow_text)
                    if overall_avg_speed != 0:
                        self.info_labels[cam_index][1].setText(f"Tốc độ trung bình: {overall_avg_speed} km/h")

                    dir_map = {"top": "Bắc", "bottom": "Nam", "left": "Tây", "right": "Đông"}
                    densities = []
                    for d in direction_data:
                        direction_info = direction_data.get(d, {})
                        count = direction_info.get("count", 0)
                        density = direction_info.get("traffic_level", "N/A")

                        direction_name = dir_map.get(d, d)  # fallback là chính key nếu không tìm thấy
                        if count > 0:
                            densities.append(f"{direction_name}: {count} - {density}")
                        else:
                            densities.append(f"{direction_name}: 0 - Không có")
                    self.info_labels[cam_index][2].setText("Mật độ xe:\n" + "\n".join(densities) if densities else "")

        except Exception as e:
            print("Lỗi lấy dữ liệu giao thông:", e)

    def update_light_data_from_api(self):
        try:
            response = requests.get("http://127.0.0.1:5000/light_times")
            if response.status_code == 200:
                response_data = response.json()
                light_index = response_data["light_index"]
                light_direction = response_data["light_direction"]
                light_data = response_data["data"]
                if light_direction == "tb":
                    ld = "chu kỳ lên - xuống tiếp theo"
                else:
                    ld = "chu kỳ trái - phải tiếp theo"

                if 0 <= light_index < len(self.info_labels):
                    green_time = light_data[light_direction].get("green", 0)
                    yellow_time = light_data[light_direction].get("yellow", 0)
                    red_opposite = light_data[light_direction].get("red", 0)

                    light_text = f"Đèn xanh: {green_time}s\nĐèn vàng: {yellow_time}s\nĐèn đỏ bên kia: {red_opposite}s"
                    if int(green_time) != 0:
                        self.info_labels[light_index][3].setText(f"Dự đoán đèn của {ld}:\n" + light_text)
        except Exception as e:
            print("Lỗi lấy dữ liệu đèn:", e)

    def update_timestamp_labels(self):
        current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        for cam in self.info_labels:
            cam[4].setText("Thời gian: " + current_time)

    def open_chatbot_url(self):
        webbrowser.open("http://dummylink:8501/")  

