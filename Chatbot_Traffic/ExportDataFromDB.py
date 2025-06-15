import psycopg2
from datetime import datetime

def export_data_for_chatbot(output_file="./Data_Traffic/TXT/chatbot_data.txt"):
    try:
        conn = psycopg2.connect(
            host="",
            port="",
            database="",
            user="",
            password=""
        )
        cursor = conn.cursor()

        direction_map = {
            "top": "Bắc",
            "bottom": "Nam",
            "left": "Tây",
            "right": "Đông"
        }

        event_type_map = {
            "congestion": "ùn tắc giao thông",
            "accident": "tai nạn giao thông",
            "roadwork": "công trình đang thi công"
        }

        vehicle_type_map = {
            "truck": "xe tải",
            "motorbike": "xe máy",
            "bus": "xe buýt",
            "car": "xe ô tô"
        }

        with open(output_file, "w", encoding="utf-8") as f:

            # 1. Thống kê số lượng xe theo ngày, giờ, loại và hướng
            print("1")
            cursor.execute("""
                SELECT 
                    c.name AS camera_name,
                    a.description AS camera_location,
                    a.name AS area_name,
                    ca.location_detail AS location_detail,
                    v.date,
                    v.time_slot,
                    v.vehicle_type,
                    v.direction,
                    SUM(v.vehicle_count) AS total_count
                FROM vehicle_stats v
                JOIN cameras c ON v.camera_id = c.id
                JOIN camera_area ca ON c.id = ca.camera_id
                JOIN areas a ON ca.area_id = a.id
                GROUP BY c.name, a.description, a.name, ca.location_detail, v.date, v.time_slot, v.vehicle_type, v.direction
                ORDER BY v.date DESC, v.time_slot;
            """)
            f.write("=== Thống kê lượng xe cộ ===\n")
            for row in cursor.fetchall():
                camera_name = row[0]
                camera_location = row[1]
                area_name = row[2]
                location_detail = row[3]
                date = row[4]
                time_slot = row[5]
                vehicle_type = row[6]
                direction = row[7]
                total_count = row[8]

                # Chuyển định dạng ngày sang tiếng Việt
                date_vietnamese = f"ngày {date.day} tháng {date.month} năm {date.year}"

                # Tính khung giờ từ time_slot (mỗi slot là 30 phút)
                hour_start = time_slot * 30 // 60
                minute_start = (time_slot * 30) % 60
                hour_end = (time_slot + 1) * 30 // 60
                minute_end = ((time_slot + 1) * 30) % 60

                time_range = f"{hour_start:02d}:{minute_start:02d}–{hour_end:02d}:{minute_end:02d}"

                sentence = (
                    f"Vào {date_vietnamese}, tại camera được đặt tại {location_detail}, "
                    f"thuộc khu vực {area_name} của {camera_location}, "
                    f"trong khoảng thời gian {time_range}, hệ thống ghi nhận {total_count} phương tiện loại \"{vehicle_type_map.get(vehicle_type, vehicle_type)}\" "
                    f"di chuyển từ hướng \"{direction_map.get(direction, direction)}\".\n"
                )

                f.write(sentence)

            # 2. Tốc độ trung bình theo loại xe, hướng, camera
            print("2")
            cursor.execute("""
                SELECT 
                    c.name AS camera_name,
                    a.date,
                    a.time_slot,
                    a.average_speed,
                    ca.location_detail,
                    ar.name AS area_name,
                    ar.description AS area_description
                FROM avg_speeds a
                JOIN cameras c ON a.camera_id = c.id
                JOIN camera_area ca ON c.id = ca.camera_id
                JOIN areas ar ON ca.area_id = ar.id
                ORDER BY a.date DESC, a.time_slot;
            """)
            f.write("\n=== Tốc độ trung bình ===\n")
            for row in cursor.fetchall():
                cam_name = row[0]
                date = row[1]
                time_slot = row[2]
                speed = row[3]
                location_detail = row[4]
                area_name = row[5]
                area_description = row[6]

                if speed != 0:
                    # Chuyển định dạng ngày sang tiếng Việt
                    date_vietnamese = f"ngày {date.day} tháng {date.month} năm {date.year}"

                    # Tính khung giờ từ time_slot
                    hour_start = time_slot * 30 // 60
                    minute_start = (time_slot * 30) % 60
                    hour_end = (time_slot + 1) * 30 // 60
                    minute_end = ((time_slot + 1) * 30) % 60

                    time_range = f"{hour_start:02d}:{minute_start:02d}–{hour_end:02d}:{minute_end:02d}"

                    # Tạo câu văn tự nhiên
                    sentence = (
                        f"Vào {date_vietnamese}, trong khung giờ {time_range}, tốc độ trung bình ghi nhận tại camera "
                        f"đặt tại {location_detail}, khu vực {area_name} – {area_description} là {speed:.2f} km/h.\n"
                    )

                    f.write(sentence)

            # 3. Camera & khu vực
            print("3")
            cursor.execute("""
                SELECT ca.location_detail, a.name, a.description
                FROM camera_area ca
                JOIN cameras c ON ca.camera_id = c.id
                JOIN areas a ON ca.area_id = a.id
            """)
            f.write("\n=== Khu vực lắp camera ===\n")
            for row in cursor.fetchall():
                cam_name, area_name, loc_detail = row
                f.write(f"Camera quan sát được đặt tại {cam_name} thuộc {area_name} của khu vực: {loc_detail}\n")

            # 4. Daily summary
            print("4")
            cursor.execute("""
                SELECT 
                    c.name AS camera_name,
                    d.date,
                    d.total_vehicle_count,
                    d.avg_speed,
                    d.peak_time_slot,
                    d.direction_with_most_traffic,
                    ca.location_detail,
                    a.name AS area_name,
                    a.description AS area_description
                FROM daily_traffic_summary d
                JOIN cameras c ON d.camera_id = c.id
                JOIN camera_area ca ON c.id = ca.camera_id
                JOIN areas a ON ca.area_id = a.id
                ORDER BY d.date DESC;
            """)
            f.write("\n=== Tổng hợp tình hình giao thông hằng ngày ===\n")
            for row in cursor.fetchall():
                cam_name = row[0]
                date = row[1]
                total = row[2]
                avg_speed = row[3]
                peak_slot = row[4]
                top_dir = row[5]
                location_detail = row[6]
                area_name = row[7]
                area_description = row[8]

                if avg_speed != 0:
                    # Chuyển định dạng ngày sang tiếng Việt
                    date_vietnamese = f"ngày {date.day} tháng {date.month} năm {date.year}"

                    # Tính khung giờ cao điểm
                    hour_start = peak_slot * 30 // 60
                    minute_start = (peak_slot * 30) % 60
                    hour_end = (peak_slot + 1) * 30 // 60
                    minute_end = ((peak_slot + 1) * 30) % 60
                    time_range = f"{hour_start:02d}:{minute_start:02d}–{hour_end:02d}:{minute_end:02d}"

                    # Câu văn mô tả
                    sentence = (
                        f"Vào {date_vietnamese}, tại camera đặt tại {location_detail}, khu vực {area_name} của {area_description}, "
                        f"đã ghi nhận tổng cộng {total} lượt xe. Tốc độ trung bình là {avg_speed:.2f} km/h. "
                        f"Khung giờ cao điểm là từ {time_range}, với hướng lưu thông đông nhất là hướng {direction_map.get(top_dir, top_dir)}.\n"
                    )

                    f.write(sentence)

            # 5. Traffic events (optional)
            print("5")
            cursor.execute("""
                SELECT 
                    c.name AS camera_name,
                    e.event_time,
                    e.event_type,
                    e.description,
                    ca.location_detail,
                    a.name AS area_name,
                    a.description AS area_description
                FROM traffic_events e
                JOIN cameras c ON e.camera_id = c.id
                JOIN camera_area ca ON c.id = ca.camera_id
                JOIN areas a ON ca.area_id = a.id
                ORDER BY e.event_time DESC
                LIMIT 100;
            """)
            f.write("\n=== Các sự kiện giao thông gần đây ===\n")
            for row in cursor.fetchall():
                cam_name = row[0]
                event_time = row[1]
                e_type = row[2]
                desc = row[3]
                location_detail = row[4]
                area_name = row[5]
                area_description = row[6]

                event_date = event_time.date()
                event_hour = event_time.hour
                event_minute = event_time.minute

                event_time_vn = (
                    f"lúc {event_hour:02d} giờ {event_minute:02d} phút, ngày {event_date.day} tháng {event_date.month} năm {event_date.year}"
                )

                # Câu mô tả rõ ràng, dễ hiểu
                sentence = (
                    f"Vào {event_time_vn}, camera đặt tại {location_detail}, khu vực {area_name} của {area_description}, "
                    f"đã ghi nhận sự kiện \"{event_type_map.get(e_type, e_type)}\". Mô tả chi tiết cho sự việc: {desc}.\n"
                )

                f.write(sentence)

        cursor.close()
        conn.close()
        print(f"Dữ liệu đã được xuất ra '{output_file}' thành công!")

    except Exception as e:
        print("Lỗi khi xuất dữ liệu:", str(e))

# export_data_for_chatbot()
