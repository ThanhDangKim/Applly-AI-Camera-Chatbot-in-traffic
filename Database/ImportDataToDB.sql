-- INSERT INTO cameras (name, location, installed_at) VALUES
-- ('Camera 1', 'Xa lo ha noi - Duong D1', '2025-05-26 12:08:44.437418'),
-- ('Camera 2', 'Xa lo ha noi - Duong D400', '2025-05-26 12:08:44.437418'),
-- ('Camera 3', 'Xa lo ha noi - Duong Le Van Viet', '2025-05-26 12:08:44.437418'),
-- ('Camera 4', 'Xa lo ha noi - Len cau Nga tu thu duc', '2025-05-26 12:08:44.437418'),
-- ('Camera 5', 'Xa lo ha noi - Thao Dien', '2025-05-26 12:08:44.437418'),
-- ('Camera 6', 'Xa lo ha noi - Duong 120', '2025-05-26 12:08:44.437418');

INSERT INTO cameras (name, location) VALUES
('Camera 1', 'Xa lo ha noi - Duong D1'),
('Camera 2', 'Xa lo ha noi - Duong D400'),
('Camera 3', 'Xa lo ha noi - Duong Le Van Viet'),
('Camera 4', 'Xa lo ha noi - Len cau Nga tu thu duc'),
('Camera 5', 'Xa lo ha noi - Thao Dien'),
('Camera 6', 'Xa lo ha noi - Duong 120');

INSERT INTO areas (name, description) VALUES
('Phường Thảo Điền', 'Quận 2, TP.HCM'),
('Phường Tân Phú', 'TP Thủ Đức, TP.HCM'),
('Phường Hiệp Phú', 'TP Thủ Đức, TP.HCM'),
('Phường Linh Trung', 'TP Thủ Đức, TP.HCM');

INSERT INTO camera_area (camera_id, area_id, location_detail) VALUES
(1, 1, 'Xa lộ Hà Nội – Đường D1'),
(2, 2, 'Xa lộ Hà Nội – Đường D400'),
(3, 3, 'Xa lộ Hà Nội – Đường Lê Văn Việt'),
(4, 4, 'Xa lộ Hà Nội – Lên cầu Ngã Tư Thủ Đức'),
(5, 1, 'Xa lộ Hà Nội – Thảo Điền'),
(6, 2, 'Xa lộ Hà Nội – Đường 120');

INSERT INTO traffic_events (camera_id, event_time, event_type, description) VALUES
(1, '2025-08-12 07:30:00', 'congestion', 'Kẹt xe nghiêm trọng vào giờ cao điểm buổi sáng gần Đường D1.'),
(2, '2025-09-03 18:15:00', 'accident', 'Va chạm nhẹ giữa hai ô tô xảy ra tại Đường D400.'),
(3, '2025-10-05 14:00:00', 'roadwork', 'Thi công bảo trì mặt đường tại khu vực Lê Văn Việt – đóng một làn đường.'),
(4, '2025-08-25 17:45:00', 'congestion', 'Ùn tắc giao thông tại Ngã tư Thủ Đức do lượng xe đông vào giờ tan tầm.'),
(5, '2025-09-20 09:00:00', 'roadwork', 'Cải tạo hệ thống thoát nước tại khu vực Thảo Điền – hạn chế giao thông.'),
(6, '2025-10-10 20:00:00', 'accident', 'Tai nạn giao thông giữa hai xe máy tại khu vực Đường 120 – có lực lượng chức năng xử lý.');

