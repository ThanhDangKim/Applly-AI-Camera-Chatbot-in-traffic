from flask import Blueprint, jsonify
from API.data_store import *
from collections import deque
import time

api_bp = Blueprint('api', __name__)

@api_bp.route("/traffic_data", methods=["GET"])
def get_traffic_data():
    CAM_CONFIGS, cam_index = get_cam_configs()
    if CAM_CONFIGS is None:
        return jsonify({"error": "No data available"}), 404

    now = time.time()
    direction_data = {}
    for direction in CAM_CONFIGS.direction_zones.keys():
        count = CAM_CONFIGS.direction_counts[direction]
        speeds = CAM_CONFIGS.direction_speeds[direction]
        valid_speeds = [s for s in speeds if 8 < s < 70]
        avg_speed = round(sum(valid_speeds) / len(valid_speeds), 2) if valid_speeds else 0
        # avg_speed = round(sum(speeds) / len(speeds), 2) if speeds else 0

        CAM_CONFIGS.recent_counts[direction] = deque([
        t for t in CAM_CONFIGS.recent_counts[direction] if now - t <= CAM_CONFIGS.window_seconds
        ])
        count_den = len(CAM_CONFIGS.recent_counts[direction])
        if direction == "top" or direction == "bottom":
            density = count_den / CAM_CONFIGS.tb if CAM_CONFIGS.tb > 0 else 0
        else:
            density = count_den / CAM_CONFIGS.lr if CAM_CONFIGS.lr > 0 else 0
        traffic_level = CAM_CONFIGS.classify_traffic(density, avg_speed)
        direction_data[direction] = {
            "count": count,
            "avg_speed": avg_speed,
            "traffic_level": traffic_level
        }

    return jsonify({
        "cam_index": cam_index,
        "data": direction_data,
        "cycle": CAM_CONFIGS.current_cycle_direction
    })


@api_bp.route("/light_times", methods=["GET"])
def get_light_times():
    light_data, light_direction, light_index = get_latest_light_times()
    if not light_data:
        return jsonify({"message": "No light time data available"}), 404
    
    return jsonify({
        "light_index": light_index,
        "light_direction": light_direction,
        "data": light_data
    })
