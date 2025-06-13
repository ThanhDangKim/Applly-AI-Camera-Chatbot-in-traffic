# data_store.py
## Object class
CAM_CONFIGS = None  
cam_index = 0
def update_cam_configs(new_configs, index):
    global CAM_CONFIGS, cam_index
    cam_index = index
    CAM_CONFIGS = new_configs

def get_cam_configs():
    return CAM_CONFIGS, cam_index

## Thời gian đèn
light_index = 0
light_direction = None
latest_light_times = {}
def update_light_time(index, direction, green, yellow, red):
    global latest_light_times, light_index, light_direction
    light_index = index
    light_direction = direction
    latest_light_times[direction] = {
        "green": green,
        "yellow": yellow,
        "red": red
    }

def get_latest_light_times():
    return latest_light_times, light_direction, light_index
