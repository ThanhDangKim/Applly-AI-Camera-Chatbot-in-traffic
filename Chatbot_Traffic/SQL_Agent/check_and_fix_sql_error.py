import json
import os
from difflib import SequenceMatcher

ERROR_MEMORY_FILE = "sql_error_memory.json"

def load_error_memory():
    if os.path.exists(ERROR_MEMORY_FILE):
        with open(ERROR_MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_error_memory(memory):
    with open(ERROR_MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

def is_similar(a, b, threshold=0.8):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold

def check_and_fix_sql_error(question, sql_wrong, error_msg):
    memory = load_error_memory()
    
    # 1. Tìm lỗi đã gặp
    for entry in memory:
        if is_similar(entry["question"], question) or is_similar(entry["sql_wrong"], sql_wrong):
            if is_similar(entry["error"], error_msg):
                print("🔁 Đã từng gặp lỗi này, dùng lại SQL đã sửa.")
                return entry["sql_fixed"]

    # 2. Chưa có trong lịch sử => cần sửa thủ công sau đó thêm vào
    return None  # Cho phép Agent retry lần sau

def add_sql_error_entry(question, sql_wrong, error_msg, sql_fixed):
    memory = load_error_memory()
    memory.append({
        "question": question.strip(),
        "sql_wrong": sql_wrong.strip(),
        "error": error_msg.strip(),
        "sql_fixed": sql_fixed.strip()
    })
    save_error_memory(memory)
    print("💾 Đã lưu lỗi SQL và cách sửa.")
