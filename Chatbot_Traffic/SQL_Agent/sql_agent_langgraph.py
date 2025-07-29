import re
import json
import os
import psycopg2
from difflib import SequenceMatcher
from langgraph.graph import StateGraph, END
from typing import TypedDict, Any, List, Tuple, Optional
# Các module đã có
import sys
# Thêm đường dẫn thư mục cha của folder chứa file .py vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_utils import *


# ============================ BƯỚC 1: CẤU HÌNH MÔ HÌNH ============================
# model_name = "bigcode/starcoder2-3b"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "../model/4BIT00006")
model, tokenizer = load_model(model_path)

def call_llm(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    match = re.search(r"(SELECT|WITH|INSERT|UPDATE|DELETE)[\s\S]+?;", text, re.IGNORECASE)
    return match.group(0) if match else text.strip()


# ============================ BƯỚC 2: ĐỊNH NGHĨA TRẠNG THÁI ============================
class SQLState(TypedDict):
    question: str
    sql_query: Optional[str]
    result: Any
    error: Optional[str]
    retry_count: int
    history: List[Tuple[str, str]]
    feedback_score: int  # +1 nếu người dùng thấy kết quả đúng, -1 nếu sai
    feedback_count: int  # tổng số lần feedback tiêu cực


# ============================ BƯỚC 3: MEMORY LƯU LỊCH SỬ LỖI ============================
ERROR_MEMORY_FILE = "SQL_Agent/sql_error_memory.json"

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
    for entry in memory:
        if is_similar(entry["question"], question) or is_similar(entry["sql_wrong"], sql_wrong):
            if is_similar(entry["error"], error_msg):
                print("🔁 Đã từng gặp lỗi này, dùng lại SQL đã sửa.")
                return entry["sql_fixed"]
    return None

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
 

# ============================ BƯỚC 4: DB & PROMPT ============================
def get_db_connection():
    try:
        conn = psycopg2.connect(
            host="localhost",
            port="5432",
            database="traffic_app_db",
            user="",
            password=""
        )
        return conn, None
    except Exception as e:
        return None, str(e)

def execute_sql_query(sql: str):
    try:
        conn, err = get_db_connection()
        if err:
            return None, err
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows, None
    except Exception as e:
        return None, str(e)

def get_schema_text():
    return """
BẢNG cameras:
- id: mã camera (PK)
- name: tên camera
- location: mô tả vị trí đặt camera (ví dụ: "Xa lộ Hà Nội - Đường D1")
- installed_at: ngày lắp đặt

BẢNG areas:
- id: mã khu vực (PK)
- name: tên khu vực (ví dụ: "Phường Thảo Điền")
- description: mô tả chi tiết khu vực

BẢNG camera_area:
- id: mã (PK)
- camera_id: khóa ngoại đến bảng cameras
- area_id: khóa ngoại đến bảng areas
- location_detail: mô tả chi tiết vị trí như "Xa lộ Hà Nội giao nhau với Lê Văn Việt"

BẢNG vehicle_stats:
- id: mã (PK)
- camera_id: khóa ngoại đến bảng cameras
- date: ngày thống kê
- time_slot: khung giờ trong ngày (0–47, mỗi khung 30 phút)
- direction: hướng di chuyển (top/bottom/left/right)
- vehicle_type: loại xe (car, bike, ...)
- vehicle_count: số lượng xe ghi nhận

BẢNG avg_speeds:
- id: mã (PK)
- camera_id: khóa ngoại đến bảng cameras
- date: ngày đo
- time_slot: khung giờ
- average_speed: vận tốc trung bình

BẢNG daily_traffic_summary:
- id: mã (PK)
- camera_id: khóa ngoại đến bảng cameras
- date: ngày thống kê
- total_vehicle_count: tổng số xe trong ngày
- avg_speed: vận tốc trung bình trong ngày
- peak_time_slot: khung giờ cao điểm
- direction_with_most_traffic: hướng có lưu lượng xe lớn nhất

BẢNG traffic_events:
- id: mã sự kiện (PK)
- camera_id: liên kết camera
- event_time: thời gian diễn ra sự kiện
- event_type: loại sự kiện (kẹt xe, tai nạn...)
- description: mô tả chi tiết
"""

def build_sql_prompt(state: SQLState) -> str:
    schema = get_schema_text()
    question = state["question"]
    retry_count = state.get("retry_count", 0)
    feedback_score = state.get("feedback_score", 0)

    # Prompt tuning khi hiệu suất kém
    tuning_tip = ""
    if retry_count >= 3 or feedback_score < 0:
        tuning_tip = """
            ❗ Lưu ý khi viết SQL:
            - Phải rõ ràng, đúng cột và bảng trong schema.
            - Tránh viết JOIN không cần thiết.
            - Đảm bảo điều kiện WHERE phù hợp với câu hỏi.
            - Ưu tiên GROUP BY hoặc hàm tổng hợp nếu có từ như 'trung bình', 'tổng', 'nhiều nhất'...
        """

    return f"""Bạn là chuyên gia SQL. Hãy viết một câu truy vấn SQL duy nhất cho câu hỏi bên dưới.

    Schema:
    {schema}
    {tuning_tip}
    Câu hỏi: {question}
    Câu lệnh SQL:"""


# ============================ BƯỚC 5: CÁC NÚT LANGGRAPH ============================
def ask_llm_node(state: SQLState) -> dict:
    prompt = build_sql_prompt(state)
    sql = call_llm(prompt)
    return {
        **state,
        "sql_query": sql
    }

def run_sql_node(state: SQLState) -> dict:
    result, error = execute_sql_query(state["sql_query"])
    return {
        **state,
        "result": result,
        "error": error
    }

def check_sql_error_node(state: SQLState) -> dict:
    return {
        **state,
        "__output__": "has_error" if state.get("error") else "no_error"
    }

def retry_sql_node(state: SQLState) -> dict:
    question = state["question"]
    sql_query = state["sql_query"]
    error = state["error"]

    # Kiểm tra trong lịch sử lỗi trước
    remembered_fix = check_and_fix_sql_error(state["question"], state["sql_query"], state["error"])
    if remembered_fix:
        print("♻️ Dùng lại truy vấn đã sửa trước đó.")
        return {
            **state,
            "sql_query": remembered_fix
        }

    # Nếu không có bản sửa trước -> gọi lại LLM để tạo SQL mới
    prompt = f"""Lỗi SQL: {error}

Câu hỏi: {question}
SQL sai: {sql_query}

💡 Hãy viết lại câu SQL đúng hơn:"""
    new_sql = call_llm(prompt)

    # Chạy thử new_sql
    new_result, new_error = execute_sql_query(new_sql)

    # Nếu vẫn lỗi → KHÔNG thêm vào bộ nhớ
    if new_error:
        print("❌ Truy vấn sửa vẫn sai, sẽ tiếp tục retry.")
        history = state.get("history", [])
        history.append((new_sql, new_error))

        return {
            **state,
            "sql_query": new_sql,
            "error": new_error,
            "result": None,
            "retry_count": state["retry_count"] + 1,
            "history": history
        }

    # ✅ Nếu không lỗi → thêm vào memory + cập nhật kết quả
    add_sql_error_entry(question, sql_query, error, new_sql)

    history = state.get("history", [])
    history.append((new_sql, error))  # Lưu lỗi cũ + SQL mới đã fix được

    return {
        **state,
        "sql_query": new_sql,
        "result": new_result,
        "error": None,
        "retry_count": state["retry_count"] + 1,
        "history": history
    }

def validate_result_node(state: SQLState) -> dict:
    return {
        **state,
        "__output__": "valid" if state.get("result") else "invalid"
    }

def handle_feedback_node(state: SQLState) -> dict:
    # Nhận phản hồi từ người dùng sau khi truy vấn SQL
    print("\n🔍 Bạn có hài lòng với kết quả không?")
    print("1: Hài lòng ✅  |  0: Không hài lòng ❌")
    try:
        feedback = int(input("→ Nhập phản hồi (1/0): ").strip())
    except:
        feedback = 0  # Mặc định là không hài lòng nếu không nhập hợp lệ

    if "feedback_score" not in state:
        state["feedback_score"] = 0
    if "feedback_count" not in state:
        state["feedback_count"] = 0

    if feedback == 1:
        state["feedback_score"] += 1
    else:
        state["feedback_score"] -= 1
        state["feedback_count"] += 1

    return state

def exit_node(state: SQLState) -> dict:
    return state

# ============================ BƯỚC 6: XÂY DỰNG LANGGRAPH ============================
graph = StateGraph(SQLState)

graph.add_node("ask_llm", ask_llm_node)
graph.add_node("run_sql", run_sql_node)
graph.add_node("check_sql_error", check_sql_error_node)
graph.add_node("retry_sql", retry_sql_node)
graph.add_node("validate_result", validate_result_node)
graph.add_node("handle_feedback", handle_feedback_node)
graph.add_node("exit", exit_node)

graph.set_entry_point("ask_llm")
graph.add_edge("ask_llm", "run_sql")
graph.add_edge("run_sql", "check_sql_error")

graph.add_conditional_edges("check_sql_error", {
    "has_error": retry_sql_node,
    "no_error": validate_result_node
})

graph.add_conditional_edges("validate_result", {
    "valid": handle_feedback_node,
    "invalid": retry_sql_node
})

graph.add_edge("handle_feedback", "exit")
graph.add_edge("retry_sql", "run_sql")
graph.set_finish_point("exit")

app = graph.compile()

# ============================ BƯỚC 7: CHẠY AGENT ============================
def run_agent(question: str):
    state: SQLState = {
        "question": question,
        "sql_query": None,
        "result": None,
        "error": None,
        "retry_count": 1,
        "history": [],
        "feedback_score": -1,
        "feedback_count": 1
    }
    result = app.invoke(state)

    # Nếu feedback_count >= 3 → có thể gọi logic tự học ở đây (gợi ý)
    if result["feedback_count"] >= 3 or result["feedback_score"] < 0:
        print("🎯 Đang áp dụng prompt tuning do hiệu suất chưa tốt.")

    return result

