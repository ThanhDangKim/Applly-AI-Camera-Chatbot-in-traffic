import re
import json
import os
import psycopg2
from difflib import SequenceMatcher
from langgraph.graph import StateGraph, END
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional

# ============================ BƯỚC 1: CẤU HÌNH MÔ HÌNH ============================
model_name = "bigcode/starcoder2-3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

def call_llm(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    match = re.search(r"(SELECT|WITH|INSERT|UPDATE|DELETE)[\s\S]+?;", text, re.IGNORECASE)
    return match.group(0) if match else text.strip()

# ============================ BƯỚC 2: ĐỊNH NGHĨA TRẠNG THÁI ============================
class SQLState:
    def __init__(self, question, sql_query=None, result=None, error=None, retry_count=0):
        self.question = question
        self.sql_query = sql_query
        self.result = result
        self.error = error
        self.retry_count = retry_count
        self.history = []

# ============================ BƯỚC 3: MEMORY LƯU LỊCH SỬ LỖI ============================
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
            dbname="your_db", user="your_user", password="your_pass", host="localhost", port="5432"
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
### BẢNG users
- id (PK)
- username, password, full_name, role, created_at

### BẢNG cameras
- id (PK), name, location, installed

### BẢNG vehicle_stats
- id (PK), camera_id (FK), date (DATE), time_slot (0–47), direction, vehicle_type, vehicle_count

### BẢNG avg_speeds
- id (PK), camera_id (FK), date (DATE), time_slot (0–47), average_speed

### BẢNG camera_area
- id (PK), camera_id (FK), area_id (FK), location_detail

### BẢNG areas
- id (PK), name, description

### BẢNG daily_traffic_summary
- id (PK), camera_id (FK), date (DATE), total_vehicle_count, avg_speed, peak_time_slot, direction_with_most_traffic

### BẢNG traffic_events
- id (PK), camera_id (FK), event_time, event_type, description
"""

def build_sql_prompt(question: str):
    schema = get_schema_text()
    return f"""Bạn là chuyên gia SQL. Hãy viết truy vấn SQL duy nhất cho câu hỏi bên dưới.

Schema:
{schema}

❗Chỉ sử dụng các cột có thật trong schema.
Câu hỏi: {question}
Câu lệnh SQL:"""

# ============================ BƯỚC 5: CÁC NÚT LANGGRAPH ============================
def ask_llm_node(state: SQLState):
    prompt = build_sql_prompt(state.question)
    sql = call_llm(prompt)
    state.sql_query = sql
    return state

def run_sql_node(state: SQLState):
    result, error = execute_sql_query(state.sql_query)
    state.result = result
    state.error = error
    return state

def check_sql_error_node(state: SQLState):
    if state.error:
        return "has_error"
    return "no_error"

def retry_sql_node(state: SQLState):
    # Kiểm tra trong lịch sử lỗi trước
    remembered_fix = check_and_fix_sql_error(state.question, state.sql_query, state.error)
    if remembered_fix:
        state.sql_query = remembered_fix
    else:
        prompt = f"Lỗi SQL: {state.error}\n\nCâu hỏi: {state.question}\nSQL sai: {state.sql_query}\n\nHãy viết lại SQL đúng hơn:"
        new_sql = call_llm(prompt)
        add_sql_error_entry(state.question, state.sql_query, state.error, new_sql)
        state.sql_query = new_sql
    state.retry_count += 1
    state.history.append((state.sql_query, state.error))
    return state

def validate_result_node(state: SQLState):
    if state.result and len(state.result) > 0:
        return "valid"
    return "invalid"

def exit_node(state: SQLState):
    return state

# ============================ BƯỚC 6: XÂY DỰNG LANGGRAPH ============================
graph = StateGraph(SQLState)

graph.add_node("ask_llm", ask_llm_node)
graph.add_node("run_sql", run_sql_node)
graph.add_node("check_sql_error", check_sql_error_node)
graph.add_node("retry_sql", retry_sql_node)
graph.add_node("validate_result", validate_result_node)
graph.add_node("exit", exit_node)

graph.set_entry_point("ask_llm")
graph.add_edge("ask_llm", "run_sql")
graph.add_edge("run_sql", "check_sql_error")

graph.add_conditional_edges("check_sql_error", {
    "has_error": "retry_sql",
    "no_error": "validate_result"
})

graph.add_conditional_edges("validate_result", {
    "valid": "exit",
    "invalid": "retry_sql"
})

graph.add_edge("retry_sql", "run_sql")
graph.set_finish_point("exit")

app = graph.compile()

# ============================ BƯỚC 7: CHẠY AGENT ============================
def run_agent(question: str):
    state = SQLState(question)
    result = app.invoke(state)
    return result

if __name__ == "__main__":
    query = "Ngày 6 tháng 6 tại camera đường Xa lộ Hà Nội giao nhau với Lê Văn Việt có tổng hợp tất cả bao nhiêu lượt xe với vận tốc trung bình bao nhiêu?"
    final = run_agent(query)
    print("✅ Câu SQL tạo ra:")
    print(final.sql_query)
    print("✅ Kết quả truy vấn:")
    print(final.result)
    print("📝 Lịch sử sửa lỗi:")
    for i, (sql, err) in enumerate(final.history):
        print(f"Lần {i+1}:\nSQL:", sql, "\nError:", err)
