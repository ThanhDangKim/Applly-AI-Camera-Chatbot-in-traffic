#### AI_Agent
# from AI_Agent.langgraph_agent import *
# if __name__ == "__main__":
#     state = {
#         "question": "Làm sao để giảm ùn tắc giao thông tại ngã tư lớn giờ cao điểm?",
#         "feedback": "Trả lời chưa đúng trọng tâm, còn bị dài dòng",  # có thể cập nhật sau nếu cần
#         "memory": [],
#         "prompt_version": 2
#     }

#     result = agent_app.invoke(state)
#     print("\n🤖 Trả lời cuối cùng:", result["answer"])

''''''
#### SQL Agent
from SQL_Agent.sql_agent_langgraph import *
if __name__ == "__main__":
    query = "Ngày 6 tháng 6 tại camera đường Xa lộ Hà Nội giao nhau với Lê Văn Việt có tổng hợp tất cả bao nhiêu lượt xe với vận tốc trung bình bao nhiêu?"
    final = run_agent(query)
    print("✅ Câu SQL tạo ra:")
    print(final["sql_query"])
    print("✅ Kết quả truy vấn:")
    print(final["result"])
    print("📝 Lịch sử sửa lỗi:")
    for i, (sql, err) in enumerate(final["history"]):
        print(f"Lần {i+1}:\nSQL:", sql, "\nError:", err)

    print("📊 Feedback tích lũy:", final["feedback_score"])
    print("❌ Feedback tiêu cực:", final["feedback_count"])
