# langgraph_agent.py

from langgraph.graph import StateGraph, END
from typing import Dict, TypedDict, List, Optional
from threading import Thread
import os, sys, time
from langchain.tools.tavily_search import TavilySearchResults
from langchain_core.tools import Tool
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

# Các module đã có
# Thêm đường dẫn thư mục cha của AI_Agent vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_utils import *
from RAG_Process.rag_process import *

# ===== Load Model & Search Tool =====
model, tokenizer = load_model()
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
# Load biến môi trường từ file .env
load_dotenv()
## Set up API key
TAVILY_API_KEY=os.environ.get("TAVILY_API_KEY")
## Set up search tool
search_tool = TavilySearchResults(tavily_api_key=TAVILY_API_KEY, max_results=2)

# ===== Agent State Type =====
class AgentState(TypedDict):
    question: str
    context: str
    answer: str
    feedback: str
    feedback_history: List[Dict[str, str]] # NEW: lưu danh sách phản hồi dạng {"feedback": ..., "classification": ...}
    index_static: int
    index_realtime: int
    memory: List[str]
    prompt_version: Optional[int]

# ===== Node: Search Tool (ReAct style) =====
def search_node(state: AgentState) -> AgentState:
    query = state["question"]
    results = search_tool.invoke({"query": query})
    docs = [r["content"] for r in results]
    context = "\n".join(docs)
    print("🌐 Search Tool Context Loaded")
    return {"context": context}

# ====== Node: Dual Retriever (Real-Time + Static) ======
def retrieve_dual_node(state: AgentState) -> AgentState:
    query = state["question"]
    realtime_docs, index_rt = realtime_retriever(query)
    static_docs, index_st = static_retriever(query)
    print(f"Real-time index: {index_rt} - Static index: {index_st}")

    # Chọn theo độ ưu tiên
    if index_rt == 4 and index_st == 4:
        print("❌ Không tìm thấy thông tin.")
        return {"context": [], "index_static": 4, "index_realtime": 4}

    if index_rt < index_st:
        context_docs = realtime_docs
    elif index_st < index_rt:
        context_docs = static_docs
    else:
        merged_docs = realtime_docs + static_docs
        # Rerank nếu có
        if reranker:
            pairs = [[query, doc.page_content] for doc in merged_docs]
            scores = reranker.compute_score(pairs, normalize=True)
            for i in range(len(merged_docs)):
                merged_docs[i].metadata['score'] = scores[i]
            merged_docs = sorted(merged_docs, key=lambda x: x.metadata['score'], reverse=True)
        context_docs = merged_docs[:5]  # lấy top 5

    # Gộp văn bản lại
    context_str = "\n".join([doc.page_content for doc in context_docs])
    return {
        "context": context_str,
        "index_static": index_st,
        "index_realtime": index_rt
    }

# ===== Node: Generate LLM Response =====
def generate_llm_node(state: AgentState) -> AgentState:
    question = state["question"]
    context = state["context"]
    memory = state.get("memory", [])
    prompt_version = state.get("prompt_version", 1)
    feedback_history = state.get("feedback_history", [])

    # Nếu có nhiều phản hồi tiêu cực, nhắc nhở LLM rõ ràng hơn
    recent_feedbacks = feedback_history[-2:] if feedback_history else []
    bad_count = sum(1 for fb in recent_feedbacks if fb["classification"] == "tiêu cực")
    # Gắn thêm cảnh báo nếu người dùng từng không hài lòng
    if bad_count >= 2:
        last_feedback = recent_feedbacks[-1]["feedback"] if recent_feedbacks else ""
        context += (
            "\n**Lưu ý đặc biệt cho trợ lý:**\n"
            "- Người dùng đã không hài lòng với các phản hồi trước.\n"
            f"- Nội dung phản hồi gần nhất: \"{last_feedback}\"\n"
            "- Hãy đảm bảo trả lời rõ ràng, cụ thể, và tránh phỏng đoán nếu thiếu dữ liệu.\n"
        )

    system_prompt = """
        Bạn là ViVi – một nhà quy hoạch giao thông đô thị thông minh.
        🎯 Vai trò: Phân tích dữ liệu & đề xuất giải pháp quy hoạch giao thông đô thị.
        📌 Lưu ý: Không phỏng đoán nếu thiếu thông tin. Luôn rõ ràng, dựa trên dữ liệu.
    """

    reasoning_part = """
        Hãy suy nghĩ từng bước, đánh giá dữ liệu, lập kế hoạch nếu cần. Dạng: (Suy nghĩ -> Hành động -> Quan sát).
    """

    prompt_v1 = f"""
        # Câu hỏi:
        {question}

        # Dữ liệu:
        {context}

        # Lịch sử:
        {" | ".join(memory[-3:]) if memory else "Không có."}

        {reasoning_part if prompt_version > 1 else ""}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_v1}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True
    ).to("cuda")

    attention_mask = (inputs != tokenizer.pad_token_id).long()
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)

    generation_kwargs = {
        "inputs": inputs,
        "streamer": streamer,
        "max_new_tokens": 512,
        "temperature": 1.2,
        "top_p": 0.95,
        "min_length": 30,
        "attention_mask": attention_mask
    }

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    full_response = ""
    for chunk in streamer:
        if "<|eot_id|>" in chunk:
            chunk = chunk.replace("<|eot_id|>", "")
        full_response += chunk
        print(chunk, end="", flush=True)
        time.sleep(0.01)

    return {"answer": full_response}

# ===== Node: Evaluate Feedback & Prompt Repair =====
def rewrite_prompt_with_llm(old_question, feedback):
    prompt = (
        "Bạn là một chuyên gia NLP, chuyên sửa câu hỏi không rõ ràng.\n"
        f"Người dùng hỏi: \"{old_question}\"\n"
        f"Phản hồi của họ: \"{feedback}\"\n"
        "→ Viết lại câu hỏi sao cho rõ ràng, chính xác và dễ hiểu hơn."
    )

    response = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)[0]["generated_text"]
    rewritten = response.replace(prompt, "").strip()  # Loại bỏ prompt đầu nếu cần
    return rewritten

def feedback_node(state: AgentState) -> AgentState:
    feedback = state.get("feedback", "").lower()
    memory = state.get("memory", [])
    answer = state.get("answer", "")
    question = state["question"]
    prompt_version = state.get("prompt_version", 1)
    feedback_history = state.get("feedback_history", [])

    print(f"📨 Feedback: {feedback}")
    # Prompt để nhờ LLM đánh giá phản hồi
    feedback_prompt = f"""
        Bạn là một chuyên gia đánh giá phản hồi người dùng.
        Câu hỏi: {question}
        Câu trả lời: {answer}
        Phản hồi của người dùng: {feedback}
        → Phản hồi là tích cực hay tiêu cực? Trả lời một từ: "tích cực" hoặc "tiêu cực".
    """

    result = pipe(feedback_prompt, max_new_tokens=10, do_sample=False)[0]["generated_text"]
    classification = result.replace(feedback_prompt, "").strip().lower()
    print(f"🤖 LLM đánh giá phản hồi: {classification}")

    # Cập nhật feedback history
    feedback_history.append({"feedback": feedback, "classification": classification})

    if "tiêu cực" in classification:
        print("❗ Phản hồi cho biết trả lời sai → Ghi log & sinh prompt mới")
        with open("AI_Agent/log_failed_cases.txt", "a", encoding="utf-8") as f:
            f.write(f"[❌] Q: {state['question']}\nA: {answer}\nFeedback: {feedback}\n---\n")
        
        # Gợi ý câu hỏi viết lại từ LLM nếu phản hồi quá nhiều lần
        prompt_version += 1
        if prompt_version >= 3:
            print("⚠️ Liên tiếp trả lời sai. Chỉnh sửa method tiếp cận để solve problem")
            # Tự động sửa câu hỏi bằng LLM (Prompt Repair)
            rewritten_question = rewrite_prompt_with_llm(state['question'], feedback)
            print("🔁 Prompt mới được tạo lại:", rewritten_question)
            return {
                "question": rewritten_question,
                "memory": memory + [question],
                "prompt_version": prompt_version,
                "feedback_history": feedback_history,
                "next_step": "generate"
            }
        
        return {
            "memory": memory + [question],
            "prompt_version": prompt_version,
            "feedback_history": feedback_history,
            "next_step": "generate"  # ✅ đây là điều kiện branch
        }
    
    else:
        print("✅ Phản hồi tốt.")
        return {
            "feedback_history": feedback_history,
            "next_step": "end" # ✅ mặc định là kết thúc
        }  

# ===== LangGraph: Build the Agent =====
graph = StateGraph(AgentState)

graph.add_node("search", search_node)
graph.add_node("retrieve", retrieve_dual_node)
graph.add_node("generate", generate_llm_node)
graph.add_node("feedback", feedback_node)

# Entry → retrieve (RAG) → generate → feedback → END
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "feedback")
graph.add_conditional_edges("feedback", lambda state: state.get("next_step", "end"), {
    "generate": "generate",
    "end": END
})

agent_app = graph.compile()
