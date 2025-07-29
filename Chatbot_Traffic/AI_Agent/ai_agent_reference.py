import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.agents import AgentExecutor

import sys
# Thêm đường dẫn thư mục cha của AI_Agent vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_utils import load_model

# Load biến môi trường từ file .env
load_dotenv()

## Set up API key
TAVILY_API_KEY=os.environ.get("TAVILY_API_KEY")

## Set up search tool
search_tool = TavilySearchResults(tavily_api_key=TAVILY_API_KEY, max_results=2)

## Set up AI Agent with Search Tool Functionality
# Lấy đường dẫn tuyệt đối đến thư mục chứa model_utils.py
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "../model/4BIT00006")
model, tokenizer = load_model(model_path)

# 1. Create prompt with system prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Act as an AI Chatbot who is smart and friendly"),
    ("human", "{input}")
])

# 2. Create LLM chain with prompt
agent = create_react_agent(
    model=model,
    tools=[search_tool],
    prompt=prompt
)

# 4. Tạo executor từ agent
agent_executor = AgentExecutor(agent=agent, tools=[search_tool], verbose=True)

# 5. Đặt câu hỏi
query = "Tell me about the trends in crypto markets"
response = agent_executor.invoke({"input": query})

print(response["output"])  # hoặc print(response) tùy dạng output