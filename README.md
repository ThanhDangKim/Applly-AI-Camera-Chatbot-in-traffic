# 📊 Intelligent Traffic Surveillance System with Integrated Chatbot

## 📌 Project Overview

This project is designed to address the challenges of **urban traffic monitoring and signal control** by combining **real-time computer vision** with **natural language AI**.

It consists of two main components:

- **Smart Traffic Camera System** – applies deep learning models to detect and track vehicles, analyze traffic flow, and dynamically control traffic lights. The model has been **fine-tuned on real-world data** and optimized for deployment using **ONNX, TensorRT, and OpenVINO**.

- **Traffic Chatbot with LLM + RAG** – allows users to query traffic data and urban planning documents using natural language. The chatbot uses a **fine-tuned LLAMA3.2‑3B model**, powered by a **Smart Retriever** for accurate and context-aware responses.

---

## 📷 1. Smart Traffic Camera System

The camera system processes real-time video input from intersections and applies the following techniques:

### 🎞️ Image and Video Processing
- Preprocessing using RGB/HSV color spaces, Gaussian blur, and Canny edge detection
- Geometric transformations: Homography and Perspective Transform

### 🧠 Vehicle Detection and Tracking
- Detection using **Faster R-CNN with FPN** (fine-tuned)
- Object tracking with **DeepSORT + TorchReID**

### 🕒 Speed Estimation and Signal Control
- Estimate speed using pixel-to-distance conversion (MPP) via perspective transform
- Apply **Exponential Moving Average (EMA)** to smooth vehicle count over time
- Use linear regression to detect cycle changes in traffic flow
- Automatically adjust signal timing (green/red lights) based on traffic density

### 🚀 Inference Optimization
- Model is converted and deployed using **ONNX**, **TensorRT**, and **OpenVINO** to enhance performance across devices

---

## 💬 2. Traffic Chatbot System (LLM + RAG)

The chatbot enables users to ask traffic or planning-related questions in natural language, and generates responses from:

- **Real-time traffic data** stored in PostgreSQL
- **Preprocessed urban planning documents** (PDF, JSON)

### 🧠 Urban Planning Document Processing
- Semantic chunking + embedding vectors for each paragraph
- Three retrieval strategies:
  - FAISS Semantic Index
  - BM25 + Pinecone
  - Reranker
- Combine results into a **Smart Retriever – Planning**, and pass context to the fine-tuned LLM for final answer generation

### 📡 Real-time Camera Data Query
- Periodically fetches data from PostgreSQL
- Creates a **Smart Retriever – Real-time** using the same 3-step strategy
- Compares semantic similarity between real-time and planning data to pick or combine the most relevant source

---

### 🔔 Note – `SQL_Agent/` Module

Inside the `chatbot/` folder, there's a subfolder named **`SQL_Agent/`**, which is a **custom-built module using LangGraph** that enhances the chatbot’s ability to **query the PostgreSQL database** efficiently and intelligently.

| **Component** | **Description** |
|---------------|------------------|
| **LLM-powered SQL generation** | Uses **StarCoder 2 3B** (via HuggingFace Transformers) to generate SQL based on schema and user questions |
| **LangGraph workflow** | Implements a **state graph workflow**: generate SQL ➜ execute ➜ detect/fix error ➜ re-run ➜ finalize |
| **Error memory (`sql_error_memory.json`)** | Stores previously seen SQL errors and corrections for reuse, reducing LLM calls and speeding up responses |
| **Auto-repair mechanism** | When an error occurs, the agent prompts the LLM with context to generate an improved SQL query (with retry limits) |
| **RAG integration** | Combines semantic search + BM25 + reranker to select relevant context for complex queries |
| **Results** | Returns the final SQL, query results, and a detailed history of any error-fixing steps for debugging |

> **Summary:** The `SQL_Agent/` module allows the chatbot to **automatically generate, execute, and fix SQL queries**, ensuring **fast and accurate data retrieval** from the real-time traffic database.

---

## ✅ Key Features

1. Ingest and process live video streams from traffic cameras  
2. Detect and classify vehicles using deep learning  
3. Track vehicles across video frames  
4. Segment movement direction for flow analysis  
5. Calculate traffic metrics: vehicle count, average speed, density  
6. Estimate signal control metrics (VTw and VTL) per direction  
7. Make real-time traffic light decisions automatically  
8. Log system states: vehicle count, speed, signal duration  
9. Provide API access for dashboards and external systems  
10. Support for edge-based or cloud-integrated multi-camera networks  
11. Chatbot support for natural language traffic queries  
12. LLM + RAG-based response generation from real-time and planning data sources  
