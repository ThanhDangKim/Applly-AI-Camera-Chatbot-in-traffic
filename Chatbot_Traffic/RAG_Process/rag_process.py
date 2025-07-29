# Các module bạn đã có
import sys
import os
# Thêm đường dẫn thư mục cha của folder chứa file .py vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_utils import *

# Load models and data
hf_embedding = load_embedding_model()
list_of_chunks = load_chunk("vectorstore/db_document/documents.pkl")
vector_db_path = 'vectorstore/db_faiss'
json_directory = "Data_Traffic/JSON"
pdf_directory = "Data_Traffic/PDF"
txt_directory = "Data_Traffic/TXT"

##
'''Static RAG'''
pdf_documents = load_documents_from_json(os.path.join(pdf_directory, "processed_pdf_documents.json"))
json_documents = load_all_json_files_in_directory(json_directory)
documents = json_documents + pdf_documents
semantic_results = load_Faiss_index(hf_embedding, vector_db_path)
bm25_retriever = Bm25_Retriever(list_of_chunks)

static_retriever = Retriever(
    semantic_retriever=semantic_results,
    bm25_retriever=bm25_retriever,
    reranker=reranker(),
    documents=documents
)
print("Done static Retriever")

##
'''Real-time RAG'''
txt_documents = load_documents_from_directory(txt_directory)
txt_chunks = semantic_chunk_real_time(hf_embedding, txt_documents)
realtime_faiss, txt_chunks = build_realtime_faiss_index(hf_embedding, txt_chunks)
realtime_bm25 = Bm25_Retriever(txt_chunks)

realtime_retriever = Retriever(
    semantic_retriever=realtime_faiss,
    bm25_retriever=realtime_bm25,
    reranker=reranker(),
    documents=txt_documents
)
print("Done real - time Retriever")
