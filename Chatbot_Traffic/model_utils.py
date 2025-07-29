from threading import Thread
from transformers import TextIteratorStreamer
import torch
import time
import os 
from transformers import AutoTokenizer
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
import pickle
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from FlagEmbedding import FlagReranker
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import BitsAndBytesConfig
from langchain.schema import Document
import json
import pdfplumber
import unidecode
from uuid import uuid4
from ExportDataFromDB import *
import re
import shelve

# Xử lý PDF
# ✅ Normalize filenames for consistency
def normalize_filename(filename):
    filename = unidecode.unidecode(filename)
    filename = filename.replace(" ", "_")
    return filename

# ✅ Remove page numbers from text
def remove_page_numbers(text):
    # Remove lines that only contain numbers (likely page numbers)
    lines = text.split('\n')
    cleaned_lines = [line for line in lines if not re.match(r'^\s*\d+\s*$', line.strip())]
    return "\n".join(cleaned_lines)

# ✅ Clean and merge fragmented lines
def clean_and_merge_lines(lines):
    merged_lines = []
    buffer = ""
    for line in lines:
        stripped_line = line.strip()
        if buffer:
            if stripped_line and not stripped_line[0].isupper():
                buffer += " " + stripped_line  # Merge fragmented lines
            else:
                merged_lines.append(buffer)
                buffer = stripped_line
        else:
            buffer = stripped_line
    if buffer:
        merged_lines.append(buffer)
    return merged_lines

# ✅ Read a PDF file using pdfplumber and apply all processing
def read_pdf_with_plumber(file_path):
    content = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text = remove_page_numbers(text)
                    content += text + "\n"
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
    # Xử lý ghép dòng
    lines = content.split("\n")
    merged_lines = clean_and_merge_lines(lines)
    return "\n".join(merged_lines)

# ✅ Read all PDF files in a folder and return a dictionary with normalized filename and cleaned content
def read_all_pdfs_with_plumber(folder_path):
    all_contents = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            content = read_pdf_with_plumber(file_path)
            normalized_name = normalize_filename(os.path.splitext(filename)[0])
            all_contents[normalized_name] = content
    return all_contents

def load_documents_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    documents = []
    for item in data:
        doc = Document(
            page_content=item["content"],
            metadata={
                "source": item["filename"],
                "title": f"Tiêu đề: {item['title']}",
                "category": item["category"]
            }
        )
        documents.append(doc)
        print("Đã load PDF")
    return documents

# Biến đổi nội dung PDF thành các Document của LangChain
def extract_title_from_content(content):
    for line in content.split("\n"):
        if len(line.strip()) > 10 and line.strip().istitle():
            return line.strip()
    return None

def extract_main_topic(content):
    topic_patterns = [
        r"Chủ đề[:\-]\s*(.+)",
        r"Topic[:\-]\s*(.+)",
        r"Category[:\-]\s*(.+)",
        r"Chương\s+\d+[:\-]?\s*(.+)",
        r"Phần\s+\d+[:\-]?\s*(.+)"
    ]
    for line in content.split('\n')[:20]:  # Chỉ quét 20 dòng đầu
        for pattern in topic_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match.group(1).strip()
    return "Unknown"

def convert_to_documents(pdf_dict):
    documents = []
    for filename, content in pdf_dict.items():
        title = extract_title_from_content(content) or filename.replace("_", " ")
        main_topic = extract_main_topic(content)
        doc = Document(
            page_content=content,
            metadata={
                "source": filename,
                "title": f"Tiêu đề: {title}",
                "category": main_topic
            }
        )
        documents.append(doc)
    return documents
'''------------------------------------------------'''


# Đọc tệp JSON
def load_json_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

# Chuyển dữ liệu từ JSON thành định dạng Document của LangChain
def ensure_string(value):
    if isinstance(value, list):
        # Nếu là danh sách dict, chuyển dict thành chuỗi
        return "\n".join(
            [json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else str(v) for v in value]
        )
    elif isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    elif isinstance(value, str):
        return value
    return str(value)  # fallback

# Chuyển dữ liệu từ JSON thành định dạng Document của LangChain
def convert_json_to_documents(json_data):
    documents = []

    # Duyệt qua các disease trong dữ liệu JSON
    for main_topic, items in json_data.items():
        for item in items:
            title = item.get("title", "")
            description = ensure_string(item.get("description", ""))
            causes = ensure_string(item.get("causes", ""))
            mechanism = ensure_string(item.get("mechanism", ""))
            meaning = ensure_string(item.get("meaning", ""))

            # Tạo content và metadata cho mỗi Document
            content = "Mô tả chủ đề: " + description + "\nNguyên nhân: " + causes + "\nPhương pháp thực hiện: " + mechanism + "\nÝ nghĩa của hành vi:" + meaning
            metadata = {
                "title": f"Tiêu đề: {title}",
                "category": main_topic
            }

            # Tạo Document
            document = Document(page_content=content, metadata=metadata)
            documents.append(document)

    return documents

# Hàm để load tất cả các tệp JSON trong thư mục
def load_all_json_files_in_directory(directory_path):
    documents = []
    # Duyệt qua tất cả các tệp trong thư mục
    for filename in os.listdir(directory_path):
        # Kiểm tra nếu tệp có phần mở rộng là .json
        if filename.endswith('.json'):
            json_file_path = os.path.join(directory_path, filename)
            # Đọc và chuyển tệp JSON thành documents
            json_data = load_json_data(json_file_path)
            documents.extend(convert_json_to_documents(json_data))
            print(f"Loaded: {filename}")
    return documents
'''------------------------------------------------'''


# Hàm xử lý file TXT
def parse_txt_to_documents(raw_text):
    documents = []
    current_category = None
    lines = raw_text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Phát hiện đề mục
        if line.startswith("===") and line.endswith("==="):
            current_category = line.strip("= ").strip()
            continue

        # Tìm tên vị trí (tên đường + camera)
        match = re.search(r"camera.*?tại (Xa lộ Hà Nội.*?Đường [^,–-]*)", line)
        if not match:
            match = re.search(r"tại (Xa lộ Hà Nội.*?Đường [^,–-]*)", line)
        title_location = match.group(1).strip() if match else "Không rõ vị trí"

        # Ghép thành title hoàn chỉnh
        full_title = f"{current_category} tại vị trí {title_location}"

        # Tạo document
        doc = Document(
            page_content=line,
            metadata={
                "title": f"Tiêu đề: {full_title}",
                "category": current_category or "Không rõ chủ đề"
            }
        )
        documents.append(doc)

    return documents

# ✅ Đọc toàn bộ file .txt trong thư mục
def load_documents_from_directory(directory):
    all_documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                raw_text = f.read()
            docs = parse_txt_to_documents(raw_text)  # <-- raw_text, không phải filepath
            all_documents.extend(docs)
    return all_documents
'''------------------------------------------------'''


# Xử lý chunk
def load_chunk(documents_path):
    # Tải lại tất cả các Document từ tệp pickle
    with open(documents_path, "rb") as f:
        list_of_chunks = pickle.load(f)
    
    return list_of_chunks

def load_embedding_model():
    model_name = "BAAI/bge-base-en-v1.5"
    # model_kwargs = {'device': 'cpu'}
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': False} # Chuẩn hóa vector = True sẽ limit length vector = 1

    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    return hf_embeddings

def semantic_chunk_real_time(hf_embeddings, documents):
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
    text_splitter = SemanticChunker(hf_embeddings,
                                breakpoint_threshold_type="percentile",
                                breakpoint_threshold_amount=85)
    # Split the documents and count the number of chunks in each document, as well as the number of tokens in each chunk
    count = 0
    total = 0
    list_of_chunks = []
    for idx,doc in enumerate(documents):
        chunks = text_splitter.create_documents([doc.page_content])
        # print(f'Number of chunks: {len(chunks)} - Tokens of each chunk',end=' ')
        for chunk in chunks:
            text = chunk.page_content
            tokens = tokenizer.tokenize(text)
            num_tokens = len(tokens)
            if num_tokens > 1:
                total = total + 1
                # Use the parent document index as metadata to retrieve the parent document from the child chunk.
                chunk.metadata['parent'] = idx
                list_of_chunks.append(chunk)
            if num_tokens > 512:
                count = count + 1
            # print(num_tokens, end =' ')
        # print()
    print('Toltal chunks: ',total)
    # print('Number of chunks which is larger than 512 tokens: ',count)
    return list_of_chunks

def build_realtime_faiss_index(hf_embeddings, txt_chunks):
    uuids = [str(uuid4()) for _ in range(len(txt_chunks))]
    index = faiss.IndexFlatL2(len(hf_embeddings.embed_query("hello world")))

    vector_store = FAISS(
        embedding_function=hf_embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    vector_store.add_documents(documents=txt_chunks, ids=uuids)
    
    return vector_store, txt_chunks

def save_to_vector_memory(hf_embeddings, db_path, question, answer):
    vector_store = FAISS.load_local(db_path, embeddings=hf_embeddings)
    vector_store.add_texts([f"Q: {question}\nA: {answer}"])
    vector_store.save_local(db_path)
    
def load_Faiss_index(hf_embeddings, db_path):
    index = faiss.IndexFlatL2(len(hf_embeddings.embed_query("hello world")))
    vector_db_path = db_path

    # Tải lại FAISS index từ file
    vector_store = FAISS.load_local(vector_db_path, hf_embeddings, allow_dangerous_deserialization = True)
    return vector_store

def Bm25_Retriever(list_of_chunks):
    # Khởi tạo BM25 retriever với tham số tìm kiếm top 10 các kết quả liên quan nhất
    bm25_retriever = BM25Retriever.from_documents(
        list_of_chunks, k = 10
    )
    return bm25_retriever

def reranker():
    # Load the tokenizer for the BAAI/bge-m3 model
    reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
    return reranker

class Retriever:
    def __init__(self, semantic_retriever, bm25_retriever, reranker, documents):
        self.semantic_retriever = semantic_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        self.documents = documents

    def __call__(self, query):
        semantic_results = self.semantic_retriever.similarity_search(
            query,
            k=10,
        )
        bm25_results = self.bm25_retriever.invoke(query)

        content = set()
        retrieval_docs = []

        for result in semantic_results:
            if result.page_content not in content:
                content.add(result.page_content)
                retrieval_docs.append(result)

        for result in bm25_results:
            if result.page_content not in content:
                content.add(result.page_content)
                retrieval_docs.append(result)

        pairs = [[query, doc.page_content] for doc in retrieval_docs]

        scores = self.reranker.compute_score(pairs, normalize = True)

        # Lấy tài liệu nguồn từ phần tử con dựa trên điểm số ngưỡng
        context_1 = []
        context_2 = []
        context_3 = []
        context = []
        index = None
        parent_ids = set()
        for i in range(len(retrieval_docs)):
            # Điểm liên quan >= 0.75 sẽ được sử dụng làm kiểu ngữ cảnh 1 (chỉ ra sự liên quan cao hơn đối với truy vấn).
            if scores[i] >= 0.75:
                parent_idx = retrieval_docs[i].metadata['parent']
                if parent_idx not in parent_ids:
                    parent_ids.add(parent_idx)
                    context_1.append(self.documents[parent_idx])

            # Điểm liên quan >= 0.60 sẽ được sử dụng làm kiểu ngữ cảnh 2 (chỉ ra sự liên quan trung bình đến thấp đối với truy vấn).
            elif scores[i] >= 0.60:
                parent_idx = retrieval_docs[i].metadata['parent']
                if parent_idx not in parent_ids:
                    parent_ids.add(parent_idx)
                    context_2.append(self.documents[parent_idx])

            elif scores[i] >= 0.1:
                parent_idx = retrieval_docs[i].metadata['parent']
                if parent_idx not in parent_ids:
                    parent_ids.add(parent_idx)
                    context_3.append(self.documents[parent_idx])
        
        if len(context_1) > 0:
            # print('Context 1')
            context = context_1
            index = 1

        elif len(context_2) > 0:
            # print('Context 2')
            context = context_2
            index = 2

        elif len(context_3) > 0:
            # print('Context 3')
            context = context_3
            index = 3

        else:
            index = 4
            # Nếu điểm liên quan < 0.1, điều này chỉ ra rằng không có tài liệu liên quan.
            print('No relevant context')

        return context, index

def load_model(model_ori="model/4BIT00006"):
    # Đường dẫn tới mô hình và tokenizer
    model_path = model_ori

    # Tham số lượng tử hóa mô hình
    use_4bit, bnb_4bit_compute_dtype, bnb_4bit_quant_type, use_nested_quant = True, "float16", "nf4", False 
    device_map = {"": 0}
    # Tải tokenizer và mô hình với tham số QLoRA 
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Thiết lập pad token thành eos_token (có 2 trường hợp)
        ## Lựa chọn 1: Thiết lập eos_token thành pad_token
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token 

        # Hoặc, lựa chọn 2: Thêm một padding token mới
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=device_map
        )
    # model.config.use_cache = False
    # model.config.pretraining_tp = 1
    return model, tokenizer

# Kiểm tra với Retriever 
def get_answer_with_two_retrievers(query, realtime_retriever, static_retriever, reranker=None, top_k=5):
    # Sử dụng retriever để lấy context (dữ liệu từ các tài liệu liên quan)
    
    realtime_docs, index_real_time = realtime_retriever(query)
    static_docs, index_static = static_retriever(query)
    if index_real_time == 4 and index_static == 4:
        print("Dừng hàm get_answer_with_two_retrievers")
        return
    
    if int(index_real_time) < int(index_static):
        all_docs = realtime_docs
    elif int(index_static) < int(index_real_time):
        all_docs = static_docs
    else:
        # Gộp kết quả, nhưng đặt realtime lên trước
        all_docs = realtime_docs + static_docs

        # Rerank nếu có
        if reranker:
            pairs = [[query, doc.page_content] for doc in all_docs]
            scores = reranker.compute_score(pairs, normalize=True)
            # Gắn điểm vào tài liệu để sắp xếp
            for i in range(len(all_docs)):
                all_docs[i].metadata['score'] = scores[i]
            # Sắp xếp giảm dần theo điểm
            all_docs = sorted(all_docs, key=lambda x: x.metadata['score'], reverse=True)

        # Cắt lấy top_k kết quả sau khi rerank (hoặc đơn giản là ưu tiên real-time)
        selected_docs = all_docs[:top_k] if len(all_docs) >= top_k else all_docs
        all_docs = selected_docs

    # Nếu có dữ liệu thì sinh câu trả lời
    if all_docs:
        return all_docs
    else:
        return "Không tìm thấy thông tin liên quan để trả lời câu hỏi."
    
def generate_response_streaming_from_prompt_format(model, tokenizer, user_input, context):

    system_prompt = f"""Bạn là ViVi – một nhà quy hoạch giao thông đô thị thông minh.

    🎯 Vai trò chính:
    Bạn là chuyên gia trong lĩnh vực quy hoạch hạ tầng giao thông đô thị, có khả năng tự phân tích dữ liệu thực tế và đưa ra các giải pháp tối ưu hóa mạng lưới giao 
    thông, giảm ùn tắc và tăng cường an toàn.

    📊 Kỹ năng chuyên môn:
    - Phân tích dữ liệu giao thông: vận tốc, lưu lượng xe, mật độ, điểm nóng ùn tắc và tai nạn.
    - Xây dựng và đề xuất quy hoạch: thiết kế tuyến đường, mạng lưới kết nối, giao lộ, bãi đỗ xe, hạ tầng dành cho người đi bộ và xe đạp, hệ thống giao thông thông minh 
    (ITS).
    - Hỗ trợ chính sách: đề xuất các chính sách quản lý giao thông, thu phí, phân luồng, điều tiết theo thời gian thực.
    - Tư duy định hướng phát triển bền vững, lấy người tham gia giao thông làm trung tâm.

    🔍 Phong cách trả lời:
    - Chính xác, có căn cứ dữ liệu rõ ràng
    - Phân tích chuyên sâu nhưng dễ hiểu
    - Không trả lời nếu thiếu thông tin, tránh suy đoán vô căn cứ

    ❗ Lưu ý quan trọng:
    Nếu không có đủ dữ liệu hoặc thông tin rõ ràng, hãy nói rõ điều đó. Tuyệt đối không phỏng đoán hoặc bịa đặt.
    """

    
    formatted_prompt = f"""
        Bạn là một trợ lý giao thông thông minh. Dựa vào các thông tin đã được thu thập, hãy trả lời chính xác câu hỏi của người dùng.
        # Câu hỏi:
        {user_input}

        # Dữ liệu liên quan:
        {context}
    """
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": formatted_prompt}]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    attention_mask = (inputs != tokenizer.pad_token_id).long()
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    generation_kwargs = {
        "inputs": inputs,
        "streamer": streamer,
        "max_new_tokens": 512,
        "temperature": 1.5,
        "min_length": 30,  
        "use_cache": True,
        "top_p": 0.95,
        "min_p": 0.1,
        "attention_mask" : attention_mask,
    }
    
    # Start the generation in a separate thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Stream the generated text
    for _, new_text in enumerate(streamer):
        if "<|eot_id|>" in new_text:
            new_text = new_text.replace("<|eot_id|>", "")
        print(new_text)
        yield new_text
        time.sleep(0.02)