# Hướng dẫn cấu hình Pinecone

## Bước 1: Tạo tài khoản Pinecone

1. Truy cập https://www.pinecone.io/
2. Đăng ký tài khoản miễn phí (Free tier)
3. Sau khi đăng nhập, vào Dashboard

## Bước 2: Lấy API Key

1. Trong Pinecone Dashboard, click vào **API Keys** ở menu bên trái
2. Copy **API Key** của bạn
3. Lưu lại **Environment** (ví dụ: `us-east-1`, `gcp-starter`, etc.)

## Bước 3: Cập nhật file .env

Mở file `.env` và cập nhật các giá trị sau:

```env
# Pinecone Configuration
PINECONE_API_KEY=pcsk_XXXXXX_XXXXXXXXXXXXXXXXXXXXXXXXXX
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=code-assistant
```

**Lưu ý:**
- `PINECONE_API_KEY`: API key bạn vừa copy từ dashboard
- `PINECONE_ENVIRONMENT`: Region của Pinecone (thường là `us-east-1` cho free tier)
- `PINECONE_INDEX_NAME`: Tên index sẽ được tự động tạo (có thể giữ nguyên `code-assistant`)

## Bước 4: Kiểm tra cấu hình

Chạy lệnh test:

```bash
python test_rag.py
```

Nếu thành công, bạn sẽ thấy:
```
Using Pinecone vector database (index: code-assistant)...
Creating Pinecone index: code-assistant...
Index code-assistant created successfully!
✅ RAG service initialized successfully!
```

## Cấu hình hiện tại

- ✅ Vector Database: **Pinecone** (thay vì FAISS local)
- ✅ Embeddings: **HuggingFace sentence-transformers** (miễn phí)
- ✅ LLM: **GPT-4o-mini** từ Elevate AI Ready

## Ưu điểm của Pinecone

1. **Persistent storage**: Dữ liệu được lưu trên cloud, không mất khi restart
2. **Scalable**: Tự động scale theo nhu cầu
3. **Fast retrieval**: Tìm kiếm vector rất nhanh
4. **Multi-user**: Nhiều instance có thể dùng chung index
5. **Free tier**: 1 index, 100K vectors miễn phí

## Quay lại FAISS local

Nếu muốn quay lại dùng FAISS (không cần Pinecone):

```env
USE_PINECONE=false
```
