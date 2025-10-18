# AI Dev Helper Chatbot

Chatbot hỗ trợ lập trình Python sử dụng OpenAI API.

## 🚀 Cách sử dụng nhanh

### Bước 1: Cấu hình API Key
Có 2 cách để cấu hình API key:

**Cách 1: Thay trực tiếp trong code (đơn giản)**
1. Mở file `main.py`
2. Tìm dòng `API_KEY = "your-openai-api-key-here"`
3. Thay thế `your-openai-api-key-here` bằng API key thực của bạn

**Cách 2: Sử dụng file .env (bảo mật hơn)**
1. Sao chép file `.env.example` thành `.env`
2. Mở file `.env` và thay thế API key

### Bước 2: Chạy chatbot
```cmd
python main.py
```

## 📋 Yêu cầu
- Python 3.7+
- Packages: `openai`, `python-dotenv`

## 🔧 Cài đặt packages
```cmd
python -m pip install openai python-dotenv
```

## 💡 Lưu ý
- API key có thể lấy từ: https://platform.openai.com/api-keys
- Chatbot sử dụng model `gpt-3.5-turbo` để tiết kiệm chi phí
- Gõ 'exit' hoặc 'thoát' để kết thúc chương trình