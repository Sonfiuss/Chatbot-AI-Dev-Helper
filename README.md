# AI Dev Helper – Python Assistant (Web + CLI)

Trợ lý AI hỗ trợ lập trình Python. Dự án gồm 2 cách sử dụng:
- Web (Flask) với giao diện 2 phần: nhập code và chat (night theme)
- CLI (chạy trong terminal)

## Cấu trúc thư mục
```
Chatbot_AIDevHelper/
├─ app.py                 # Flask server
├─ main.py                # CLI chat (terminal)
├─ templates/
│  └─ index.html          # Giao diện web
├─ static/
│  ├─ style.css           # CSS (night theme)
│  └─ script.js           # Frontend logic
├─ .env.example           # Mẫu biến môi trường
├─ .gitignore             # Bảo vệ .env và artifacts
├─ requirements.txt       # Phụ thuộc Python
└─ README.md
```

## Biến môi trường (.env)
Sao chép `.env.example` thành `.env` và điền giá trị thật:
```
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://aiportalapi.stu-platform.live/jpe
OPENAI_MODEL=gpt-4o-mini
```

## Cài đặt phụ thuộc
```cmd
python -m pip install -r requirements.txt
```

Nếu pip không nhận, dùng:
```cmd
python -m pip install openai python-dotenv flask flask-cors
```

## Chạy giao diện Web (Flask)
```cmd
python app.py
```
Truy cập: http://localhost:5000

## Chạy CLI chat
```cmd
python main.py
```
Gõ câu hỏi, nhập `exit` để thoát.

## Troubleshooting
- 404 CSS/JS: đảm bảo file trong `static/` và `index.html` dùng `{{ url_for('static', filename='...') }}`
- Lỗi key: kiểm tra `.env` đã có `OPENAI_API_KEY`
- `pip` không nhận: dùng `python -m pip install ...`

## Bảo mật
- API key chỉ nằm trong `.env`
- Không hard-code khóa trong code
- `.env` đã được ignore bởi Git

## Giấy phép
Dùng cho mục đích học tập và demo.