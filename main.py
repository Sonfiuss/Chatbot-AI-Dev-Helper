from openai import OpenAI
from dotenv import load_dotenv
import os

# ===== CẤU HÌNH API (qua .env) =====
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://aiportalapi.stu-platform.live/jpe")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

print("🔧 Đang cấu hình API...")
print(f"📡 Endpoint: {BASE_URL}")
print(f"🤖 Model: {MODEL_NAME}")

if not API_KEY:
    print("❌ Thiếu OPENAI_API_KEY. Hãy tạo file .env và thêm OPENAI_API_KEY=... hoặc đặt biến môi trường.")
    exit(1)

try:
    # Khởi tạo OpenAI client với endpoint tùy chỉnh
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    # Test kết nối ngắn
    _ = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=5
    )
    print("✅ Kết nối API thành công!")
except Exception as e:
    print(f"❌ Lỗi kết nối API: {e}")
    print("🔍 Kiểm tra lại API key, endpoint, model và kết nối internet")
    exit(1)

# Lưu lịch sử hội thoại
messages = [
    {"role": "system", "content": "You are AI Dev Helper, a friendly assistant that helps Python developers. Reply in Vietnamese when user asks in Vietnamese."}
]

print("💬 AI Dev Helper đã sẵn sàng! Gõ 'exit' để thoát.\n")

while True:
    user_input = input("👨‍💻 Bạn: ")
    
    if user_input.lower() in ["exit", "quit", "thoát"]:
        print("👋 Tạm biệt!")
        break
    
    if not user_input.strip():
        continue
        
    messages.append({"role": "user", "content": user_input})

    try:
        # Gửi yêu cầu đến API
        response = client.chat.completions.create(
            model=MODEL_NAME,  # Sử dụng model gpt-4o-mini theo cấu hình
            messages=messages,
            max_tokens=500,  # Giảm token để tiết kiệm budget 3$
            temperature=0.7
        )

        reply = response.choices[0].message.content
        print(f"🤖 DevHelper: {reply}\n")

        # Lưu phản hồi để duy trì ngữ cảnh
        messages.append({"role": "assistant", "content": reply})
        
    except Exception as e:
        print(f"❌ Lỗi API: {e}")
        print("🔄 Thử lại hoặc kiểm tra kết nối mạng\n")