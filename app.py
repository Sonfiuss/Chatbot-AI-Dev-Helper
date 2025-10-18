from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
import os
import logging

app = Flask(__name__)
CORS(app)

# Logging cấu hình
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Nạp biến môi trường từ .env (nếu có)
load_dotenv()

BASE_URL = os.getenv("OPENAI_BASE_URL", "https://aiportalapi.stu-platform.live/jpe")
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

client = None

def init_client():
    global client
    try:
        if not API_KEY:
            logger.warning("OPENAI_API_KEY is not set. API calls will fail until it's provided.")
            client = None
            return
        client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
        logger.info("✅ OpenAI client initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize OpenAI client: {e}")
        client = None


@app.route('/')
def index():
    """Trang chủ - hiển thị giao diện chat"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Xử lý tin nhắn chat"""
    try:
        if client is None:
            init_client()
        if client is None:
            return jsonify({'success': False, 'error': 'Chưa cấu hình OPENAI_API_KEY trong .env'}), 400

        data = request.get_json() or {}
        user_message = (data.get('message') or '').strip()
        chat_history = data.get('history') or []

        if not user_message:
            return jsonify({'success': False, 'error': 'Tin nhắn không được để trống'}), 400

        logger.info("Processing chat message …")

        # Chuẩn bị messages cho OpenAI
        messages = [{
            'role': 'system',
            'content': (
                'Bạn là AI Dev Helper - trợ lý lập trình Python. '
                'Trả lời bằng tiếng Việt khi người dùng dùng tiếng Việt. '
                'Sử dụng markdown, ưu tiên ví dụ code ngắn gọn.'
            )
        }]

        # Thêm lịch sử chat (giới hạn 8 tin gần nhất)
        if chat_history:
            messages.extend(chat_history[-8:])

        messages.append({'role': 'user', 'content': user_message})

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=800,
            temperature=0.7,
        )

        bot_reply = response.choices[0].message.content.strip()
        return jsonify({'success': True, 'response': bot_reply})

    except Exception as e:
        error_message = str(e)
        logger.error(f"❌ Chat error: {error_message}")
        if '401' in error_message or 'unauthorized' in error_message.lower():
            error_message = 'API key không hợp lệ hoặc đã hết hạn.'
        return jsonify({'success': False, 'error': error_message}), 500

@app.route('/api/status')
def status():
    """Kiểm tra trạng thái server"""
    return jsonify({
        'status': 'running',
        'api_connected': client is not None,
        'config': {
            'endpoint': BASE_URL,
            'model': MODEL_NAME,
            'has_api_key': bool(API_KEY)
        }
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'API endpoint not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 AI Dev Helper - Python Assistant")
    print("=" * 60)
    print("📡 Server đang khởi động tại: http://localhost:5000")
    print("🌐 Mở trình duyệt và truy cập địa chỉ trên để sử dụng")
    print("🔐 Lưu ý: cấu hình API được đọc từ biến môi trường/.env (không commit lên git)")

    # Khởi tạo client (nếu có API key)
    init_client()

    app.run(host='0.0.0.0', port=5000, debug=True)
