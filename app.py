from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
import logging

app = Flask(__name__)
CORS(app)

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cấu hình API cố định (lấy từ main.py)
API_CONFIG = {
    'base_url': 'https://aiportalapi.stu-platform.live/jpe',
    'api_key': 'sk-NYx_MReZLJNz1UzNnYvE4w',
    'model': 'gpt-4o-mini'
}

# Khởi tạo OpenAI client
try:
    client = OpenAI(
        base_url=API_CONFIG['base_url'],
        api_key=API_CONFIG['api_key']
    )
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
        if not client:
            return jsonify({
                'success': False,
                'error': 'Lỗi kết nối API. Hãy khởi động lại server.'
            })
        
        data = request.get_json()
        user_message = data.get('message', '').strip()
        chat_history = data.get('history', [])
        
        if not user_message:
            return jsonify({
                'success': False,
                'error': 'Tin nhắn không được để trống'
            })
        
        logger.info(f"Processing chat message: {user_message[:100]}...")
        
        # Chuẩn bị messages cho OpenAI
        messages = [
            {
                "role": "system", 
                "content": """Bạn là AI Dev Helper - trợ lý lập trình Python chuyên nghiệp. 

Nhiệm vụ của bạn:
- Giải thích code Python một cách chi tiết và dễ hiểu
- Tối ưu hóa code để có performance tốt hơn
- Tìm và sửa lỗi trong code Python
- Viết unit test cho code
- Đưa ra gợi ý cải thiện code
- Trả lời bằng tiếng Việt, ngắn gọn nhưng đầy đủ thông tin
- Sử dụng markdown để format code và làm nổi bật thông tin quan trọng"""
            }
        ]
        
        # Thêm lịch sử chat (giới hạn 8 tin nhắn gần nhất để tiết kiệm token)
        if chat_history:
            messages.extend(chat_history[-8:])
        
        # Thêm tin nhắn hiện tại
        messages.append({"role": "user", "content": user_message})
        
        # Gửi request đến OpenAI
        response = client.chat.completions.create(
            model=API_CONFIG['model'],
            messages=messages,
            max_tokens=16000,
            temperature=0.7
        )
        
        bot_reply = response.choices[0].message.content.strip()
        
        logger.info("✅ Chat response generated successfully")
        
        return jsonify({
            'success': True,
            'response': bot_reply
        })
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"❌ Chat error: {error_message}")
        
        # Xử lý lỗi phổ biến
        if "rate limit" in error_message.lower():
            error_message = "Đã vượt giới hạn request. Hãy thử lại sau."
        elif "quota" in error_message.lower():
            error_message = "Đã hết quota API. Hãy kiểm tra tài khoản."
        elif "model" in error_message.lower():
            error_message = "Model không hỗ trợ hoặc không tồn tại."
        elif "401" in error_message or "unauthorized" in error_message.lower():
            error_message = "API key không hợp lệ."
        
        return jsonify({
            'success': False,
            'error': error_message
        })

@app.route('/api/status')
def status():
    """Kiểm tra trạng thái server"""
    return jsonify({
        'status': 'running',
        'api_connected': client is not None,
        'config': {
            'endpoint': API_CONFIG['base_url'],
            'model': API_CONFIG['model'],
            'has_api_key': bool(API_CONFIG['api_key'])
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
    print("🎯 Tính năng:")
    print("   • Nhập code Python vào panel trái")
    print("   • Chat với AI về code của bạn")
    print("   • Giải thích, tối ưu, debug, viết test")
    print("   • Theme dark mode đẹp mắt")
    
    if client:
        print("✅ API đã kết nối sẵn sàng!")
    else:
        print("❌ Lỗi API - hãy kiểm tra cấu hình")
    
    print("=" * 60)
    
    # Tạo thư mục templates nếu chưa có
    os.makedirs('templates', exist_ok=True)
    
    # Copy file index.html vào thư mục templates
    try:
        import shutil
        if os.path.exists('index.html'):
            shutil.copy2('index.html', 'templates/index.html')
            print("📁 Templates folder ready")
    except Exception as e:
        print(f"⚠️ Warning: {e}")
    
    app.run(host='0.0.0.0', port=5000, debug=True)