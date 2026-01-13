"""
Workshop 4: Enhanced Code Assistant Chatbot với RAG System
- FAISS vector store với code knowledge base
- Langchain ConversationalRetrievalChain
- Azure OpenAI function calling cho code analysis
- Hỗ trợ upload code files và analyze logic/errors
"""

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import os
import logging
import uuid
from typing import Dict, List
from werkzeug.utils import secure_filename

from rag_service import RAGService

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.secret_key = os.urandom(24)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'py', 'txt', 'json', 'js', 'java', 'cpp', 'c'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global RAG service instance
rag_service = None

def init_rag_service():
    """Initialize RAG service"""
    global rag_service
    try:
        logger.info("🚀 Initializing RAG service...")
        rag_service = RAGService()
        rag_service.build()
        logger.info("✅ RAG service initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG service: {e}")
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_session_id():
    """Get or create session ID"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    if 'chat_history' not in session:
        session['chat_history'] = []
    return session['session_id']

@app.route('/')
def index():
    """Home page - chat interface"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Enhanced chat endpoint với RAG + function calling"""
    try:
        if not rag_service or not rag_service.ready:
            return jsonify({
                'success': False,
                'error': 'RAG service chưa được khởi tạo. Vui lòng reload trang.'
            }), 500

        # Get request data
        data = request.get_json() or {}
        user_message = (data.get('message') or '').strip()
        
        if not user_message:
            return jsonify({'success': False, 'error': 'Tin nhắn không được để trống'}), 400

        # Get session and chat history
        session_id = get_session_id()
        chat_history = session.get('chat_history', [])
        
        logger.info(f"Processing message: {user_message[:100]}...")

        # Query RAG system
        result = rag_service.chat(user_message, chat_history)
        
        # Update chat history
        chat_history.append((user_message, result['answer']))
        session['chat_history'] = chat_history[-10:]  # Keep last 10 exchanges
        
        return jsonify({
            'success': True,
            'response': result['answer'],
            'sources': result.get('sources', []),
            'session_id': session_id
        })

    except Exception as e:
        error_message = str(e)
        logger.error(f"❌ Chat error: {error_message}")
        
        return jsonify({'success': False, 'error': f'Lỗi xử lý: {error_message}'}), 500

@app.route('/api/analyze-code', methods=['POST'])
def analyze_code():
    """Direct code analysis endpoint"""
    try:
        data = request.get_json() or {}
        code = data.get('code', '').strip()
        analysis_type = data.get('analysis_type', 'complete')
        
        if not code:
            return jsonify({'success': False, 'error': 'Code không được để trống'}), 400
        
        # Import analysis function
        from rag_service import analyze_code_snippet
        
        result = analyze_code_snippet(code, analysis_type)
        
        return jsonify({
            'success': True,
            'analysis': result
        })
        
    except Exception as e:
        logger.error(f"❌ Analysis error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/upload-code', methods=['POST'])
def upload_code():
    """Upload code file for analysis"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'Không có file được upload'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Không có file được chọn'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False, 
                'error': f'Chỉ chấp nhận các file: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Read file content
        with open(filepath, 'r', encoding='utf-8') as f:
            code_content = f.read()
        
        # Analyze code
        from rag_service import analyze_code_snippet
        analysis_result = analyze_code_snippet(code_content, 'complete')
        
        return jsonify({
            'success': True,
            'filename': filename,
            'code': code_content,
            'analysis': analysis_result
        })
        
    except Exception as e:
        logger.error(f"❌ Upload error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    """Clear chat history"""
    try:
        session['chat_history'] = []
        return jsonify({'success': True, 'message': 'Chat history cleared'})
    except Exception as e:
        logger.error(f"❌ Clear history error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/status')
def status():
    """Status endpoint"""
    return jsonify({
        'status': 'running',
        'rag_ready': rag_service is not None and rag_service.ready,
        'features': {
            'rag_retrieval': True,
            'code_analysis': True,
            'file_upload': True,
            'function_calling': True,
            'conversation_memory': True
        },
        'config': {
            'max_file_size_mb': MAX_FILE_SIZE / (1024 * 1024),
            'allowed_extensions': list(ALLOWED_EXTENSIONS)
        }
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'API endpoint not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("=" * 80)
    print("🚀 Workshop 4: Code Assistant Chatbot với RAG System")
    print("=" * 80)
    print("✨ Features:")
    print("   • FAISS Vector Store với Code Knowledge Base")
    print("   • Langchain ConversationalRetrievalChain")
    print("   • Azure OpenAI Function Calling cho Code Analysis")
    print("   • Upload & Analyze Code Files")
    print("   • Real-time Error Explanation & Bug Detection")
    print("   • Best Practices Suggestions")
    print("=" * 80)
    print("📡 Server URL: http://localhost:5000")
    print("🌐 Open browser và navigate đến URL trên")
    print("🔐 Đọc config từ .env file")
    print("=" * 80)

    # Initialize RAG service
    if not init_rag_service():
        print("⚠️  WARNING: RAG service initialization failed!")
        print("   Kiểm tra .env file và API keys")
        print("   Server sẽ start nhưng AI features có thể không hoạt động")
    
    # Start Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)


