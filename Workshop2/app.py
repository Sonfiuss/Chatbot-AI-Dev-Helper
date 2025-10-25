"""
Advanced AI Development Helper - Workshop 2
Enhanced chatbot system với Azure OpenAI API, function calling, 
conversation management, và advanced prompting techniques
"""

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
import os
import logging
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import custom utilities
from utils.conversation import ConversationManager, BatchProcessor
from utils.prompts import SYSTEM_PROMPTS, FEW_SHOT_EXAMPLES, FUNCTION_DEFINITIONS
from utils.functions import code_analyzer, solution_generator

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

# Load environment variables
load_dotenv()

# Configuration
class Config:
    # Azure OpenAI Configuration (primary)
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
    
    # Fallback OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://aiportalapi.stu-platform.live/jpe")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    # Application settings
    MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "20"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1500"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# Global instances
openai_client = None
conversation_manager = ConversationManager(max_history=Config.MAX_CONVERSATION_HISTORY)
batch_processor = BatchProcessor()

def init_openai_client():
    """Initialize OpenAI client với Azure hoặc fallback"""
    global openai_client
    
    try:
        # Try Azure OpenAI first
        if Config.AZURE_OPENAI_API_KEY and Config.AZURE_OPENAI_ENDPOINT:
            logger.info("Initializing Azure OpenAI client...")
            
            # For Azure OpenAI, we need to use AzureOpenAI client
            try:
                from openai import AzureOpenAI
                openai_client = AzureOpenAI(
                    api_key=Config.AZURE_OPENAI_API_KEY,
                    api_version=Config.AZURE_OPENAI_API_VERSION,
                    azure_endpoint=Config.AZURE_OPENAI_ENDPOINT
                )
                logger.info("✅ Azure OpenAI client initialized successfully")
                return True
            except ImportError:
                logger.warning("Azure OpenAI client not available, falling back to OpenAI")
        
        # Fallback to regular OpenAI
        if Config.OPENAI_API_KEY:
            logger.info("Initializing OpenAI client...")
            openai_client = OpenAI(
                api_key=Config.OPENAI_API_KEY,
                base_url=Config.OPENAI_BASE_URL
            )
            logger.info("✅ OpenAI client initialized successfully")
            return True
            
        logger.error("❌ No valid API keys found")
        return False
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize OpenAI client: {e}")
        openai_client = None
        return False

def get_session_id():
    """Get or create session ID"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']

@app.route('/')
def index():
    """Trang chủ - hiển thị enhanced chat interface"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Enhanced chat endpoint với function calling và conversation management"""
    try:
        if not openai_client:
            return jsonify({
                'success': False, 
                'error': 'OpenAI client chưa được khởi tạo. Vui lòng kiểm tra API key.'
            }), 500

        # Get request data
        data = request.get_json() or {}
        user_message = (data.get('message') or '').strip()
        chat_mode = data.get('mode', 'advanced_chatbot')  # advanced_chatbot, code_reviewer, debugging_assistant
        use_functions = data.get('use_functions', True)
        
        if not user_message:
            return jsonify({'success': False, 'error': 'Tin nhắn không được để trống'}), 400

        # Get session and conversation history
        session_id = get_session_id()
        
        # Add user message to conversation
        conversation_manager.add_message(session_id, 'user', user_message)
        
        logger.info(f"Processing chat message in {chat_mode} mode...")

        # Prepare messages for OpenAI
        messages = []
        
        # Add system prompt based on mode
        system_prompt = SYSTEM_PROMPTS.get(chat_mode, SYSTEM_PROMPTS['advanced_chatbot'])
        messages.append({'role': 'system', 'content': system_prompt})
        
        # Add few-shot examples for better performance
        if chat_mode in FEW_SHOT_EXAMPLES:
            examples = FEW_SHOT_EXAMPLES[chat_mode]['examples'][:2]  # Limit examples
            for example in examples:
                messages.append({'role': 'user', 'content': example['user_input']})
                messages.append({'role': 'assistant', 'content': example['assistant_response']})
        
        # Add conversation history
        history = conversation_manager.get_conversation_history(session_id, last_n=8)
        messages.extend(history)

        # Prepare API call parameters
        api_params = {
            'model': Config.AZURE_OPENAI_DEPLOYMENT_NAME if 'azure' in str(type(openai_client)).lower() else Config.OPENAI_MODEL,
            'messages': messages,
            'max_tokens': Config.MAX_TOKENS,
            'temperature': Config.TEMPERATURE,
        }
        
        # Add function calling if enabled
        if use_functions:
            api_params['functions'] = FUNCTION_DEFINITIONS
            api_params['function_call'] = 'auto'

        # Make API call
        response = openai_client.chat.completions.create(**api_params)
        
        # Process response
        assistant_message = response.choices[0].message
        
        # Handle function calling
        if assistant_message.function_call:
            function_name = assistant_message.function_call.name
            function_args = json.loads(assistant_message.function_call.arguments)
            
            logger.info(f"Function called: {function_name} with args: {function_args}")
            
            # Execute function
            function_result = execute_function(function_name, function_args)
            
            # Add function call và response to conversation
            conversation_manager.add_message(
                session_id, 
                'assistant', 
                assistant_message.content or '',
                function_call=assistant_message.function_call.__dict__
            )
            
            conversation_manager.add_message(
                session_id,
                'function',
                json.dumps(function_result, ensure_ascii=False),
                metadata={'function_name': function_name}
            )
            
            # Get final response after function execution
            messages.append({
                'role': 'assistant',
                'content': assistant_message.content,
                'function_call': assistant_message.function_call.__dict__
            })
            messages.append({
                'role': 'function',
                'name': function_name,
                'content': json.dumps(function_result, ensure_ascii=False)
            })
            
            # Second API call for final response
            api_params['messages'] = messages
            final_response = openai_client.chat.completions.create(**api_params)
            final_content = final_response.choices[0].message.content
            
            # Add final response to conversation
            conversation_manager.add_message(session_id, 'assistant', final_content)
            
            return jsonify({
                'success': True,
                'response': final_content,
                'function_called': function_name,
                'function_result': function_result,
                'conversation_context': conversation_manager.get_conversation_context(session_id)
            })
        
        else:
            # Regular response without function calling
            bot_reply = assistant_message.content.strip()
            
            # Add to conversation history
            conversation_manager.add_message(session_id, 'assistant', bot_reply)
            
            return jsonify({
                'success': True,
                'response': bot_reply,
                'conversation_context': conversation_manager.get_conversation_context(session_id)
            })

    except Exception as e:
        error_message = str(e)
        logger.error(f"❌ Chat error: {error_message}")
        
        # Handle common errors
        if '401' in error_message or 'unauthorized' in error_message.lower():
            error_message = 'API key không hợp lệ hoặc đã hết hạn.'
        elif 'rate_limit' in error_message.lower():
            error_message = 'Đã vượt quá giới hạn API calls. Vui lòng thử lại sau.'
        elif 'timeout' in error_message.lower():
            error_message = 'Request timeout. Vui lòng thử lại.'
            
        return jsonify({'success': False, 'error': error_message}), 500

def execute_function(function_name: str, arguments: Dict) -> Any:
    """Execute function based on name and arguments"""
    try:
        if function_name == 'analyze_code':
            return code_analyzer.analyze_code(
                arguments.get('code', ''),
                arguments.get('analysis_type', 'complete'),
                arguments.get('context', '')
            )
        
        elif function_name == 'generate_solution':
            return solution_generator.generate_solution(
                arguments.get('problem_description', ''),
                arguments.get('requirements', []),
                arguments.get('constraints', ''),
                arguments.get('user_level', 'mid')
            )
        
        elif function_name == 'create_test_cases':
            # Simplified test case generation
            code = arguments.get('code_to_test', '')
            framework = arguments.get('test_framework', 'pytest')
            
            return {
                "test_code": f"# Test cases for the provided code\nimport {framework}\n\ndef test_example():\n    # TODO: Implement test cases\n    pass",
                "framework": framework,
                "coverage_areas": ["happy_path", "edge_cases", "error_handling"]
            }
        
        else:
            return {"error": f"Unknown function: {function_name}"}
            
    except Exception as e:
        logger.error(f"Function execution error: {e}")
        return {"error": f"Function execution failed: {str(e)}"}

@app.route('/api/conversation/export/<session_id>')
def export_conversation(session_id: str):
    """Export conversation logs for analysis"""
    try:
        conversation_data = conversation_manager.export_conversation(session_id)
        
        if not conversation_data:
            return jsonify({'error': 'Session not found'}), 404
            
        return jsonify({
            'success': True,
            'conversation': conversation_data
        })
        
    except Exception as e:
        logger.error(f"Export error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/conversation/clear')
def clear_conversation():
    """Clear current conversation"""
    try:
        session_id = get_session_id()
        conversation_manager.clear_session(session_id)
        
        return jsonify({'success': True, 'message': 'Conversation cleared'})
        
    except Exception as e:
        logger.error(f"Clear conversation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def status():
    """Enhanced status endpoint"""
    return jsonify({
        'status': 'running',
        'api_connected': openai_client is not None,
        'client_type': 'Azure OpenAI' if 'azure' in str(type(openai_client)).lower() else 'OpenAI',
        'config': {
            'max_history': Config.MAX_CONVERSATION_HISTORY,
            'max_tokens': Config.MAX_TOKENS,
            'temperature': Config.TEMPERATURE,
            'has_azure_key': bool(Config.AZURE_OPENAI_API_KEY),
            'has_openai_key': bool(Config.OPENAI_API_KEY)
        },
        'features': {
            'function_calling': True,
            'conversation_management': True,
            'batch_processing': True,
            'few_shot_prompting': True,
            'chain_of_thought': True
        }
    })

@app.route('/api/conversation/context')
def get_conversation_context():
    """Get current conversation context"""
    try:
        session_id = get_session_id()
        context = conversation_manager.get_conversation_context(session_id)
        
        return jsonify({
            'success': True,
            'context': context
        })
        
    except Exception as e:
        logger.error(f"Context error: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'API endpoint not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 AI Development Expert - Workshop 2")
    print("=" * 70)
    print("🎯 Enhanced Features:")
    print("   • Azure OpenAI API integration")
    print("   • Advanced function calling")
    print("   • Multi-turn conversation management")
    print("   • Few-shot & chain-of-thought prompting")
    print("   • Comprehensive code analysis")
    print("   • Solution generation for real problems")
    print("=" * 70)
    print("📡 Server starting at: http://localhost:5000")
    print("🌐 Open browser and navigate to the address above")
    print("🔐 API Configuration will be read from .env file")
    print("=" * 70)

    # Initialize OpenAI client
    if not init_openai_client():
        print("⚠️  WARNING: OpenAI client initialization failed!")
        print("   Please check your .env file and API keys")
        print("   Server will start but AI features may not work")
    
    # Start Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)