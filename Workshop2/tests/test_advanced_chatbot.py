"""
Test suite for Advanced AI Development Expert - Workshop 2
Comprehensive testing cho conversation management, function calling, và API endpoints
"""

import pytest
import json
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from utils.conversation import ConversationManager, BatchProcessor
from utils.functions import CodeAnalyzer, SolutionGenerator

class TestConversationManager:
    """Test conversation management functionality"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.conv_manager = ConversationManager(max_history=5)
        
    def test_create_session(self):
        """Test session creation"""
        session_id = self.conv_manager.create_session()
        assert session_id is not None
        assert session_id in self.conv_manager.conversations
        assert len(self.conv_manager.conversations[session_id]) == 0
        
    def test_add_message(self):
        """Test adding messages to conversation"""
        session_id = self.conv_manager.create_session()
        
        message = self.conv_manager.add_message(
            session_id, 'user', 'Test message'
        )
        
        assert message.role == 'user'
        assert message.content == 'Test message'
        assert len(self.conv_manager.conversations[session_id]) == 1
        
    def test_conversation_history_limit(self):
        """Test conversation history limiting"""
        session_id = self.conv_manager.create_session()
        
        # Add more messages than max_history
        for i in range(10):
            self.conv_manager.add_message(session_id, 'user', f'Message {i}')
            
        # Should only keep last 5 messages
        assert len(self.conv_manager.conversations[session_id]) == 5
        assert self.conv_manager.conversations[session_id][-1].content == 'Message 9'
        
    def test_get_conversation_history(self):
        """Test getting conversation history in OpenAI format"""
        session_id = self.conv_manager.create_session()
        
        self.conv_manager.add_message(session_id, 'user', 'Hello')
        self.conv_manager.add_message(session_id, 'assistant', 'Hi there!')
        
        history = self.conv_manager.get_conversation_history(session_id)
        
        assert len(history) == 2
        assert history[0]['role'] == 'user'
        assert history[0]['content'] == 'Hello'
        assert history[1]['role'] == 'assistant'
        assert history[1]['content'] == 'Hi there!'
        
    def test_conversation_context(self):
        """Test conversation context generation"""
        session_id = self.conv_manager.create_session()
        
        self.conv_manager.add_message(session_id, 'user', 'Test python function')
        self.conv_manager.add_message(session_id, 'assistant', 'Sure, here is a Flask API')
        
        context = self.conv_manager.get_conversation_context(session_id)
        
        assert context['session_id'] == session_id
        assert context['message_count'] == 2
        assert context['user_messages'] == 1
        assert context['assistant_messages'] == 1
        assert 'python' in context['topics_discussed'] or 'flask' in context['topics_discussed']
        
    def test_export_conversation(self):
        """Test conversation export"""
        session_id = self.conv_manager.create_session()
        
        self.conv_manager.add_message(session_id, 'user', 'Test message')
        
        export_data = self.conv_manager.export_conversation(session_id)
        
        assert 'session_id' in export_data
        assert 'messages' in export_data
        assert 'context' in export_data
        assert len(export_data['messages']) == 1

class TestBatchProcessor:
    """Test batch processing functionality"""
    
    def setup_method(self):
        self.batch_processor = BatchProcessor()
        
    def test_add_request(self):
        """Test adding request to batch"""
        request_data = {"message": "test", "mode": "chatbot"}
        request_id = self.batch_processor.add_request(request_data)
        
        assert request_id is not None
        assert len(self.batch_processor.pending_requests) == 1
        
    def test_should_process_batch_size(self):
        """Test batch processing trigger by size"""
        # Add requests up to batch size
        for i in range(self.batch_processor.batch_size):
            self.batch_processor.add_request({"message": f"test {i}"})
            
        assert self.batch_processor.should_process_batch() == True
        
    def test_get_batch(self):
        """Test getting batch for processing"""
        # Add some requests
        for i in range(3):
            self.batch_processor.add_request({"message": f"test {i}"})
            
        batch = self.batch_processor.get_batch()
        
        assert len(batch) == 3
        assert len(self.batch_processor.pending_requests) == 0

class TestCodeAnalyzer:
    """Test code analysis functionality"""
    
    def setup_method(self):
        self.analyzer = CodeAnalyzer()
        
    def test_analyze_valid_code(self):
        """Test analyzing valid Python code"""
        code = """
def hello_world():
    print("Hello, World!")
    return True
"""
        
        result = self.analyzer.analyze_code(code, "complete")
        
        assert 'issues' in result
        assert 'rating' in result
        assert 'suggestions' in result
        assert result['rating'] >= 5
        
    def test_detect_security_issues(self):
        """Test detection of security issues"""
        code = """
import os
def dangerous_function(user_input):
    eval(user_input)  # Security issue
    os.system(user_input)  # Another security issue
"""
        
        result = self.analyzer.analyze_code(code, "security")
        
        security_issues = [issue for issue in result['issues'] if issue['type'] == 'security']
        assert len(security_issues) >= 2
        assert any('eval' in issue['description'] for issue in security_issues)
        
    def test_detect_performance_issues(self):
        """Test detection of performance issues"""
        code = """
def slow_function():
    users = User.query.all()  # Performance issue
    for user in User.query.filter_by(active=True):  # N+1 query
        print(user.name)
"""
        
        result = self.analyzer.analyze_code(code, "performance")
        
        perf_issues = [issue for issue in result['issues'] if issue['type'] == 'performance']
        assert len(perf_issues) >= 1
        
    def test_syntax_error_detection(self):
        """Test detection of syntax errors"""
        code = """
def broken_function(
    print("Missing closing parenthesis")
"""
        
        result = self.analyzer.analyze_code(code, "bugs")
        
        assert any(issue['type'] == 'bug' and 'Syntax Error' in issue['description'] 
                  for issue in result['issues'])

class TestSolutionGenerator:
    """Test solution generation functionality"""
    
    def setup_method(self):
        self.generator = SolutionGenerator()
        
    def test_generate_api_solution(self):
        """Test generating API solution"""
        problem = "Create a REST API for user management"
        requirements = ["CRUD operations", "Authentication", "Validation"]
        
        result = self.generator.generate_solution(
            problem, requirements, user_level="mid"
        )
        
        assert 'solution_approach' in result
        assert 'code_implementation' in result
        assert 'explanation' in result
        assert 'best_practices' in result
        
    def test_user_level_adaptation(self):
        """Test solution adaptation based on user level"""
        problem = "Create a simple API endpoint"
        requirements = ["GET request", "JSON response"]
        
        junior_result = self.generator.generate_solution(
            problem, requirements, user_level="junior"
        )
        
        senior_result = self.generator.generate_solution(
            problem, requirements, user_level="senior"
        )
        
        # Senior solution should be more complex
        assert len(senior_result['code_implementation']) > len(junior_result['code_implementation'])
        assert 'decorator' in senior_result['code_implementation'].lower()

class TestFlaskAPI:
    """Test Flask API endpoints"""
    
    def setup_method(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        
    def test_status_endpoint(self):
        """Test status API endpoint"""
        response = self.client.get('/api/status')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'status' in data
        assert 'features' in data
        assert data['status'] == 'running'
        
    def test_conversation_context_endpoint(self):
        """Test conversation context endpoint"""
        with self.client.session_transaction() as sess:
            sess['session_id'] = 'test-session'
            
        response = self.client.get('/api/conversation/context')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'success' in data
        
    def test_clear_conversation_endpoint(self):
        """Test clear conversation endpoint"""
        response = self.client.get('/api/conversation/clear')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['success'] == True
        
    def test_chat_endpoint_validation(self):
        """Test chat endpoint input validation"""
        # Test empty message
        response = self.client.post('/api/chat', 
                                  json={'message': ''})
        assert response.status_code == 400
        
        # Test missing message
        response = self.client.post('/api/chat', json={})
        assert response.status_code == 400

    @patch('app.openai_client')
    def test_chat_endpoint_success(self, mock_client):
        """Test successful chat endpoint call"""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].message.function_call = None
        
        mock_client.chat.completions.create.return_value = mock_response
        
        response = self.client.post('/api/chat', 
                                  json={'message': 'Test message'})
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
        assert 'response' in data

class TestIntegration:
    """Integration tests for complete workflows"""
    
    def setup_method(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        
    def test_complete_conversation_flow(self):
        """Test complete conversation workflow"""
        with self.client.session_transaction() as sess:
            sess['session_id'] = 'integration-test'
        
        # Check initial status
        response = self.client.get('/api/status')
        assert response.status_code == 200
        
        # Get conversation context (should be empty)
        response = self.client.get('/api/conversation/context')
        assert response.status_code == 200
        
        # Clear conversation
        response = self.client.get('/api/conversation/clear')
        assert response.status_code == 200
        
    def test_code_analysis_workflow(self):
        """Test code analysis integration"""
        analyzer = CodeAnalyzer()
        
        # Sample code with multiple issue types
        test_code = """
def risky_function(user_input):
    # Security issue
    eval(user_input)
    
    # Performance issue  
    users = User.query.all()
    
    # Style issue - no docstring
    pass
"""
        
        # Analyze code
        result = analyzer.analyze_code(test_code, "complete")
        
        # Should detect issues across categories
        issue_types = set(issue['type'] for issue in result['issues'])
        assert 'security' in issue_types
        assert len(result['suggestions']) > 0
        assert result['rating'] < 8  # Should be low due to issues

class TestMockDataIntegration:
    """Test integration with mock data scenarios"""
    
    def test_mock_data_loading(self):
        """Test loading and using mock data"""
        # This would test loading mock_data.json
        # and using it in conversation scenarios
        pass
        
    def test_problem_scenario_handling(self):
        """Test handling of company problem scenarios"""
        generator = SolutionGenerator()
        
        # Simulate API optimization problem
        problem = "API endpoint slow with high concurrent users"
        requirements = [
            "Identify bottlenecks",
            "Implement caching",
            "Optimize database queries"
        ]
        
        result = generator.generate_solution(problem, requirements, "senior")
        
        assert 'performance' in result['solution_approach']['type']
        assert 'caching' in result['code_implementation'].lower()

# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    # Add custom markers
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "slow: mark test as slow running")

# Test fixtures
@pytest.fixture
def sample_conversation_data():
    """Fixture providing sample conversation data"""
    return {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "Analyze this code: def test(): pass"},
            {"role": "assistant", "content": "This is a simple function..."}
        ]
    }

@pytest.fixture
def sample_code():
    """Fixture providing sample code for testing"""
    return """
def calculate_total(items):
    '''Calculate total price of items'''
    total = 0
    for item in items:
        total += item.price * item.quantity
    return total

class ShoppingCart:
    def __init__(self):
        self.items = []
        
    def add_item(self, item):
        self.items.append(item)
        
    def get_total(self):
        return calculate_total(self.items)
"""

if __name__ == '__main__':
    # Run tests with coverage
    pytest.main([
        '--cov=app',
        '--cov=utils',
        '--cov-report=html',
        '--cov-report=term-missing',
        '-v'
    ])