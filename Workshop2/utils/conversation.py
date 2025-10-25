"""
Conversation Management Utilities
Quản lý lịch sử hội thoại, context và function calls
"""

import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import uuid

@dataclass
class ConversationMessage:
    """Represent a single message in conversation"""
    id: str
    role: str  # 'user', 'assistant', 'system', 'function'
    content: str
    timestamp: datetime
    function_call: Optional[Dict] = None
    function_response: Optional[Dict] = None
    metadata: Optional[Dict] = None

class ConversationManager:
    """Quản lý lịch sử hội thoại và context"""
    
    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self.conversations: Dict[str, List[ConversationMessage]] = {}
        
    def create_session(self, session_id: str = None) -> str:
        """Tạo session mới cho conversation"""
        if not session_id:
            session_id = str(uuid.uuid4())
        self.conversations[session_id] = []
        return session_id
    
    def add_message(self, session_id: str, role: str, content: str, 
                   function_call: Dict = None, function_response: Dict = None,
                   metadata: Dict = None) -> ConversationMessage:
        """Thêm message vào conversation history"""
        if session_id not in self.conversations:
            self.create_session(session_id)
            
        message = ConversationMessage(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=datetime.now(),
            function_call=function_call,
            function_response=function_response,
            metadata=metadata or {}
        )
        
        self.conversations[session_id].append(message)
        
        # Giới hạn history length
        if len(self.conversations[session_id]) > self.max_history:
            self.conversations[session_id] = self.conversations[session_id][-self.max_history:]
            
        return message
    
    def get_conversation_history(self, session_id: str, 
                               include_system: bool = False,
                               last_n: int = None) -> List[Dict]:
        """Lấy lịch sử conversation theo format OpenAI"""
        if session_id not in self.conversations:
            return []
            
        messages = self.conversations[session_id]
        
        if not include_system:
            messages = [m for m in messages if m.role != 'system']
            
        if last_n:
            messages = messages[-last_n:]
            
        # Convert to OpenAI format
        result = []
        for msg in messages:
            openai_msg = {
                "role": msg.role,
                "content": msg.content
            }
            
            if msg.function_call:
                openai_msg["function_call"] = msg.function_call
                
            result.append(openai_msg)
            
        return result
    
    def get_conversation_context(self, session_id: str) -> Dict:
        """Lấy context summary của conversation"""
        if session_id not in self.conversations:
            return {}
            
        messages = self.conversations[session_id]
        
        # Phân tích conversation
        user_messages = [m for m in messages if m.role == 'user']
        assistant_messages = [m for m in messages if m.role == 'assistant'] 
        function_calls = [m for m in messages if m.function_call]
        
        # Topics được thảo luận (simple keyword extraction)
        topics = set()
        for msg in user_messages:
            # Simple keyword extraction
            words = msg.content.lower().split()
            python_keywords = ['python', 'flask', 'django', 'fastapi', 'api', 'database', 
                             'sql', 'postgresql', 'mysql', 'redis', 'celery', 'docker']
            topics.update([w for w in words if w in python_keywords])
        
        return {
            "session_id": session_id,
            "message_count": len(messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "function_calls_used": len(function_calls),
            "topics_discussed": list(topics),
            "last_activity": messages[-1].timestamp if messages else None,
            "conversation_duration": (messages[-1].timestamp - messages[0].timestamp).total_seconds() if len(messages) > 1 else 0
        }
    
    def export_conversation(self, session_id: str) -> Dict:
        """Export conversation để testing/analysis"""
        if session_id not in self.conversations:
            return {}
            
        messages = self.conversations[session_id]
        
        return {
            "session_id": session_id,
            "exported_at": datetime.now().isoformat(),
            "message_count": len(messages),
            "messages": [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "function_call": msg.function_call,
                    "function_response": msg.function_response,
                    "metadata": msg.metadata
                }
                for msg in messages
            ],
            "context": self.get_conversation_context(session_id)
        }
    
    def clear_session(self, session_id: str):
        """Xóa conversation history"""
        if session_id in self.conversations:
            del self.conversations[session_id]

class BatchProcessor:
    """Xử lý batch requests để tối ưu API calls"""
    
    def __init__(self):
        self.pending_requests = []
        self.batch_size = 5
        self.batch_timeout = 2.0  # seconds
        
    def add_request(self, request_data: Dict) -> str:
        """Thêm request vào batch queue"""
        request_id = str(uuid.uuid4())
        self.pending_requests.append({
            "id": request_id,
            "data": request_data,
            "timestamp": datetime.now()
        })
        return request_id
    
    def should_process_batch(self) -> bool:
        """Kiểm tra có nên process batch không"""
        if len(self.pending_requests) >= self.batch_size:
            return True
            
        if self.pending_requests:
            oldest_request = min(self.pending_requests, key=lambda x: x["timestamp"])
            age = (datetime.now() - oldest_request["timestamp"]).total_seconds()
            return age >= self.batch_timeout
            
        return False
    
    def get_batch(self) -> List[Dict]:
        """Lấy batch requests để process"""
        batch = self.pending_requests[:self.batch_size]
        self.pending_requests = self.pending_requests[self.batch_size:]
        return batch

def create_function_call_message(function_name: str, arguments: Dict) -> Dict:
    """Tạo function call message format cho OpenAI"""
    return {
        "role": "assistant",
        "content": None,
        "function_call": {
            "name": function_name,
            "arguments": json.dumps(arguments, ensure_ascii=False)
        }
    }

def create_function_response_message(function_name: str, response: Any) -> Dict:
    """Tạo function response message format cho OpenAI"""
    return {
        "role": "function", 
        "name": function_name,
        "content": json.dumps(response, ensure_ascii=False) if isinstance(response, (dict, list)) else str(response)
    }