"""
Function implementations cho AI chatbot system
Các functions được gọi bởi OpenAI để thực hiện specific tasks
"""

import ast
import re
import json
from typing import Dict, List, Any
import subprocess
import tempfile
import os

class CodeAnalyzer:
    """Phân tích code Python với nhiều tiêu chí khác nhau"""
    
    def __init__(self):
        self.security_patterns = [
            (r'eval\s*\(', 'HIGH', 'Sử dụng eval() có thể dẫn đến code injection'),
            (r'exec\s*\(', 'HIGH', 'Sử dụng exec() có thể dẫn đến code injection'),
            (r'__import__\s*\(', 'MEDIUM', 'Dynamic import có thể là security risk'),
            (r'open\s*\([^)]*["\']w["\']', 'MEDIUM', 'File write operation cần validate path'),
            (r'subprocess\.|os\.system', 'HIGH', 'Command execution cần validate input'),
            (r'pickle\.loads?', 'HIGH', 'Pickle deserialization có thể unsafe'),
            (r'yaml\.load\s*\((?!.*Loader)', 'MEDIUM', 'yaml.load() cần specify safe Loader')
        ]
        
        self.performance_patterns = [
            (r'\.query\.all\(\)', 'MEDIUM', 'Có thể load quá nhiều records vào memory'),
            (r'for\s+\w+\s+in\s+\w+\.query\.[^:]+:', 'HIGH', 'Potential N+1 query problem'),
            (r'time\.sleep\s*\(', 'MEDIUM', 'Blocking sleep trong request thread'),
            (r'requests\.(get|post)', 'MEDIUM', 'Synchronous HTTP calls có thể chậm'),
            (r'\.join\s*\(\s*\)', 'LOW', 'String concatenation trong loop kém hiệu quả')
        ]
        
    def analyze_code(self, code: str, analysis_type: str, context: str = "") -> Dict:
        """Main analysis function"""
        try:
            results = {
                "analysis_type": analysis_type,
                "code_length": len(code.splitlines()),
                "context": context,
                "issues": [],
                "suggestions": [],
                "rating": 0,
                "summary": ""
            }
            
            if analysis_type in ['security', 'complete']:
                results["issues"].extend(self._check_security(code))
                
            if analysis_type in ['performance', 'complete']:
                results["issues"].extend(self._check_performance(code))
                
            if analysis_type in ['bugs', 'complete']:
                results["issues"].extend(self._check_bugs(code))
                
            if analysis_type in ['style', 'complete']:
                results["issues"].extend(self._check_style(code))
            
            # Generate overall rating
            results["rating"] = self._calculate_rating(results["issues"])
            results["suggestions"] = self._generate_suggestions(results["issues"], code)
            results["summary"] = self._generate_summary(results)
            
            return results
            
        except Exception as e:
            return {
                "error": f"Code analysis failed: {str(e)}",
                "analysis_type": analysis_type
            }
    
    def _check_security(self, code: str) -> List[Dict]:
        """Check for security issues"""
        issues = []
        for pattern, severity, description in self.security_patterns:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                issues.append({
                    "type": "security",
                    "severity": severity,
                    "line": line_num,
                    "description": description,
                    "code_snippet": self._get_line_context(code, line_num)
                })
        return issues
    
    def _check_performance(self, code: str) -> List[Dict]:
        """Check for performance issues"""
        issues = []
        for pattern, severity, description in self.performance_patterns:
            matches = re.finditer(pattern, code)
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                issues.append({
                    "type": "performance", 
                    "severity": severity,
                    "line": line_num,
                    "description": description,
                    "code_snippet": self._get_line_context(code, line_num)
                })
        return issues
    
    def _check_bugs(self, code: str) -> List[Dict]:
        """Check for potential bugs"""
        issues = []
        try:
            # Parse AST để kiểm tra syntax errors
            ast.parse(code)
            
            # Check for common bug patterns
            bug_patterns = [
                (r'if\s+\w+\s*=\s*', 'HIGH', 'Assignment trong if condition (có thể muốn dùng ==)'),
                (r'except\s*:', 'MEDIUM', 'Bare except clause - nên specify exception type'),
                (r'return\s+\w+\s*=', 'HIGH', 'Assignment trong return statement'),
                (r'\.format\s*\(\s*\)', 'LOW', 'Empty .format() call'),
            ]
            
            for pattern, severity, description in bug_patterns:
                matches = re.finditer(pattern, code)
                for match in matches:
                    line_num = code[:match.start()].count('\n') + 1
                    issues.append({
                        "type": "bug",
                        "severity": severity, 
                        "line": line_num,
                        "description": description,
                        "code_snippet": self._get_line_context(code, line_num)
                    })
                    
        except SyntaxError as e:
            issues.append({
                "type": "bug",
                "severity": "HIGH",
                "line": e.lineno,
                "description": f"Syntax Error: {e.msg}",
                "code_snippet": self._get_line_context(code, e.lineno) if e.lineno else ""
            })
            
        return issues
    
    def _check_style(self, code: str) -> List[Dict]:
        """Check for style issues"""
        issues = []
        
        style_patterns = [
            (r'def\s+\w+\([^)]*\):\s*$', 'LOW', 'Function thiếu docstring'),
            (r'class\s+\w+[^:]*:\s*$', 'LOW', 'Class thiếu docstring'),
            (r'\t', 'LOW', 'Sử dụng tabs thay vì spaces'),
            (r'  {5,}', 'LOW', 'Indentation không đều (>4 spaces)'),
            (r'\w{50,}', 'LOW', 'Tên variable/function quá dài'),
        ]
        
        for pattern, severity, description in style_patterns:
            matches = re.finditer(pattern, code, re.MULTILINE)
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                issues.append({
                    "type": "style",
                    "severity": severity,
                    "line": line_num, 
                    "description": description,
                    "code_snippet": self._get_line_context(code, line_num)
                })
                
        return issues
    
    def _get_line_context(self, code: str, line_num: int, context_lines: int = 2) -> str:
        """Get code context around specific line"""
        lines = code.splitlines()
        start = max(0, line_num - context_lines - 1)
        end = min(len(lines), line_num + context_lines)
        
        context = []
        for i in range(start, end):
            marker = "→ " if i == line_num - 1 else "  "
            context.append(f"{i+1:3d} {marker}{lines[i]}")
            
        return "\n".join(context)
    
    def _calculate_rating(self, issues: List[Dict]) -> int:
        """Calculate overall code rating 1-10"""
        if not issues:
            return 9
            
        score = 10
        for issue in issues:
            if issue["severity"] == "HIGH":
                score -= 2
            elif issue["severity"] == "MEDIUM":
                score -= 1
            elif issue["severity"] == "LOW":
                score -= 0.5
                
        return max(1, int(score))
    
    def _generate_suggestions(self, issues: List[Dict], code: str) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []
        
        # Group issues by type
        issue_types = {}
        for issue in issues:
            issue_type = issue["type"]
            if issue_type not in issue_types:
                issue_types[issue_type] = []
            issue_types[issue_type].append(issue)
        
        # Generate suggestions per type
        if "security" in issue_types:
            suggestions.append("🔒 Implement input validation và sanitization")
            suggestions.append("🔒 Sử dụng parameterized queries cho database")
            
        if "performance" in issue_types:
            suggestions.append("⚡ Implement caching cho frequently accessed data")
            suggestions.append("⚡ Sử dụng async/await cho I/O operations")
            suggestions.append("⚡ Add database indexes cho query optimization")
            
        if "bug" in issue_types:
            suggestions.append("🐛 Add comprehensive error handling")
            suggestions.append("🐛 Write unit tests để catch edge cases")
            
        if "style" in issue_types:
            suggestions.append("📝 Add docstrings cho functions và classes")
            suggestions.append("📝 Follow PEP8 coding standards")
            
        return suggestions
    
    def _generate_summary(self, results: Dict) -> str:
        """Generate analysis summary"""
        issue_count = len(results["issues"])
        rating = results["rating"]
        
        if issue_count == 0:
            return f"✅ Code chất lượng tốt (Rating: {rating}/10). Không phát hiện issues nghiêm trọng."
        
        severity_counts = {}
        for issue in results["issues"]:
            severity = issue["severity"]
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
        summary_parts = [f"📊 Phát hiện {issue_count} issues (Rating: {rating}/10):"]
        
        if "HIGH" in severity_counts:
            summary_parts.append(f"🔴 {severity_counts['HIGH']} critical issues")
        if "MEDIUM" in severity_counts:
            summary_parts.append(f"🟡 {severity_counts['MEDIUM']} major issues") 
        if "LOW" in severity_counts:
            summary_parts.append(f"🔵 {severity_counts['LOW']} minor issues")
            
        return " • ".join(summary_parts)

class SolutionGenerator:
    """Generate code solutions for specific problems"""
    
    def generate_solution(self, problem_description: str, requirements: List[str], 
                         constraints: str = "", user_level: str = "mid") -> Dict:
        """Generate solution based on problem description"""
        
        try:
            solution = {
                "problem": problem_description,
                "user_level": user_level,
                "solution_approach": self._analyze_problem(problem_description),
                "code_implementation": self._generate_code(problem_description, requirements, user_level),
                "explanation": self._generate_explanation(problem_description, user_level),
                "best_practices": self._get_best_practices(problem_description),
                "testing_strategy": self._get_testing_strategy(problem_description),
                "next_steps": self._get_next_steps(user_level)
            }
            
            return solution
            
        except Exception as e:
            return {"error": f"Solution generation failed: {str(e)}"}
    
    def _analyze_problem(self, problem: str) -> Dict:
        """Analyze problem to determine approach"""
        problem_lower = problem.lower()
        
        # Detect problem type
        if any(word in problem_lower for word in ['api', 'endpoint', 'rest', 'flask']):
            problem_type = "api_development"
        elif any(word in problem_lower for word in ['database', 'sql', 'query']):
            problem_type = "database_optimization"
        elif any(word in problem_lower for word in ['performance', 'slow', 'optimize']):
            problem_type = "performance_optimization"
        elif any(word in problem_lower for word in ['test', 'testing', 'unit']):
            problem_type = "testing"
        else:
            problem_type = "general_development"
            
        return {
            "type": problem_type,
            "complexity": self._estimate_complexity(problem),
            "technologies": self._detect_technologies(problem),
            "approach": self._get_approach_for_type(problem_type)
        }
    
    def _generate_code(self, problem: str, requirements: List[str], user_level: str) -> str:
        """Generate code implementation"""
        
        # This would normally use more sophisticated code generation
        # For demo, providing template-based generation
        
        if "api" in problem.lower():
            return self._generate_api_code(requirements, user_level)
        elif "database" in problem.lower():
            return self._generate_database_code(requirements, user_level)
        elif "performance" in problem.lower():
            return self._generate_performance_code(requirements, user_level)
        else:
            return self._generate_general_code(requirements, user_level)
    
    def _generate_api_code(self, requirements: List[str], user_level: str) -> str:
        """Generate API-specific code"""
        if user_level == "junior":
            return '''
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/data', methods=['GET'])
def get_data():
    """Get data endpoint with basic error handling"""
    try:
        # TODO: Implement your business logic here
        data = {"message": "Hello World", "status": "success"}
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/data', methods=['POST'])
def create_data():
    """Create data endpoint with validation"""
    try:
        data = request.get_json()
        
        # Basic validation
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # TODO: Process and save data
        result = {"id": 1, "data": data, "status": "created"}
        return jsonify(result), 201
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
'''
        else:
            return '''
from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import wraps
import logging
from typing import Dict, Any

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def handle_errors(f):
    """Decorator for consistent error handling"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            logger.warning(f"Validation error: {str(e)}")
            return jsonify({"error": "Invalid input", "details": str(e)}), 400
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
    return decorated_function

def validate_json(required_fields: list = None):
    """Decorator for JSON validation"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            data = request.get_json()
            if not data:
                return jsonify({"error": "JSON data required"}), 400
                
            if required_fields:
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    return jsonify({
                        "error": "Missing required fields",
                        "missing": missing_fields
                    }), 400
                    
            return f(data, *args, **kwargs)
        return decorated_function
    return decorator

@app.route('/api/data', methods=['GET'])
@handle_errors
def get_data():
    """Get data with pagination and filtering"""
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 10, type=int)
    filter_param = request.args.get('filter', '')
    
    # Validate pagination
    if page < 1 or limit < 1 or limit > 100:
        raise ValueError("Invalid pagination parameters")
    
    # TODO: Implement data retrieval with filtering
    data = {
        "items": [],  # Your data here
        "pagination": {
            "page": page,
            "limit": limit,
            "total": 0,
            "has_next": False
        },
        "filter": filter_param
    }
    
    logger.info(f"Data requested - page: {page}, limit: {limit}")
    return jsonify(data), 200

@app.route('/api/data', methods=['POST'])
@handle_errors
@validate_json(['name', 'type'])
def create_data(data: Dict[str, Any]):
    """Create data with comprehensive validation"""
    
    # TODO: Implement business logic validation
    # TODO: Save to database
    
    result = {
        "id": 1,  # Generated ID
        "data": data,
        "status": "created",
        "created_at": "2024-01-01T00:00:00Z"
    }
    
    logger.info(f"Data created: {result['id']}")
    return jsonify(result), 201

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
'''
    
    def _generate_explanation(self, problem: str, user_level: str) -> str:
        """Generate explanation based on user level"""
        if user_level == "junior":
            return """
**Giải thích từng bước:**

1. **Import libraries**: Flask cho web framework, CORS cho cross-origin requests
2. **Error handling**: Sử dụng try-catch để bắt lỗi và trả về response phù hợp
3. **JSON validation**: Kiểm tra input data trước khi xử lý
4. **Response format**: Consistent JSON response với status codes

**Cách sử dụng:**
- Chạy file Python này
- Test với Postman hoặc curl
- Modify business logic trong TODO sections
"""
        else:
            return """
**Architecture Overview:**

1. **Decorator Pattern**: Sử dụng decorators cho error handling và validation
2. **Separation of Concerns**: Tách biệt validation, business logic và response handling  
3. **Logging Strategy**: Comprehensive logging cho monitoring và debugging
4. **Error Handling**: Multi-level error handling với specific error types

**Advanced Features:**
- Type hints cho better code documentation
- Pagination support cho large datasets
- Configurable validation decorators
- Structured logging với context

**Production Considerations:**
- Add authentication/authorization
- Implement rate limiting
- Add API documentation (Swagger)
- Database connection pooling
- Monitoring và metrics
"""
    
    def _get_best_practices(self, problem: str) -> List[str]:
        """Get relevant best practices"""
        return [
            "🔐 Always validate input data",
            "📝 Use consistent error handling",
            "🧪 Write comprehensive tests",
            "📊 Implement proper logging",
            "⚡ Consider performance implications",
            "🔒 Follow security best practices",
            "📚 Document your API endpoints",
            "🚀 Use environment-specific configurations"
        ]
    
    def _estimate_complexity(self, problem: str) -> str:
        """Estimate problem complexity"""
        complexity_indicators = {
            "low": ["simple", "basic", "easy", "quick"],
            "medium": ["optimize", "improve", "enhance", "moderate"],
            "high": ["complex", "advanced", "scalable", "enterprise", "distributed"]
        }
        
        problem_lower = problem.lower()
        for level, indicators in complexity_indicators.items():
            if any(indicator in problem_lower for indicator in indicators):
                return level
        return "medium"
    
    def _detect_technologies(self, problem: str) -> List[str]:
        """Detect technologies mentioned in problem"""
        tech_patterns = {
            "flask": ["flask"],
            "django": ["django"],
            "fastapi": ["fastapi"],
            "postgresql": ["postgres", "postgresql"],
            "mysql": ["mysql"],
            "redis": ["redis"],
            "docker": ["docker", "container"],
            "aws": ["aws", "amazon"]
        }
        
        problem_lower = problem.lower()
        detected = []
        for tech, patterns in tech_patterns.items():
            if any(pattern in problem_lower for pattern in patterns):
                detected.append(tech)
        return detected
    
    def _get_approach_for_type(self, problem_type: str) -> str:
        """Get approach description for problem type"""
        approaches = {
            "api_development": "RESTful API design với proper error handling và validation",
            "database_optimization": "Query analysis, indexing strategy, connection pooling",
            "performance_optimization": "Profiling, caching, async processing, load testing",
            "testing": "Unit tests, integration tests, mocking strategies",
            "general_development": "Clean code principles, SOLID design patterns"
        }
        return approaches.get(problem_type, "General problem-solving approach")
    
    def _generate_database_code(self, requirements: List[str], user_level: str) -> str:
        """Generate database-related code"""
        return '''
# Database optimization example
from sqlalchemy import create_engine, Index
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

class DatabaseManager:
    def __init__(self, connection_string: str):
        self.engine = create_engine(
            connection_string,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False  # Set True for query logging
        )
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    @contextmanager
    def get_session(self):
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def optimize_queries(self):
        """Add indexes for common queries"""
        # Example: Add index for user lookups
        Index('idx_user_email', User.email)
        Index('idx_post_user_created', Post.user_id, Post.created_at)
'''
    
    def _generate_performance_code(self, requirements: List[str], user_level: str) -> str:
        """Generate performance optimization code"""
        return '''
# Performance optimization patterns
import asyncio
import aiohttp
from functools import lru_cache
import redis

class PerformanceOptimizer:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    @lru_cache(maxsize=128)
    def cached_computation(self, input_data: str) -> str:
        """Cache expensive computations"""
        # Expensive operation here
        return f"processed_{input_data}"
    
    async def batch_api_calls(self, urls: list) -> list:
        """Process multiple API calls concurrently"""
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_url(session, url) for url in urls]
            return await asyncio.gather(*tasks)
    
    async def fetch_url(self, session, url):
        async with session.get(url) as response:
            return await response.json()
    
    def cache_with_redis(self, key: str, value: any, expiration: int = 3600):
        """Cache data with Redis"""
        self.redis_client.setex(key, expiration, str(value))
    
    def get_from_cache(self, key: str):
        """Retrieve from cache"""
        return self.redis_client.get(key)
'''
    
    def _generate_general_code(self, requirements: List[str], user_level: str) -> str:
        """Generate general purpose code"""
        return '''
# General Python solution template
import logging
from typing import Optional, List, Dict
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Solution:
    """Template solution class"""
    name: str
    parameters: Dict
    
    def execute(self) -> Optional[any]:
        """Main execution method"""
        try:
            logger.info(f"Executing solution: {self.name}")
            
            # TODO: Implement your solution logic here
            result = self.process_data()
            
            logger.info("Solution executed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Solution failed: {str(e)}")
            raise
    
    def process_data(self) -> any:
        """Process data according to requirements"""
        # Implementation placeholder
        return {"status": "success", "data": self.parameters}
    
    def validate_input(self) -> bool:
        """Validate input parameters"""
        required_fields = ["field1", "field2"]  # Customize as needed
        return all(field in self.parameters for field in required_fields)

# Usage example
if __name__ == "__main__":
    solution = Solution(
        name="ExampleSolution",
        parameters={"field1": "value1", "field2": "value2"}
    )
    
    result = solution.execute()
    print(f"Result: {result}")
'''
    
    def _get_testing_strategy(self, problem: str) -> List[str]:
        """Get testing strategy recommendations"""
        return [
            "🧪 Unit tests cho individual functions",
            "🔗 Integration tests cho API endpoints", 
            "📊 Performance tests cho optimization",
            "🛡️ Security tests cho authentication",
            "🎭 Mock external dependencies",
            "📈 Test coverage >= 80%",
            "🚀 Automated testing trong CI/CD"
        ]
    
    def _get_next_steps(self, user_level: str) -> List[str]:
        """Get recommended next steps"""
        if user_level == "junior":
            return [
                "1. Implement basic functionality",
                "2. Add error handling",
                "3. Write simple tests",
                "4. Test manually với Postman",
                "5. Deploy to staging environment"
            ]
        else:
            return [
                "1. Review và refactor code architecture",
                "2. Add comprehensive test suite",
                "3. Implement monitoring và logging",
                "4. Performance testing và optimization",
                "5. Security audit và penetration testing",
                "6. Documentation và API specs",
                "7. Production deployment với CI/CD"
            ]

# Initialize analyzer and solution generator
code_analyzer = CodeAnalyzer()
solution_generator = SolutionGenerator()