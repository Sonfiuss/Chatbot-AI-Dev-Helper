"""
Enhanced RAG Service for Workshop 4 - Code Assistant Chatbot
- FAISS vector store với code knowledge base (errors, best practices, patterns)
- LangChain ConversationalRetrievalChain cho context-aware responses
- Azure OpenAI function calling cho code analysis
- Hỗ trợ phân tích code, tìm lỗi, suggest improvements
"""
from __future__ import annotations

import os
import json
import ast
import re
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import AzureChatOpenAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate

from openai import AzureOpenAI

# Load code knowledge base từ file
def load_code_knowledge_base() -> List[Dict[str, Any]]:
    """Load code knowledge base từ JSON file"""
    data_path = Path(__file__).parent / "data" / "code_knowledge_base.json"
    
    if data_path.exists():
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # Fallback to minimal default docs if file not found
        return [
            {
                "page_content": "Common Python Error: AttributeError when accessing None. Always check if object is not None before accessing attributes.",
                "metadata": {"category": "common_errors", "language": "python", "severity": "high"}
            },
            {
                "page_content": "N+1 Query Problem in database queries. Use eager loading with join to prevent multiple queries.",
                "metadata": {"category": "performance", "language": "python", "severity": "high"}
            }
        ]

DEFAULT_DOCS = load_code_knowledge_base()

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "analyze_code_snippet",
            "description": "Analyzes Python code for bugs, security issues, performance problems, and style issues",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to analyze"},
                    "analysis_type": {
                        "type": "string", 
                        "enum": ["security", "performance", "bugs", "style", "complete"],
                        "description": "Type of analysis to perform"
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_error",
            "description": "Explains Python error messages and provides solutions",
            "parameters": {
                "type": "object",
                "properties": {
                    "error_message": {"type": "string", "description": "Error message or traceback"},
                    "code_context": {"type": "string", "description": "Code that caused the error (optional)"}
                },
                "required": ["error_message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_improvements",
            "description": "Suggests improvements and best practices for given code",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code to improve"},
                    "focus_area": {
                        "type": "string",
                        "enum": ["performance", "readability", "security", "maintainability", "all"],
                        "description": "Area to focus improvements on"
                    }
                },
                "required": ["code"],
            },
        },
    },
]

SYSTEM_PROMPT = """Bạn là Code Assistant Expert - chuyên gia phân tích code và debugging.

NHIỆM VỤ:
🔍 Phân tích code Python, tìm bugs, security issues, performance problems
💡 Giải thích error messages và suggest solutions
✨ Đề xuất improvements và best practices
📚 Sử dụng knowledge base để đưa ra câu trả lời accurate

CÁCH LÀM VIỆC:
1. Đọc và hiểu câu hỏi/vấn đề của user
2. Sử dụng retrieved knowledge từ code knowledge base
3. Gọi tools khi cần analyze code cụ thể
4. Đưa ra giải pháp chi tiết với code examples

RESPONSE FORMAT:
- Giải thích vấn đề rõ ràng
- Chỉ ra root cause (nếu là bug/error)
- Provide solution với code example
- Suggest best practices
- Include severity level nếu cần (🔴 Critical, 🟡 Major, 🔵 Minor)

Hãy helpful, precise, và actionable!"""


def analyze_code_snippet(code: str, analysis_type: str = "complete") -> str:
    """Analyze code for various issues"""
    issues = []
    
    # Basic syntax check
    try:
        ast.parse(code)
    except SyntaxError as e:
        return json.dumps({
            "status": "syntax_error",
            "error": f"Syntax Error at line {e.lineno}: {e.msg}",
            "severity": "critical"
        })
    
    # Security checks
    if analysis_type in ["security", "complete"]:
        security_patterns = [
            (r'eval\s*\(', 'CRITICAL', 'Use of eval() can lead to code injection'),
            (r'exec\s*\(', 'CRITICAL', 'Use of exec() can lead to code injection'),
            (r'pickle\.loads?', 'HIGH', 'Pickle deserialization can be unsafe'),
        ]
        for pattern, severity, desc in security_patterns:
            if re.search(pattern, code):
                issues.append({"type": "security", "severity": severity, "description": desc})
    
    # Performance checks
    if analysis_type in ["performance", "complete"]:
        if 'for' in code and '.append(' in code:
            issues.append({
                "type": "performance",
                "severity": "LOW",
                "description": "List comprehension might be more efficient than for-loop with append"
            })
    
    # Bug patterns
    if analysis_type in ["bugs", "complete"]:
        if re.search(r'if\s+\w+\s*=\s*', code):
            issues.append({
                "type": "bug",
                "severity": "HIGH",
                "description": "Assignment in if condition - did you mean == ?"
            })
        if 'except:' in code:
            issues.append({
                "type": "bug",
                "severity": "MEDIUM",
                "description": "Bare except clause - should specify exception type"
            })
    
    return json.dumps({
        "status": "analyzed",
        "issues_found": len(issues),
        "issues": issues,
        "recommendation": "Address critical and high severity issues first" if issues else "Code looks good!"
    }, indent=2)


def explain_error(error_message: str, code_context: str = "") -> str:
    """Explain error message and provide solution"""
    error_type = "Unknown error"
    
    if "AttributeError" in error_message:
        error_type = "AttributeError"
        explanation = "Trying to access attribute/method on None or object that doesn't have it"
        solution = "Check if object exists before accessing: if obj is not None: obj.method()"
    elif "IndexError" in error_message:
        error_type = "IndexError"
        explanation = "Trying to access list index that doesn't exist"
        solution = "Check list length: if len(list) > index: item = list[index]"
    elif "KeyError" in error_message:
        error_type = "KeyError"
        explanation = "Trying to access dictionary key that doesn't exist"
        solution = "Use .get() method: value = dict.get('key', default_value)"
    elif "TypeError" in error_message:
        error_type = "TypeError"
        explanation = "Operation on incompatible types or wrong number of arguments"
        solution = "Check function signature and argument types"
    else:
        explanation = "Check error traceback for exact line and cause"
        solution = "Read error message carefully and verify your code logic"
    
    return json.dumps({
        "error_type": error_type,
        "explanation": explanation,
        "solution": solution,
        "severity": "HIGH"
    }, indent=2)


def suggest_improvements(code: str, focus_area: str = "all") -> str:
    """Suggest code improvements"""
    suggestions = []
    
    # Check for docstrings
    if 'def ' in code and '"""' not in code and "'''" not in code:
        suggestions.append({
            "area": "readability",
            "suggestion": "Add docstrings to functions to document their purpose, parameters, and return values"
        })
    
    # Check for type hints
    if 'def ' in code and '->' not in code:
        suggestions.append({
            "area": "maintainability",
            "suggestion": "Add type hints to improve code clarity and catch type errors early"
        })
    
    # Check for error handling
    if 'try' not in code and ('open(' in code or 'requests.' in code or '.query' in code):
        suggestions.append({
            "area": "reliability",
            "suggestion": "Add try-except blocks for I/O operations and external API calls"
        })
    
    # Performance suggestions
    if focus_area in ["performance", "all"]:
        if 'for' in code and 'append' in code:
            suggestions.append({
                "area": "performance",
                "suggestion": "Consider using list comprehension instead of for-loop with append"
            })
    
    return json.dumps({
        "total_suggestions": len(suggestions),
        "suggestions": suggestions
    }, indent=2)


class RAGService:
    def __init__(
        self,
        docs: Optional[List[Dict[str, Any]]] = None,
        k: int = 3,
    ) -> None:
        self.docs = docs or DEFAULT_DOCS
        self.k = k

        # Env
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-07-01-preview")
        self.chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT") or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.embed_deployment = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")

        # Clients
        self._aoai_client: Optional[AzureOpenAI] = None
        self._retrieval_chain: Optional[ConversationalRetrievalChain] = None

    def build(self) -> None:
        # Embeddings + FAISS
        emb_kwargs = dict(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
        )
        if self.embed_deployment:
            emb_kwargs["azure_deployment"] = self.embed_deployment

        embeddings = AzureOpenAIEmbeddings(**emb_kwargs)
        texts = [d["page_content"] for d in self.docs]
        metas = [d.get("metadata", {}) for d in self.docs]
        vectorstore = FAISS.from_texts(texts, embedding=embeddings, metadatas=metas)
        retriever = vectorstore.as_retriever(search_kwargs={"k": self.k})

        # Chat model via LangChain for the retrieval chain
        chat_lc = AzureChatOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
            azure_deployment=self.chat_deployment,
            temperature=0.2,
        )
        self._retrieval_chain = ConversationalRetrievalChain.from_llm(
            llm=chat_lc, retriever=retriever, return_source_documents=True
        )

        # Azure OpenAI SDK client for tool-calling
        self._aoai_client = AzureOpenAI(
            api_key=self.api_key, api_version=self.api_version, azure_endpoint=self.endpoint
        )

    @property
    def ready(self) -> bool:
        return self._retrieval_chain is not None and self._aoai_client is not None

    def ensure_ready(self) -> None:
        if not self.ready:
            self.build()

    def _tool_execute(self, name: str, args: Dict[str, Any]) -> str:
        """Execute function calls for code analysis"""
        if name == "analyze_code_snippet":
            return analyze_code_snippet(**args)
        if name == "explain_error":
            return explain_error(**args)
        if name == "suggest_improvements":
            return suggest_improvements(**args)
        return f"Unknown tool: {name}"

    def chat(self, query: str, history: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Run retrieval + tool calling. Returns {answer, sources}."""
        self.ensure_ready()

        # 1) Retrieve with LangChain
        rag_result = self._retrieval_chain({"question": query, "chat_history": history})
        sources = rag_result.get("source_documents", [])
        knowledge = "\n".join(
            [f"- {d.metadata.get('source')}: {d.page_content}" for d in sources]
        )

        # 2) Tool-calling with Azure OpenAI
        knowledge_context = f"📚 KNOWLEDGE BASE:\n{knowledge}" if knowledge else ""
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": knowledge_context if knowledge_context else "No specific knowledge retrieved for this query."},
        ]
        for q, a in history:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": query})

        first = self._aoai_client.chat.completions.create(
            model=self.chat_deployment,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        msg = first.choices[0].message

        answer = msg.content or ""
        if getattr(msg, "tool_calls", None):
            # Append assistant message containing tool calls
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
                }
            )
            # Execute tools and append tool results
            for tc in msg.tool_calls:
                fname = tc.function.name
                raw_args = tc.function.arguments
                try:
                    parsed = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except Exception:
                    parsed = {}
                tool_output = self._tool_execute(fname, parsed)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": fname,
                        "content": tool_output,
                    }
                )

            # Finalization call
            second = self._aoai_client.chat.completions.create(
                model=self.chat_deployment,
                messages=messages,
            )
            answer = second.choices[0].message.content or ""

        return {
            "answer": answer,
            "sources": [
                {"source": d.metadata.get("source"), "content": d.page_content} for d in sources
            ],
        }
