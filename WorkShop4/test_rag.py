"""Test RAG Service initialization"""
from dotenv import load_dotenv
load_dotenv()

print("Testing RAG Service initialization...")
print()

try:
    from rag_service import RAGService
    
    print("Creating RAG service...")
    rag = RAGService()
    
    print(f"Using standard OpenAI API: {rag.use_standard_openai}")
    print(f"Endpoint: {rag.endpoint}")
    print(f"Chat deployment: {rag.chat_deployment}")
    print()
    
    print("Building RAG service (this may take a moment)...")
    rag.build()
    
    print("✅ RAG service initialized successfully!")
    print(f"Ready: {rag.ready}")
    print()
    
    # Test a simple query
    print("Testing a simple query...")
    result = rag.chat("Hello, how can you help me?", [])
    print(f"Response: {result['answer'][:200]}...")
    print()
    print("✅ All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
