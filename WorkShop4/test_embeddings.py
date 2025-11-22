"""Test available embedding models"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv('AZURE_OPENAI_ENDPOINT'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY')
)

# Try different embedding models
embedding_models = [
    "text-embedding-ada-002",
    "text-embedding-3-small",
    "text-embedding-3-large"
]

print("Testing embedding models...")
print()

for model in embedding_models:
    try:
        print(f"Testing {model}...")
        response = client.embeddings.create(
            model=model,
            input="test"
        )
        print(f"  ✅ {model} works! Dimension: {len(response.data[0].embedding)}")
    except Exception as e:
        error_msg = str(e)
        if "400" in error_msg:
            print(f"  ❌ {model} not available")
        else:
            print(f"  ❌ {model} error: {str(e)[:100]}")
    print()
