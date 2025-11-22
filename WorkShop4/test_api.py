"""Test API connection with the Elevate AI Ready configuration"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

print("Testing API connection...")
print(f"Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
print(f"Model: {os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT')}")
print()

try:
    client = OpenAI(
        base_url=os.getenv('AZURE_OPENAI_ENDPOINT'),
        api_key=os.getenv('AZURE_OPENAI_API_KEY')
    )

    response = client.chat.completions.create(
        model=os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT'),
        messages=[{"role": "user", "content": "Hello! Reply with just 'OK'"}],
        max_tokens=10
    )

    print("✅ API Connection successful!")
    print(f"Response: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"❌ API Connection failed: {e}")
    import traceback
    traceback.print_exc()
