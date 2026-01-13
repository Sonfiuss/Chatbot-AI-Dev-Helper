# Workshop 4: RAG Chatbot with FAISS, LangChain, and Azure OpenAI Function Calling

This workshop builds an end-to-end Retrieval-Augmented Generation (RAG) chatbot that retrieves knowledge from a FAISS vector store and augments Azure OpenAI chat generation. It also demonstrates Azure OpenAI tool/function calling to extend capabilities (e.g., check device status, create support tickets).

## Deliverables Covered
- Mock problem domain: IT Helpdesk (swap for your team’s domain as needed)
- FAISS vector store populated with embeddings
- LangChain conversational retrieval chain
- Azure OpenAI tool/function calling (check_system_status, create_it_ticket)
- Fully working notebook prototype (`rag_chatbot_workshop.ipynb`)
- Sample multi-turn conversation cells

## Setup

1) (Optional) Create and activate a virtual environment.

2) Install dependencies in this folder:

```
pip install -r requirements.txt
```

3) Set environment variables (recommended via a `.env` file at repo root or this folder):

```
AZURE_OPENAI_API_KEY=...                   # Your Azure OpenAI key
AZURE_OPENAI_ENDPOINT=https://<name>.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-07-01-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o-mini   # Your Chat deployment name
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=text-embedding-3-large  # Your Embeddings deployment name
```

Notes:
- Ensure the deployment names match your Azure OpenAI deployments.
- The notebook will warn if variables are missing.

## Run the Notebook

- Open `rag_chatbot_workshop.ipynb` in VS Code or Jupyter and run cells from top to bottom.
- A quick single-turn and multi-turn demo are provided near the end.

## Run the Streamlit App

After setting environment variables and installing requirements:

```
streamlit run app.py
```

This launches a simple chat UI that performs retrieval over the mock IT FAQ, calls tools when helpful (check status / create ticket), and displays source snippets.

## Customize the Domain

Replace the `mock_docs` list in the notebook with your own mock/business data. You can add more fields in `metadata` (e.g., category, owner) and tweak retriever settings (`k`, search type) to your needs.

## Troubleshooting

- If embeddings or chat calls fail, double-check the endpoint, api version, api key, and deployment names.
- If FAISS import fails on Windows, ensure you’re using Python 3.9+ and the `faiss-cpu` wheel is installed successfully.
- If you see missing env warnings, set variables in a `.env` or your shell and re-run the first cells.

## Next Steps (Ideas)

- Add more tools (e.g., knowledge base updates, ticket status lookup)
- Log chat transcripts and tool calls
- Add a lightweight UI (e.g., Streamlit) that wraps the same retrieval + tool-calling flow
- Replace FAISS with a managed vector DB like Pinecone or Azure AI Search
