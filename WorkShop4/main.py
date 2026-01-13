"""
Workshop 4 - Console RAG Chatbot

This is a simple CLI runner that uses rag_service.RAGService to:
- Build a FAISS index over mock docs (or your custom docs inside rag_service)
- Retrieve relevant context via LangChain
- Use Azure OpenAI tool-calling (check_system_status, create_it_ticket)

Usage:
  python main.py

Type 'exit' or 'quit' to stop.
"""

from __future__ import annotations

import os
import sys
import textwrap
from typing import List, Tuple

try:
	from dotenv import load_dotenv
except Exception:
	load_dotenv = None

try:
	from rag_service import RAGService
except Exception as e:
	print("Failed to import RAGService from rag_service.py:", e)
	print("Make sure you're running from the 'WorkShop4' folder and dependencies are installed.")
	sys.exit(1)


def check_env() -> None:
	required = [
		"AZURE_OPENAI_API_KEY",
		"AZURE_OPENAI_ENDPOINT",
		"AZURE_OPENAI_API_VERSION",
		"AZURE_OPENAI_CHAT_DEPLOYMENT",
		"AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT",
	]
	missing = [k for k in required if not os.getenv(k)]
	if missing:
		print("[WARN] Missing environment variables:", ", ".join(missing))
		print("       Create a .env file or set them in your shell before running.")


def print_banner() -> None:
	print("=" * 70)
	print("🤖 Workshop 4 - RAG Chatbot (Console)")
	print("=" * 70)
	print("This app will retrieve knowledge via FAISS + LangChain and can call tools")
	print("like check_system_status or create_it_ticket through Azure OpenAI.")
	print("Type 'exit' or 'quit' to stop.\n")


def main() -> None:
	if load_dotenv:
		load_dotenv()
	check_env()
	print_banner()

	# Initialize service
	try:
		rag = RAGService()
		rag.build()
		print("[OK] RAG service initialized.")
	except Exception as e:
		print("[ERROR] Failed to initialize RAG service:", e)
		sys.exit(1)

	chat_history: List[Tuple[str, str]] = []

	while True:
		try:
			query = input("You: ").strip()
		except (EOFError, KeyboardInterrupt):
			print("\nBye.")
			break

		if query.lower() in ("exit", "quit"):
			print("Bye.")
			break
		if not query:
			continue

		try:
			result = rag.chat(query, chat_history)
			answer = result.get("answer", "")
			sources = result.get("sources", [])

			print("\nAssistant:")
			print(textwrap.fill(answer or "(no answer)", width=100))

			if sources:
				print("\nSources:")
				for s in sources:
					src = s.get("source") or "unknown"
					snippet = s.get("content") or ""
					print(" -", src)
			print("")

			chat_history.append((query, answer))
		except Exception as e:
			print("[ERROR] Chat failed:", e)


if __name__ == "__main__":
	main()

