# 🛡️ AI-Powered Insurance Policy Information Chatbot

This project is a chatbot application designed to assist users in understanding various insurance policies — including health, life, home, and auto — by answering questions using natural language understanding powered by large language models (LLMs).

Built using:
- **Streamlit** for the frontend interface
- **Cohere LLM & embeddings**
- **ChromaDB** for vector storage
- **LangGraph** to construct an agentic RAG (Retrieval-Augmented Generation) pipeline

---

## 💡 Problem Statement

Insurance customers often face difficulty understanding their policies, coverage options, premiums, and claim processes. Human customer support can be slow and expensive to scale. To address this, we've created an AI chatbot that can understand queries in natural language and return accurate, contextually relevant information using uploaded company policy documents (PDFs or TXT files).

---

## 🚀 Features

- Upload insurance documents (PDF/TXT)
- Automatic text processing and splitting
- Embedding with Cohere
- Vector storage using Chroma
- Document retrieval and relevance grading
- Dynamic query rewriting
- Multi-step reasoning using LangGraph agent
- Complete Streamlit UI
- Debugging of each step in the LangGraph agent flow

---

## 🧠 Why LangGraph?

LangGraph provides a powerful way to build **stateful agents with conditional logic**, unlike traditional LangChain pipelines that often require chained monolithic logic. LangGraph lets us:

- Visually and programmatically define **node-based workflows**
- Handle tools and LLM reasoning separately
- Dynamically **rewrite queries** if retrieved documents are irrelevant
- Integrate **debugging and conditional flows** naturally

This makes LangGraph ideal for building complex, decision-based RAG pipelines like this one.

---
