**AI-Powered Legal Q&A Agent:**
This project is an intelligent Question and Answer (Q&A) Agent designed to assist legal professionals in analyzing and navigating complex PDF documents. By leveraging Retrieval-Augmented Generation (RAG), the agent acts as a knowledgeable intermediary that can understand the specific context of a legal agreement and provide grounded, accurate answers.

**Core Capabilities:**
Context-Aware Q&A: The agent doesn't just search for keywords; it uses Google Gemini and LangChain to understand the semantics of your questions and provide precise answers based on the uploaded text.

Intelligent Retrieval: Built with a FAISS vector store, the agent efficiently retrieves the most relevant document excerpts to answer queries about clauses, obligations, and terms.

Smart Summarization: Beyond direct Q&A, the agent can autonomously generate a structured summary focusing on key parties, durations, and critical deadlines.

Streamlit Interface: Offers a user-friendly, browser-based dashboard that allows for seamless PDF uploads and real-time interaction with the AI assistant.

**Tech Stack:**
AI Framework: LangChain (RetrievalQA, PromptTemplate)

LLM & Embeddings: Google Gemini (GenerativeAI)

Vector Database: FAISS

UI Framework: Streamlit
