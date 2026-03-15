# This is a Streamlit application that allows users to upload legal PDF documents, extract text, create a vector store using FAISS, and interact with the document using a Retrieval-Augmented Generation (RAG) pipeline powered by Google Gemini. Users can ask questions about the document or generate a summary of its contents. The application is designed to assist legal professionals in reviewing and analyzing legal documents efficiently.
import streamlit as st
import PyPDF2
from io import BytesIO
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# Page configuration
st.set_page_config(
    page_title="Legal Document Review Assistant",
    page_icon="⚖️",
    layout="wide"
)

# Title and description
st.title("⚖️ Legal Document Review Application")
st.markdown("""
This application helps legal professionals analyze legal documents using AI.
Upload a PDF document, ask questions, and get intelligent summaries.
""")

# Sidebar for API key input
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter Google API Key", type="password", help="Your Gemini API key")
    st.markdown("---")
    st.markdown("""
    ### How to use:
    1. Enter your Google Gemini API key
    2. Upload a legal PDF document
    3. Ask questions or generate summary
    4. Review the AI-powered insights
    """)
    st.markdown("---")
    st.info("💡 Supports text-based PDFs only. Scanned documents may not work properly.")

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'chunks' not in st.session_state:
    st.session_state.chunks = []

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))

        # Check if PDF is encrypted
        if pdf_reader.is_encrypted:
            st.error("❌ This PDF is encrypted. Please upload an unencrypted document.")
            return None

        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        if not text.strip():
            st.error("❌ No text could be extracted. This might be a scanned PDF.")
            return None

        return text
    except Exception as e:
        st.error(f"❌ Error extracting text: {str(e)}")
        return None

def chunk_text(text):
    """Split text into chunks for embedding"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(chunks, api_key):
    """Create FAISS vector store from text chunks"""
    try:
        # Create embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )

        # Convert chunks to Document objects
        documents = [Document(page_content=chunk) for chunk in chunks]

        # Create FAISS vector store
        vector_store = FAISS.from_documents(documents, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"❌ Error creating vector store: {str(e)}")
        return None

def answer_question(question, vector_store, api_key):
    """Answer questions using RAG pipeline"""
    try:
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.3
        )

        # Create custom prompt template
        prompt_template = """You are a legal assistant analyzing a legal document.
        Use the following context to answer the question accurately and concisely.
        If you cannot find the answer in the context, say so clearly.

        Context: {context}

        Question: {question}

        Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Create RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
            chain_type_kwargs={"prompt": PROMPT}
        )

        # Get answer
        result = qa_chain.invoke({"query": question})
        return result['result']
    except Exception as e:
        st.error(f"❌ Error answering question: {str(e)}")
        return None

def generate_summary(vector_store, api_key):
    """Generate document summary"""
    try:
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.3
        )

        # Retrieve relevant chunks
        retriever = vector_store.as_retriever(search_kwargs={"k": 6})
        docs = retriever.get_relevant_documents("summary of main points, key clauses, obligations, and terms")

        # Combine chunks
        context = "\n\n".join([doc.page_content for doc in docs])

        # Create summary prompt
        summary_prompt = f"""You are a legal assistant. Provide a concise summary of this legal document.
        Focus on:
        - Main purpose of the document
        - Key parties involved
        - Important clauses and obligations
        - Critical terms and conditions
        - Notable deadlines or durations

        Document excerpts:
        {context}

        Summary:"""

        # Generate summary
        summary = llm.invoke(summary_prompt)
        return summary.content
    except Exception as e:
        st.error(f"❌ Error generating summary: {str(e)}")
        return None

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload a legal PDF document",
        type=['pdf'],
        help="Select a text-based PDF file"
    )

    if uploaded_file is not None and api_key:
        with st.spinner("Extracting text from PDF..."):
            extracted_text = extract_text_from_pdf(uploaded_file)

            if extracted_text:
                st.session_state.extracted_text = extracted_text
                st.success("✅ Text extracted successfully!")

                # Show preview
                with st.expander("📖 Preview Extracted Text"):
                    st.text_area(
                        "Document Preview",
                        extracted_text[:2000] + "..." if len(extracted_text) > 2000 else extracted_text,
                        height=300
                    )

                # Chunk and create vector store
                with st.spinner("Processing document and creating embeddings..."):
                    chunks = chunk_text(extracted_text)
                    st.session_state.chunks = chunks

                    vector_store = create_vector_store(chunks, api_key)
                    if vector_store:
                        st.session_state.vector_store = vector_store
                        st.success(f"✅ Document processed! Created {len(chunks)} chunks.")
    elif uploaded_file and not api_key:
        st.warning("⚠️ Please enter your Google API key in the sidebar first.")

with col2:
    st.header("💬 Ask Questions")

    if st.session_state.vector_store is not None:
        # Question input
        question = st.text_input(
            "Ask a question about the document",
            placeholder="e.g., What are the terms for termination?"
        )

        col_q1, col_q2 = st.columns(2)

        with col_q1:
            if st.button("Get Answer", type="primary", use_container_width=True):
                if question:
                    with st.spinner("Analyzing document..."):
                        answer = answer_question(question, st.session_state.vector_store, api_key)
                        if answer:
                            st.markdown("### Answer:")
                            st.info(answer)
                else:
                    st.warning("Please enter a question.")

        with col_q2:
            if st.button("Generate Summary", use_container_width=True):
                with st.spinner("Generating summary..."):
                    summary = generate_summary(st.session_state.vector_store, api_key)
                    if summary:
                        st.markdown("### Document Summary:")
                        st.success(summary)

        # Sample questions
        st.markdown("---")
        st.markdown("### 💡 Sample Questions:")
        sample_questions = [
            "What are the terms for termination?",
            "What is the duration of the confidentiality obligation?",
            "Who are the parties involved in this agreement?",
            "What are the payment terms?",
            "What are the key obligations of each party?"
        ]
        for sq in sample_questions:
            st.markdown(f"- {sq}")
    else:
        st.info("👆 Upload a document to start asking questions")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with LangChain, FAISS, and Google Gemini | Legal Document Review Assistant</p>
</div>
""", unsafe_allow_html=True)