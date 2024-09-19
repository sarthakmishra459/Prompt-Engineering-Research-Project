import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from dotenv import load_dotenv
import tempfile

load_dotenv()

# Load groq and hugging face API
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

st.title("Chatgroq with llama3")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

def vector_embedding(uploaded_file=None, url=None):
    # Only update the vector store if a new file or URL is provided
    documents = []

    if "embeddings" not in st.session_state:
        st.session_state.embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    # Process uploaded PDF file
    if uploaded_file:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        documents += loader.load()

    # Process URL
    if url:
        try:
            web_loader = WebBaseLoader(url)
            documents += web_loader.load()
        except Exception as e:
            st.error(f"Failed to load data from URL: {e}")

    if not documents:
        st.error("No documents to process.")
        return

    # Notify user that the database is being updated
    st.write("Updating the Vector Store Database...")

    # If there are already vectors, retrieve and combine old and new documents
    if "documents" in st.session_state:
        documents = st.session_state.documents + documents

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(documents)

    # Store documents in session state for future use
    st.session_state.documents = documents

    # Create vector store from documents (rebuild)
    st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)

    # Notify user that the update is complete
    st.write("Vector Store Database is updated and ready for use!")

# Handle both URL and file upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
url = st.text_input("Enter a URL")

# Only update the vector store when a new file or URL is uploaded
if uploaded_file or url:
    st.write("Processing the input...")
    vector_embedding(uploaded_file=uploaded_file, url=url)

# Question asking process (no vector store update)
prompt1 = st.text_input("Enter the question from the uploaded document or URL")

if prompt1 and "vectors" in st.session_state:
    import time

    start = time.process_time()

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": prompt1})

    st.write("Response Time: ", time.process_time() - start)
    st.write(response["answer"])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("----------------------------------")
