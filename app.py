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
    Questions: {input}
    """
)

# Function to process the document sources
def vector_embedding(uploaded_file=None, url=None, abstract_text=None):
    documents = []

    if "embeddings" not in st.session_state:
        st.session_state.embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    # Process the abstract text input
    if abstract_text:
        from langchain_core.documents import Document
        documents.append(Document(page_content=abstract_text))

    # Process uploaded PDF file
    if uploaded_file:
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

    st.write("Updating the Vector Store Database...")

    # Initialize or append to session documents
    if "documents" not in st.session_state:
        st.session_state.documents = documents
    else:
        st.session_state.documents += documents

    # Split text into chunks and update vector store
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(st.session_state.documents)

    # Update or create the vector store
    st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)

    st.write("Vector Store Database is updated and ready for use!")


# New abstract text input section
abstract_text = st.text_area("Enter abstract text (optional)")

# Handle both URL and file upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
url = st.text_input("Enter a URL")

# If any new file, URL, or text is added, clear the query_mode and update
if uploaded_file or url or abstract_text:
    if 'query_mode' in st.session_state:  # Only clear if there's something to clear
        st.session_state.query_mode = ""  # Clear query mode
    st.write("Processing the input...")
    vector_embedding(uploaded_file=uploaded_file, url=url, abstract_text=abstract_text)
    #st.experimental_rerun()  # Force the rerun to reflect the cleared state

# Track whether an update is needed
if 'query_mode' not in st.session_state:
    st.session_state.query_mode = ""  # Initialize query_mode state

query_mode = st.text_input("Enter the question from the uploaded document or URL", value=st.session_state.query_mode)

# Process question if vector store is available
if query_mode and "vectors" in st.session_state:
    import time

    start = time.process_time()

    # Create the document chain for LLM processing
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Get the response for the query
    response = retrieval_chain.invoke({"input": query_mode})

    # Show response time and output
    st.write("Response Time: ", time.process_time() - start)
    st.write(response["answer"])

    # Show document similarity search
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("----------------------------------")
    query_mode = st.empty()
    time.sleep(5)
    
