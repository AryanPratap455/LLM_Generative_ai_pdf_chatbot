import os
import logging
import streamlit as st
from config import apikey
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set API key
API_KEY = apikey
os.environ["HUGGINGFACEHUB_API_TOKEN"] = API_KEY

# Load the model and tokenizer
def load_model_and_tokenizer(model_name):
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        text_gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=50, temperature=0.01)
        return HuggingFacePipeline(pipeline=text_gen_pipeline)
    except Exception as e:
        logger.error(f"Error loading model and tokenizer: {e}")
        return None

# Load PDF
def load_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        logger.error(f"Error loading PDF: {e}")
        return None

# Initialize the model and vector store
model_name = "Groq/Llama-3-Groq-8B-Tool-Use"
llm = load_model_and_tokenizer(model_name)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
vector_store = None

# Streamlit application
st.title("PDF Q&A Application")

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    pdf_text = load_pdf(uploaded_file)
    if pdf_text:
        documents = [Document(page_content=pdf_text)]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        text_chunks = text_splitter.split_documents(documents)

        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cpu"})
            vector_store = FAISS.from_documents(text_chunks, embeddings)
            st.success("Embeddings and Vector Store created successfully.")
            st.write("Text Chunks:")
            for chunk in text_chunks:
                st.write(chunk.page_content)
        except Exception as e:
            logger.error(f"Error creating embeddings and vector store: {e}")
            st.error(f"Error: {str(e)}")

# Question input
if vector_store:
    question = st.text_input("Ask a question about the PDF:")
    
    if st.button("Submit"):
        if not llm or not vector_store:
            st.error("Model and vector store are not initialized.")
        elif not question:
            st.error("No question provided.")
        else:
            try:
                chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    chain_type='stuff',
                    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                    memory=memory
                )
                result = chain({"question": question, "chat_history": memory})
                st.write("Answer:", result["answer"])
            except Exception as e:
                logger.error(f"Error in conversation: {e}")
                st.error(f"Error: {str(e)}")
