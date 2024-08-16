import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import faiss
import time
from dotenv import load_dotenv

load_dotenv()

# Load the Groq API key and OpenAI API key
groq_api_key = os.environ["GROQ_API_KEY"]
openai_api_key = os.environ["OPENAI_API_KEY"]

if "vector" not in st.session_state:
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)

    # Load vector store index
    print("Loading vector store...")
    index = faiss.read_index("vector_store.index")
    st.session_state.vectors = FAISS(index=index)
    print("Vector store loaded.")

st.title("ChatGroq Demo")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-It")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>{context}</context>
    Question:{input}
    """
)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

user_prompt = st.text_input("Input your prompt here")

if user_prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({"Input": user_prompt})
    print("Response time:", time.process_time() - start)
    st.write(response['answer'])

    # With a Streamlit expander
    with st.expander("Document Similarity Search"):
        # Find relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-----------------------------")
