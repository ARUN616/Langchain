import streamlit as st
import os
from dotenv import load_dotenv
from langchain.llms import OpenAI

# Load environment variables from .env file
load_dotenv(r'C:\Users\Arun\AI tamil\NLP\Langchain\.env')

def get_openai_response(question):
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Debug print to check if API key is loaded
    print("OPENAI_API_KEY:", api_key)
    
    if not api_key:
        raise ValueError("API key not found. Check your .env file.")
    
    # Initialize OpenAI with the updated model name
    llm = OpenAI(openai_api_key=api_key, model_name="gpt-3.5-turbo", temperature=0.6)
    response = llm(question)
    return response

# Initialize Streamlit
st.set_page_config(page_title="Q&A Demo")
st.header("Langchain Application")

input_text = st.text_input("Input: ", key="input")
submit = st.button("Ask the question")

if submit:
    response = get_openai_response(input_text)
    st.subheader("The Response is")
    st.write(response)
