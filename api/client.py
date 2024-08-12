import requests
import streamlit as st 

def get_openai_response(input_text):
    response = requests.post("http://localhost:8000/essay/invoke",
                             json={"input": {"topic": input_text}})
    # Debug: Print the response content and status code
    st.write("OpenAI Response Status:", response.status_code)
    st.write("OpenAI Response Content:", response.text)

    try:
        return response.json()['output']['content']
    except (KeyError, TypeError) as e:
        st.error("Error processing OpenAI response: " + str(e))
        return "Failed to get OpenAI response."


def get_ollama_response(input_text):
    response = requests.post("http://localhost:8000/poem/invoke",
                             json={"input": {"topic": input_text}})
    # Debug: Print the response content and status code
    st.write("Ollama Response Status:", response.status_code)
    st.write("Ollama Response Content:", response.text)

    try:
        # Directly access the 'output' key
        return response.json()['output']
    except (KeyError, TypeError) as e:
        st.error("Error processing Ollama response: " + str(e))
        return "Failed to get Ollama response."



# Streamlit framework
st.title("Langchain demo with OpenAI API")

input_text = st.text_input("Write an essay on")
input_text1 = st.text_input("Write a poem on")

if input_text:
    st.write(get_openai_response(input_text))

if input_text1:
    st.write(get_ollama_response(input_text1))
