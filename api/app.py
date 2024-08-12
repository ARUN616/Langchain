from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI  # Updated import
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import ollama
from dotenv import load_dotenv

load_dotenv()

# Set the API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize the FastAPI app correctly
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API server"
)

# Initialize the ChatOpenAI model with the API key
model = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Initialize the ollama model
llm = ollama.Ollama(model="llama2")

# Create prompts
prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with 50 words")
prompt2 = ChatPromptTemplate.from_template("Write me a poem about {topic} with 50 words")

# Add routes with the model and prompt combined
add_routes(
    app,
    prompt1 | model,
    path="/essay"
)

add_routes(
    app,
    prompt2 | llm,
    path="/poem"
)

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
