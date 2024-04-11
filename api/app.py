from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langserve import  add_routes
from dotenv import load_dotenv

import uvicorn
import os

load_dotenv()

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

# fastAPI - endpoints (routes)

app = FastAPI(
    title = 'Langchain Test Server',
    version = '0.69',
    description = 'Langchain test server with multiple endpoints for topic to text generation prompts'
)

# all the prompts for each section
essay_prompt = ChatPromptTemplate.from_template('Write me an essay about {topic} with 100 words.')
poem_prompt = ChatPromptTemplate.from_template('Write me a poem about {topic} for a teenager with around 100 words.')

# gemma:2b model usage
gemma_llm = Ollama(model = 'gemma:2b')

# adding route for the essay generation
add_routes(
    app, 
    essay_prompt|gemma_llm,
    path = '/essay'
)

# adding route for the poem generation
add_routes(
    app, 
    poem_prompt|gemma_llm,
    path = '/poem'
)

if __name__ == '__main__':
    uvicorn.run(app, host = 'localhost', port = 8000)