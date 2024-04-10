import streamlit as st
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_API_KEY'] = 'ls__70ac0c7d69d2495085aa3214029d50d7'

prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Please response to the user queries."),
            ("user", "Question: {question}")
        ]
)

## streamlit framework

st.title("Langchain Demo with Gemma Model: ")
input_text = st.text_input("What do you want to know about huhh: ")

## ollama gemma model

llm = Ollama(model = 'gemma:2b')
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))