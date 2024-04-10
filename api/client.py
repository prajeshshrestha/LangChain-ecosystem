# creating a simplistic frontend to interact with the app.py (API endpoints - testing)
import requests
import streamlit as st

def get_ollama_response(input_text, is_essay = True):
    if is_essay:
        response = requests.post("http://localhost:8000/essay/invoke", json = {'input': {'topic':input_text}})
    else:
        response = requests.post("http://localhost:8000/poem/invoke", json = {'input': {'topic':input_text}})
    return response.json()['output']

## streamlit framework
st.title("Langchain with 'gemma:2b'")

input_text_essay = st.text_input('Write an essay on: ')
input_text_poem = st.text_input("Write an poem on: ")

if input_text_essay:
    st.write(get_ollama_response(input_text_essay, True))

if input_text_poem:
    st.write(get_ollama_response(input_text_essay, False))
