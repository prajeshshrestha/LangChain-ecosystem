from fastapi import FastAPI
from langchain_community.document_loaders import PyPDFLoader # loader
from langchain.text_splitter import RecursiveCharacterTextSplitter # splitter
from langchain_community.embeddings import OllamaEmbeddings # vectorizer
from langchain_community.vectorstores import FAISS # vector stores (database)
from langchain_community.llms import Ollama # local (LLM)
from langchain_core.prompts import ChatPromptTemplate # basic prompt designing (one-shot prompting)
from langchain_groq import ChatGroq

# CHAINs and RETRIEVERs
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv

import streamlit as st
import os

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

st.title("Chat with you Research Paper")

if 'db_' not in st.session_state:
    st.session_state['db_'] = None

if 'pdf_uploaded' not in st.session_state:
    st.session_state['pdf_uploaded'] = False

if 'local_llm' not in st.session_state:
    st.session_state['local_llm'] = False

if 'groq_llm' not in st.session_state:
    st.session_state['groq_llm'] = False


# bunch of callback functions
def select_local_llm():
    st.session_state['pdf_uploaded'] = True
    st.session_state['local_llm'] = True

def select_groq_llm():
    st.session_state['pdf_uploaded'] = True
    st.session_state['groq_llm'] = True

if not st.session_state['pdf_uploaded']:
    pdf_file = st.file_uploader("Upload you .pdf file", type = ['pdf'])

    if pdf_file:
        with open(pdf_file.name, "wb") as file:
            file.write(pdf_file.getvalue())
            file_name = pdf_file.name

        # pdf loader
        pdf_loader = PyPDFLoader(f"./{pdf_file.name}")
        pdf_docs = pdf_loader.load()

        # delete the temp pdf file
        os.remove(os.path.join('./', pdf_file.name))

        # splitter
        pdf_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 69)
        documents = pdf_splitter.split_documents(pdf_docs)

        with st.spinner("Vectorizer the supplied PDF file ..."):
            print("Inside spinner")
            # instantiate vector embedder and create vector database
            gemma_2b_llm = OllamaEmbeddings(model = 'gemma:2b')
            
            db = FAISS.from_documents(documents[:1], gemma_2b_llm)
            st.session_state['db_'] = db
            print("-"*100)
            print(st.session_state['db_'])
            st.success("File uploader and vectorized.")
            st.button("Chat using Gemma:2b (Local LLM)", on_click = select_local_llm)
            st.button("Chat using Mixtral (Groq API)", on_click = select_groq_llm)
            
if st.session_state['local_llm']:
    # setup local ollama (gemma 2b) model 
    gemma_llm = Ollama(model = 'gemma:2b')

    # create prompt design
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context. 
    Think step by step before providing a detailed answer. 
    <context>
    {context}
    </context>
    Question: {input}""")

    # document_chaining
    document_chain = create_stuff_documents_chain(gemma_llm, prompt)

    db = st.session_state['db_']
    
    if db is not None:
        retriever = db.as_retriever()

        # retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        input_text = st.text_input("Chat with the Research Paper.")
        if input_text:
            response = retrieval_chain.invoke({'input': input_text})
            st.write(response['answer'])

elif st.session_state['groq_llm']:

    groq_model = ChatGroq(temperature = 0, model_name="mixtral-8x7b-32768")

    # create prompt design
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context. 
    Think step by step before providing a detailed answer. 
    <context>
    {context}
    </context>
    Question: {input}""")

    # document_chaining
    document_chain = create_stuff_documents_chain(groq_model, prompt)

    db = st.session_state['db_']
    
    if db is not None:
        retriever = db.as_retriever()

        # retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        input_text = st.text_input("Chat with the Research Paper.")
        if input_text:
            response = retrieval_chain.invoke({'input': input_text})
            st.write(response['answer'])
