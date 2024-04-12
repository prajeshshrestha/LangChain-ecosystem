from fastapi import FastAPI
from langchain_community.document_loaders import PyPDFLoader # loader
from langchain.text_splitter import RecursiveCharacterTextSplitter # splitter
from langchain_community.embeddings import OllamaEmbeddings # vectorizer
from langchain_community.vectorstores import FAISS # vector stores (database)
from langchain_community.llms import Ollama # local (LLM)
from langchain_core.prompts import ChatPromptTemplate # basic prompt designing (one-shot prompting)

# CHAINs and RETRIEVERs
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

import streamlit as st
import os

st.title("Chat with you Research Paper")

# initialize some session state keys
# Initialize session state variables
if 'pdf_upload' not in st.session_state:
    st.session_state['pdf_upload'] = False
if 'vectorizer_completion' not in st.session_state:
    st.session_state['vectorizer_completion'] = False
if 'show_file_uploader' not in st.session_state:
    st.session_state['show_file_uploader'] = True


if st.session_state['show_file_uploader']:
    pdf_file = st.file_uploader("Upload you .pdf file", type = ['pdf'], disabled = st.session_state['pdf_upload'])

    if pdf_file:
        st.session_state['pdf_upload'] = True
        with open(pdf_file.name, "wb") as file:
            file.write(pdf_file.getvalue())
            file_name = pdf_file.name

        # pdf loader
        pdf_loader = PyPDFLoader(f"./{pdf_file.name}")
        pdf_docs = pdf_loader.load()

        # delete the temp pdf file
        os.remove(os.path.join('./', pdf_file.name))

        # this point flags about pdf file being successfully uploaded
        st.session_state['pdf_upload'] = True

        # splitter
        pdf_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 69)
        documents = pdf_splitter.split_documents(pdf_docs)

        with st.spinner("Vectorizer the supplied PDF file ..."):
            # instantiate vector embedder and create vector database
            gemma_2b_llm = OllamaEmbeddings(model = 'gemma:2b')
            db = FAISS.from_documents(documents[:1], gemma_2b_llm)
            st.session_state['vectorizer_completion'] = True
            st.session_state['show_file_uploader'] = False
            st.success("File uploader and vectorized.")


if not st.session_state['show_file_uploader']:
    
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

    # retrievers
    retriever = db.as_retriever()

    # retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    input_text = st.text_input("Chat with the Research Paper.")
    if input_text:
        response = retrieval_chain.invoke({'input': input_text})
        st.write(response['answer'])
