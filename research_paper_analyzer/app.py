'''
    API end-point that handles the pdf-takein and analyzer
'''
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

from dotenv import load_dotenv

import uvicorn
import os

os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")

app = FastAPI(
    title = 'Chat with Research Paper',
    version = '1.0',
    description = 'Upload and chat with the research paper with gemma:2b model'
)


