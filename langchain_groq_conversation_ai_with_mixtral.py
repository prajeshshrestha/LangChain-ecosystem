from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

import os
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

model = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You're a very knowledgeable Machine Learning Engineer."),
        ("human", "{question}"),
    ]
)
runnable = prompt | model | StrOutputParser()

response = runnable.invoke({'question': "What is google?"})
print(response)