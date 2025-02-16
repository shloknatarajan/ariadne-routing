from langchain_community.chat_models import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate
import os

def prompt_perplexity(prompt):
    chat = ChatPerplexity(api_key=os.getenv("pplx_api_key"), temperature=0.7, model="llama-3.1-sonar-small-128k-online")
    system = "You are a helpful assistant."
    human = "{input}"
    prompt_template = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

    chain = prompt_template | chat
    response = chain.invoke({"input": prompt})
    return response.content

