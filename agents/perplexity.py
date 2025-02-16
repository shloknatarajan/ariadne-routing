from langchain_community.chat_models import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate

def prompt_perplexity(prompt):
    chat = ChatPerplexity(temperature=0.7, model="llama-3.1-sonar-small-128k-online")
    prompt = ChatPromptTemplate.from_messages([("human",prompt)])
    chain = prompt | chat
    response = chain.invoke({"topic": "cats"})
    return response.content