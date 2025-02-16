from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

supply_chain_agent = Agent(
    role="Supply Chain Agent",
    goal="Optimize the supply chain process, manage vendor relations, and ensure timely delivery of products.",
    backstory="Well-versed in logistics and supply chain management with a focus on cost reduction and efficiency improvements.",
    verbose=True,
    allow_delegation=False,
    llm=ChatOpenAI(),
)

task = Task(
    description="Analyze the current supply chain for a consumer electronics company and recommend strategies to reduce lead times and lower logistics costs.",
    agent=supply_chain_agent,
    expected_output="A detailed supply chain optimization report including recommendations for process improvements."
)

crew = Crew(agents=[supply_chain_agent], tasks=[task], verbose=True)
result = crew.kickoff()
print(result)
