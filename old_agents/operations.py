from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

operations_agent = Agent(
    role="Operations Agent",
    goal="Oversee daily business operations, streamline workflows, and enhance overall operational efficiency.",
    backstory="Experienced in process optimization and operational management across diverse business functions.",
    verbose=True,
    allow_delegation=False,
    llm=ChatOpenAI(),
)

task = Task(
    description="Provide a plan to improve operational efficiency at a mid-sized manufacturing plant, including process optimization and resource management.",
    agent=operations_agent,
    expected_output="A detailed operational improvement plan with key process optimizations and resource allocation strategies."
)

crew = Crew(agents=[operations_agent], tasks=[task], verbose=True)
result = crew.kickoff()
print(result)
