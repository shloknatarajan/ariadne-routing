from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

devops_agent = Agent(
    role="DevOps Agent",
    goal="Streamline development operations, manage CI/CD pipelines, and optimize deployment processes.",
    backstory="DevOps engineer with expertise in automation, containerization, and cloud infrastructure.",
    verbose=True,
    allow_delegation=False,
    llm=ChatOpenAI(),
)

task = Task(
    description="Create a deployment strategy for a microservices-based application, including CI/CD pipeline setup.",
    agent=devops_agent,
    expected_output="A comprehensive deployment strategy with CI/CD pipeline configuration details."
)

crew = Crew(agents=[devops_agent], tasks=[task], verbose=True)
result = crew.kickoff()
print(result)
