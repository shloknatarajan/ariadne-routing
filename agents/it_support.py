from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

it_support_agent = Agent(
    role="IT Support Agent",
    goal="Provide technical support for IT issues, diagnose and troubleshoot hardware and software problems.",
    backstory="Experienced in diagnosing software, hardware, and network issues. Known for quick response and practical solutions.",
    verbose=True,
    allow_delegation=False,
    llm=ChatOpenAI(),
)

task = Task(
    description="Troubleshoot a network connectivity issue experienced by an employee. Provide a step-by-step diagnostic process and suggest solutions.",
    agent=it_support_agent,
    expected_output="Detailed troubleshooting steps and resolutions."
)

crew = Crew(agents=[it_support_agent], tasks=[task], verbose=True)
result = crew.kickoff()
print(result)
