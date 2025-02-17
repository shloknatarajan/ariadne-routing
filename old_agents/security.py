from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

security_agent = Agent(
    role="Security Agent",
    goal="Monitor and manage cybersecurity threats, vulnerabilities, and implement security measures.",
    backstory="Cybersecurity expert with extensive experience in threat detection and security protocol implementation.",
    verbose=True,
    allow_delegation=False,
    llm=ChatOpenAI(),
)

task = Task(
    description="Review current security protocols and suggest improvements to protect against emerging cyber threats.",
    agent=security_agent,
    expected_output="A comprehensive security assessment with recommended improvements."
)

crew = Crew(agents=[security_agent], tasks=[task], verbose=True)
result = crew.kickoff()
print(result)
