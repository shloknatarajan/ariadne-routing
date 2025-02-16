from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

compliance_agent = Agent(
    role="Compliance Agent",
    goal="Ensure that enterprise operations adhere to regulatory standards and internal policies.",
    backstory="Highly knowledgeable in regulatory frameworks and best practices, dedicated to maintaining compliance across all operational areas.",
    verbose=True,
    allow_delegation=False,
    llm=ChatOpenAI(),
)

task = Task(
    description="Review the company's internal policies against the latest regulatory guidelines in the industry and provide a compliance audit report.",
    agent=compliance_agent,
    expected_output="An audit report outlining compliance gaps and recommended improvements."
)

crew = Crew(agents=[compliance_agent], tasks=[task], verbose=True)
result = crew.kickoff()
print(result)
