from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

software_qa_agent = Agent(
    role="Software QA Agent",
    goal="Ensure software quality through comprehensive testing and quality assurance processes.",
    backstory="Quality assurance professional specializing in software testing methodologies and automation.",
    verbose=True,
    allow_delegation=False,
    llm=ChatOpenAI(),
)

task = Task(
    description="Design a test plan for a new mobile application, including functional and performance testing strategies.",
    agent=software_qa_agent,
    expected_output="A detailed test plan with test cases and quality assurance procedures."
)

crew = Crew(agents=[software_qa_agent], tasks=[task], verbose=True)
result = crew.kickoff()
print(result)
