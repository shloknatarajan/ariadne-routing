from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

project_management_agent = Agent(
    role="Project Management Agent",
    goal="Manage projects by coordinating tasks, tracking deadlines, and ensuring timely completion.",
    backstory="An expert in agile methodologies and risk management with a successful track record of delivering projects on time.",
    verbose=True,
    allow_delegation=False,
    llm=ChatOpenAI(),
)

task = Task(
    description="Develop a project management outline for implementing a new CRM system, detailing key milestones and deliverables.",
    agent=project_management_agent,
    expected_output="A comprehensive project management outline with milestones, deadlines, and deliverables."
)

crew = Crew(agents=[project_management_agent], tasks=[task], verbose=True)
result = crew.kickoff()
print(result)
