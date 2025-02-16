from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

procurement_agent = Agent(
    role="Procurement Agent",
    goal="Manage purchasing processes, vendor negotiations, and contract management for procuring goods and services.",
    backstory="Expert in vendor management and cost optimization strategies with a track record of securing favorable terms.",
    verbose=True,
    allow_delegation=False,
    llm=ChatOpenAI(),
)

task = Task(
    description="Evaluate a proposal from a new supplier for office equipment and determine if the pricing and terms are competitive compared to market rates.",
    agent=procurement_agent,
    expected_output="A review report with recommendations to accept, negotiate, or reject the supplier proposal."
)

crew = Crew(agents=[procurement_agent], tasks=[task], verbose=True)
result = crew.kickoff()
print(result)
