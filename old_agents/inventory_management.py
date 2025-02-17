from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

inventory_management_agent = Agent(
    role="Inventory Management Agent",
    goal="Monitor stock levels, manage supply orders, and coordinate warehouse operations.",
    backstory="Proficient in inventory control systems and logistics, ensuring a balanced supply-demand ratio.",
    verbose=True,
    allow_delegation=False,
    llm=ChatOpenAI(),
)

task = Task(
    description="Generate a plan to optimize inventory levels for a retail company facing overstock issues on seasonal items.",
    agent=inventory_management_agent,
    expected_output="An action plan including recommended order adjustments and inventory reduction strategies."
)

crew = Crew(agents=[inventory_management_agent], tasks=[task], verbose=True)
result = crew.kickoff()
print(result)
