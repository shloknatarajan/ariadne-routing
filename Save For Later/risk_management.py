from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

risk_management_agent = Agent(
    role="Risk Management Agent",
    goal="Identify, assess, and mitigate potential risks across business operations.",
    backstory="Expert in risk assessment and mitigation strategies, with experience in developing comprehensive risk management frameworks.",
    verbose=True,
    allow_delegation=False,
    llm=ChatOpenAI(),
)

task = Task(
    description="Conduct a risk assessment for a new international expansion project and provide mitigation strategies.",
    agent=risk_management_agent,
    expected_output="A detailed risk assessment report with identified risks and mitigation plans."
)

crew = Crew(agents=[risk_management_agent], tasks=[task], verbose=True)
result = crew.kickoff()
print(result)
