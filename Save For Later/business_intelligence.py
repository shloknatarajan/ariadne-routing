from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

business_intelligence_agent = Agent(
    role="Business Intelligence Agent",
    goal="Analyze enterprise data to extract actionable insights and support decision making.",
    backstory="Specialized in data aggregation and visualization, providing business reports and dashboards.",
    verbose=True,
    allow_delegation=False,
    llm=ChatOpenAI(),
)

task = Task(
    description="Generate a business intelligence report summarizing key performance indicators and trends for the past fiscal quarter.",
    agent=business_intelligence_agent,
    expected_output="A comprehensive report detailing trends, KPIs, and actionable business insights."
)

crew = Crew(agents=[business_intelligence_agent], tasks=[task], verbose=True)
result = crew.kickoff()
print(result)
