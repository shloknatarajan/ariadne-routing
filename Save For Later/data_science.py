from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

data_science_agent = Agent(
    role="Data Science Agent",
    goal="Analyze large datasets, build predictive models, and apply machine learning techniques for enterprise solutions.",
    backstory="Experienced in data mining, statistical analysis, and machine learning, delivering actionable predictions and insights.",
    verbose=True,
    allow_delegation=False,
    llm=ChatOpenAI(),
)

task = Task(
    description="Develop a predictive model outline for forecasting quarterly sales based on historical data trends and market indicators.",
    agent=data_science_agent,
    expected_output="A detailed plan outlining the steps and methodologies to build a predictive sales forecasting model."
)

crew = Crew(agents=[data_science_agent], tasks=[task], verbose=True)
result = crew.kickoff()
print(result)
