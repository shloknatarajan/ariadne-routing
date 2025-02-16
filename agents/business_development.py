from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

business_development_agent = Agent(
    role="Business Development Agent",
    goal="""You are a business development expert who identifies growth opportunities and strategic partnerships.
    
    Analyze markets, identify opportunities, and develop strategic growth plans.""",
    backstory="""You have a proven track record of identifying and securing valuable business opportunities.
    Your strategic thinking has helped numerous companies expand into new markets.""",
    verbose=True,
    allow_delegation=False,
    llm=ChatOpenAI(),
)

task = Task(
    description="Analyze the potential for expanding our software development services into the healthcare sector. Consider market size, competition, and entry barriers.",
    agent=business_development_agent,
    expected_output="A detailed market analysis and expansion strategy recommendation."
)

crew = Crew(agents=[business_development_agent], tasks=[task], verbose=True)

result = crew.kickoff()
print(result)
