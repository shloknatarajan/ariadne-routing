from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

social_media_agent = Agent(
    role="Social Media Agent",
    goal="Manage social media presence and develop engaging content strategies.",
    backstory="Social media strategist with expertise in content creation and community engagement.",
    verbose=True,
    allow_delegation=False,
    llm=ChatOpenAI(),
)

task = Task(
    description="Create a social media content calendar for product launch, including post ideas and engagement strategies.",
    agent=social_media_agent,
    expected_output="A detailed social media content calendar with engagement strategies."
)

crew = Crew(agents=[social_media_agent], tasks=[task], verbose=True)
result = crew.kickoff()
print(result)
