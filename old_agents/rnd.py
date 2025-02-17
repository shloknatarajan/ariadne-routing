from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

rnd_agent = Agent(
    role="Research and Development Agent",
    goal="Drive innovation through research initiatives and new product development.",
    backstory="Innovation specialist with experience in product research and technology advancement.",
    verbose=True,
    allow_delegation=False,
    llm=ChatOpenAI(),
)

task = Task(
    description="Research emerging technologies in AI and propose potential applications for the company's products.",
    agent=rnd_agent,
    expected_output="A research report on emerging AI technologies with implementation recommendations."
)

crew = Crew(agents=[rnd_agent], tasks=[task], verbose=True)
result = crew.kickoff()
print(result)
