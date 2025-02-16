from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

network_monitoring_agent = Agent(
    role="Network Monitoring Agent",
    goal="Monitor network performance, identify bottlenecks, and maintain system uptime.",
    backstory="Network specialist with expertise in performance optimization and system monitoring.",
    verbose=True,
    allow_delegation=False,
    llm=ChatOpenAI(),
)

task = Task(
    description="Analyze network performance metrics and provide recommendations for improving system reliability.",
    agent=network_monitoring_agent,
    expected_output="A network performance analysis report with optimization recommendations."
)

crew = Crew(agents=[network_monitoring_agent], tasks=[task], verbose=True)
result = crew.kickoff()
print(result)
