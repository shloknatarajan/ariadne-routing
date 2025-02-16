from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

def prompt_EA(prompt):
# Load environment variables
    load_dotenv()

    executive_assistant_agent = Agent(
        role="Executive Assistant Agent",
        goal="Manage executive schedules, communications, and administrative tasks efficiently.",
        backstory="Experienced executive assistant with strong organizational and communication skills.",
        verbose=True,
        allow_delegation=False,
        llm=ChatOpenAI(),
    )

    task = Task(
        description=prompt,
        agent=executive_assistant_agent,
        expected_output="A complete board meeting plan with agenda and logistics details."
    )

    crew = Crew(agents=[executive_assistant_agent], tasks=[task], verbose=True)
    result = crew.kickoff()
    return result 


