from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

def legal_prompt(prompt):
# Load environment variables
    load_dotenv()

    legal_agent = Agent(
        role="Legal Agent",
        goal="Provide legal advice, review contracts, and assist with regulatory compliance.",
        backstory="Skilled in contract law and corporate regulations with experience assisting legal teams in multiple industries.",
        verbose=True,
        allow_delegation=False,
        llm=ChatOpenAI(),
    )

    task = Task(
        description=prompt,
        agent=legal_agent,
        expected_output="A summary of potential legal risks and suggestions for modifications."
    )

    crew = Crew(agents=[legal_agent], tasks=[task], verbose=True)
    result = crew.kickoff()
    return result
