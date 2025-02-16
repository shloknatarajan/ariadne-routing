from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

def prompt_coding(prompt):
    # Load environment variables
    load_dotenv()

    coding_agent = Agent(
        role="Coding Agent",
        goal="Solve coding, math, or software engineering problems",
        backstory="Skilled in software engineering and computer science with experience across many codebases.",
        verbose=True,
        allow_delegation=False,
        llm=ChatOpenAI(),
    )

    task = Task(
        description=prompt,
        agent=coding_agent,
        expected_output="A solution to a computer science or engineering problem"
    )

    crew = Crew(agents=[coding_agent], tasks=[task], verbose=True)
    result = crew.kickoff()
    return result
