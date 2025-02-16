from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

def prompt_hr(prompt):
    hr_agent = Agent(
        role="Human Resources Agent",
        goal="""You are a human resources agent whose goal is to handle disputes efficiently and correctly.

        Whenever given a scenario, handle and solve it properly like a HR agent would.
        
        """,
        backstory=("You have always loved helping people."
        ),
        verbose=True,
        allow_delegation=False,
        llm=ChatOpenAI(),
    )
    task = Task(
        description="Solve the given issue using your HR skills: " + prompt,
        agent=hr_agent,
        expected_output=""
    )

    my_crew = Crew(agents=[hr_agent], tasks=[task], verbose=True)

    result = my_crew.kickoff()
    return result