from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

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
    description="USE THE GIVEN TOOLS. make a google meet between me and shlok.natarajan@gmail.com for 2/19 at 12pm PST (send it via google calendar). In the invite, include relevant information about the company. We are trying to hire Shlok and want to wow him with our company and its mission.",
    agent=hr_agent,
    expected_output=""
)

my_crew = Crew(agents=[hr_agent], tasks=[task], verbose=True)

result = my_crew.kickoff()
print(result)