from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

def prompt_CS(prompt):
    hr_agent = Agent(
        role="Customer Service Agent",
        goal="""You are a customer service agent whose goal is to handle disputes efficiently and correctly. Answer any questions properly and thoroughly.

        Whenever given a scenario, handle and solve it properly like a customer service agent would. If you don't know the answer, make it up and make it seem believable like how a customer service agent would answer.
        
        """,
        backstory=("You have always loved helping people."
        ),
        verbose=True,
        allow_delegation=False,
        llm=ChatOpenAI(),
    )
    task = Task(
        description="Solve the given issue using your customer service skills: " + prompt,
        agent=hr_agent,
        expected_output=""
    )

    my_crew = Crew(agents=[hr_agent], tasks=[task], verbose=True)

    result = my_crew.kickoff()
    return result