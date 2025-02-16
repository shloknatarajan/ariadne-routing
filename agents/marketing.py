from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

marketing_agent = Agent(
    role="Marketing Agent",
    goal="""You are a marketing strategist who develops effective marketing campaigns and content strategies.
    
    Create compelling marketing materials and strategies that align with business objectives.""",
    backstory="""You've successfully led marketing campaigns for various industries.
    You excel at understanding target audiences and crafting messages that resonate.""",
    verbose=True,
    allow_delegation=False,
    llm=ChatOpenAI(),
)

task = Task(
    description="Create a marketing campaign outline for a new eco-friendly water bottle. Target audience: health-conscious millennials. Budget: $50,000",
    agent=marketing_agent,
    expected_output="A comprehensive marketing campaign strategy including channels, messaging, and budget allocation."
)

crew = Crew(agents=[marketing_agent], tasks=[task], verbose=True)

result = crew.kickoff()
print(result)
