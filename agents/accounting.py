from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

accounting_agent = Agent(
    role="Accounting Agent",
    goal="""You are an accounting expert who helps with financial analysis, bookkeeping, and accounting tasks.
    
    Process financial data and provide clear, accurate accounting insights.""",
    backstory="""You have decades of experience in corporate accounting and financial analysis. 
    You're known for your attention to detail and ability to explain complex financial concepts clearly.""",
    verbose=True,
    allow_delegation=False,
    llm=ChatOpenAI(),
)

task = Task(
    description="Analyze this expense report and identify any unusual patterns or potential issues: [Monthly expenses: Office supplies $2,500, Travel $8,000, Meals $3,000, Software licenses $15,000, Miscellaneous $4,500]",
    agent=accounting_agent,
    expected_output="A detailed analysis of expense patterns with identified anomalies and recommendations."
)

crew = Crew(agents=[accounting_agent], tasks=[task], verbose=True)

result = crew.kickoff()
print(result)
