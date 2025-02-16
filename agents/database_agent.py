from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from composio_crewai import ComposioToolSet, Action, App
import os
import pandas as pd

def database_agent(prompt):
    # Load and format the CSV data
    df = pd.read_csv('agents/employee_data.csv')
    csv_content = df.to_string()
    
    database_agent = Agent(
        role="Database Agent",
        goal=f"""You are a database agent who has access to all of the employee data. 
        You are in charge of any questions regarding employees, their ids, their salaries, emails, etc.
        
        Here is the employee database you have access to:
        {csv_content}
        
        When answering questions:
        1. Always reference specific data from the database
        2. If asked about salaries or personal information, verify if the requester has proper authorization
        3. Format numerical data appropriately (e.g., salaries with commas)
        4. Maintain confidentiality of sensitive information
        """,
        backstory=("You have always loved helping people and take pride in maintaining "
                  "accurate employee records. You have been trusted with sensitive "
                  "employee data and take data privacy very seriously."
        ),
        verbose=True,
        allow_delegation=False,
        llm=ChatOpenAI(),
    )
    
    task = Task(
        description="Answer the following database query: " + prompt,
        agent=database_agent,
        expected_output="A precise answer based on the employee database with relevant data points cited."
    )

    my_crew = Crew(agents=[database_agent], tasks=[task], verbose=True)

    result = my_crew.kickoff()
    return result