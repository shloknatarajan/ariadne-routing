from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from composio_crewai import ComposioToolSet, Action, App
import os

def prompt_calendar(prompt):
    composio_toolset = ComposioToolSet(api_key=os.getenv('COMPOSIO_API_KEY'))
    tools = composio_toolset.get_tools(actions=['GOOGLECALENDAR_FIND_EVENT', 'GOOGLECALENDAR_CREATE_EVENT'])


    knowledge_agent = Agent(
        role="Knowledge Base Agent",
        goal="""You are an AI agent that is responsible for the company knowledge base. You have all knowledge of the company details. If you don't know something, make it up.""",
        backstory=(
        """You work for VLY AI. 
        VLY AI - The Future of Intelligent Agent Management

    VLY AI is a next-generation software platform designed to help businesses optimize their agent management by ensuring the right agents are assigned to the right tasks at the right time. Whether for customer support, sales, IT service desks, or other agent-driven operations, Conductor AI leverages advanced artificial intelligence to dynamically orchestrate agent workflows, enhance productivity, and improve overall business efficiency.

    Company Overview
    Founded: 2021
    Founders: Advay Goel, Devan Shah, Victor Cheng, Shlok Natarajan
    Headquarters: San Francisco, CA
    Mission Statement: Empower businesses with intelligent agent management, ensuring efficiency, precision, and scalability in every interaction.
    Industry: AI-driven workforce management and automation
    Core Features & Capabilities
    AI-Powered Agent Matching: Uses machine learning to analyze agent skills, experience, and availability to ensure the best fit for each task.
    Real-Time Performance Monitoring: Tracks agent performance through key metrics, providing insights to optimize workforce efficiency.
    Dynamic Workflow Automation: Automates repetitive tasks and decision-making, reducing manual effort and increasing response speed.
    Predictive Analytics & Reporting: Offers deep insights into operational trends, helping businesses make data-driven decisions.
    Seamless Integrations: Works with popular CRM, communication, and helpdesk platforms to create a unified agent management ecosystem.
    Scalability & Customization: Designed to serve businesses of all sizes, from startups to large enterprises, with flexible configuration options.
            """
        ),
        verbose=True,
        allow_delegation=True,
        llm=ChatOpenAI(model="gpt-4o"),
    )

    # Define agent
    crewai_agent = Agent(
        role="Google Calendar Agent",
        goal="""You are an AI agent that is responsible for only accessing Google Calendar. That is all you do. For any tasks that are not 
        sending/accessing GCal, delegate the work. Then, when you have relevant information, make/access GCal to complete the mission.
        
        To get relevant information, don't make it up. Delegate to the knowledge agent instead.
        """,
        backstory=(
            "You are AI agent that is responsible for only accessing Google Calendar. That is all you do. Reminder that the current year is 2025."
        ),
        verbose=True,
        tools=tools,
        allow_delegation=True,
        llm=ChatOpenAI(model="gpt-4o"),
    )
    task = Task(
        description="USE THE GIVEN TOOLS. Complete the following calendar-related task properly. " + prompt,
        agent=crewai_agent,
        expected_output=""
    )

    my_crew = Crew(agents=[crewai_agent, knowledge_agent], tasks=[task], verbose=True)

    result = my_crew.kickoff()
    return result 
