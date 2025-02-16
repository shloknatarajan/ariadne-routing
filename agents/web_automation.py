from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from scrapybara import Scrapybara
from scrapybara.tools import BashTool, ComputerTool, EditTool
from scrapybara.anthropic import Anthropic
from scrapybara.prompts import UBUNTU_SYSTEM_PROMPT
import os

# Load environment variables
load_dotenv()

class WebAutomationTool:
    def __init__(self):
        self.api_key = os.getenv('SCRAPYBARA_API_KEY')
        self.name = "execute_web_task"
        self.description = "Executes web automation tasks using Scrapybara. Input should be a simple text description of the task."
        self.client = Scrapybara(api_key=self.api_key)
        
    def func(self, task_description: str):
        """Execute a web automation task using Scrapybara
        Args:
            task_description (str): Description of the task to execute
        """
        try:
            # Start an Ubuntu instance
            instance = self.client.start_ubuntu(
                timeout_hours=1,
            )
            
            # Execute the task using Act SDK
            response = self.client.act(
                model=Anthropic(),
                tools=[
                    BashTool(instance),
                    ComputerTool(instance),
                    EditTool(instance),
                ],
                system=UBUNTU_SYSTEM_PROMPT,
                prompt=task_description,
                on_step=lambda step: print(step.text),  # Print each step for debugging
            )
            
            # Stop the instance
            instance.stop()
            
            return f"Task executed successfully. Response: {response}"
        except Exception as e:
            return f"Error executing web task: {str(e)}"

web_automation_tool = WebAutomationTool()

web_automation_agent = Agent(
    role="Web Automation Agent",
    goal="Execute web automation tasks using Scrapybara's Act SDK.",
    backstory="Automation specialist capable of performing complex web tasks using Scrapybara's automation tools.",
    verbose=True,
    allow_delegation=False,
    llm=ChatOpenAI(),
    tools=[web_automation_tool]
)

task = Task(
    description="Go to Hacker News and get the title of the top story.",
    agent=web_automation_agent,
    expected_output="The title of the top story from Hacker News."
)

crew = Crew(agents=[web_automation_agent], tasks=[task], verbose=True)
result = crew.kickoff()
print(result)
