from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from codegen import Codebase
from codegen.extensions.langchain.agent import create_codebase_agent
from codegen.extensions.langchain.tools import (
    ViewFileTool,
    ListDirectoryTool, 
    SearchTool,
    EditFileTool,
    CreateFileTool,
    DeleteFileTool,
    RenameFileTool,
    MoveSymbolTool,
    RevealSymbolTool,
    SemanticEditTool,
    CommitTool
)
import os

# Load environment variables
load_dotenv()

class CodegenTool:
    def __init__(self):
        self.name = "execute_codegen"
        self.description = "Executes code transformations using Codegen"
        
    def func(self, codebase_path: str, query: str):
        """Execute code transformations using Codegen
        Args:
            codebase_path (str): Path to the codebase to transform
            query (str): Natural language query describing the transformation
        """
        try:
            # Initialize codebase
            codebase = Codebase(codebase_path)
            
            # Create the agent with tools
            tools = [
                ViewFileTool(codebase),      # View file contents
                ListDirectoryTool(codebase),  # List directory contents
                SearchTool(codebase),        # Search code
                EditFileTool(codebase),      # Edit files
                CreateFileTool(codebase),    # Create new files
                DeleteFileTool(codebase),    # Delete files
                RenameFileTool(codebase),    # Rename files
                MoveSymbolTool(codebase),    # Move functions/classes
                RevealSymbolTool(codebase),  # Analyze symbol relationships
                SemanticEditTool(codebase),  # Make semantic edits
                CommitTool(codebase),        # Commit changes
            ]

            # Create the agent with GPT-4
            agent = create_codebase_agent(
                codebase=codebase,
                tools=tools,
                model_name="gpt-4",
                temperature=0,
                verbose=True
            )

            # Execute the query
            result = agent.invoke(
                {"input": query},
                config={"configurable": {"session_id": "demo"}}
            )
            
            return result["output"]
            
        except Exception as e:
            return f"Error executing code transformation: {str(e)}"

codegen_tool = CodegenTool()

codegen_agent = Agent(
    role="Codegen Agent",
    goal="Execute code transformations and analysis using Codegen's powerful APIs",
    backstory="""Expert in code manipulation and analysis using Codegen. 
    Capable of performing complex code transformations while maintaining correctness 
    across Python and TypeScript/JSX codebases. Can analyze dependencies, move code,
    and make semantic edits.""",
    verbose=True,
    allow_delegation=False,
    llm=ChatOpenAI(),
    tools=[codegen_tool]
)

# Example task
task = Task(
    description="""
    Analyze the dependencies of the FastAPI class in the current codebase.
    """,
    agent=codegen_agent,
    expected_output="Analysis of FastAPI class dependencies"
)

if __name__ == "__main__":
    crew = Crew(agents=[codegen_agent], tasks=[task], verbose=True)
    result = crew.kickoff()
    print(result)
