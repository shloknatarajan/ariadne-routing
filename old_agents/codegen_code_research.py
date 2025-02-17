from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import git
import os
import ast
from rich.console import Console
from pathlib import Path
import tempfile
import shutil
import time
from codegen import Codebase
from codegen.extensions.langchain.agent import create_agent_with_tools
from codegen.extensions.langchain.tools import (
    ListDirectoryTool,
    RevealSymbolTool,
    SearchTool,
    SemanticSearchTool,
    ViewFileTool,
)
from langchain_core.messages import SystemMessage
from rich.markdown import Markdown
from rich.prompt import Prompt

# Load environment variables
load_dotenv()

# Initialize console
console = Console()

class CodeResearchTool:
    def __init__(self):
        self.name = "analyze_repository"
        self.description = "Analyzes GitHub repositories for deep code research. Input should be a simple text description of the task."
        
    def func(self, repo_name: str):
        """Analyze a GitHub repository
        Args:
            repo_name (str): GitHub repository in format 'owner/repo'
        """
        try:
            # Create a temporary directory for cloning
            temp_dir = tempfile.mkdtemp()
            repo_url = f"https://github.com/{repo_name}.git"
            
            # Clone the repository
            with console.status(f"[bold blue]Cloning {repo_name}...[/bold blue]") as status:
                repo = git.Repo.clone_from(repo_url, temp_dir, depth=1)  # Use shallow clone for speed
                status.update("[bold green]✓ Repository cloned successfully![/bold green]")

            with console.status("[bold blue]Analyzing repository...[/bold blue]") as status:
                # Initialize codebase
                codebase = Codebase(temp_dir)
                
                # Create research tools
                tools = [
                    ViewFileTool(codebase),
                    ListDirectoryTool(codebase),
                    SearchTool(codebase),
                    SemanticSearchTool(codebase),
                    RevealSymbolTool(codebase),
                ]

                # Initialize agent with research tools
                agent = create_agent_with_tools(
                    codebase=codebase,
                    tools=tools,
                    chat_history=[SystemMessage(content=RESEARCH_AGENT_PROMPT)],
                    verbose=True
                )

                # Run analysis
                result = agent.invoke(
                    {"input": "Analyze the main components and their relationships in this codebase"},
                    config={"configurable": {"session_id": "research"}}
                )
                
                status.update("[bold green]✓ Analysis complete![/bold green]")
            
            # Clean up
            shutil.rmtree(temp_dir)
            
            return result["output"]
            
        except Exception as e:
            return f"Error analyzing repository: {str(e)}"

# Research agent prompt
RESEARCH_AGENT_PROMPT = """You are a code research expert. Your goal is to help users understand codebases by:
1. Finding relevant code through semantic and text search
2. Analyzing symbol relationships and dependencies
3. Exploring directory structures
4. Reading and explaining code

Always explain your findings in detail and provide context about how different parts of the code relate to each other.
When analyzing code, consider:
- The purpose and functionality of each component
- How different parts interact
- Key patterns and design decisions
- Potential areas for improvement

Break down complex concepts into understandable pieces and use examples when helpful."""

code_research_tool = CodeResearchTool()

code_research_agent = Agent(
    role="Code Research Agent",
    goal="Analyze GitHub repositories to provide deep insights into codebases.",
    backstory="Expert code analyst with deep understanding of software architecture and design patterns.",
    verbose=True,
    allow_delegation=False,
    llm=ChatOpenAI(),
    tools=[code_research_tool]
)

task = Task(
    description="""Analyze the kennethreitz/requests-html repository (a small Python package) and explain its core architecture and main components.""",
    agent=code_research_agent,
    expected_output="A detailed analysis of the requests-html codebase architecture and components."
)

crew = Crew(agents=[code_research_agent], tasks=[task], verbose=True)
result = crew.kickoff()
print(result)
