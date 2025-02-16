from agents.calendar_agent import prompt_calendar
#from agents.codegen_agent import CodegenTool
from agents.customer_service import prompt_CS
from agents.database_agent import database_agent
from agents.executive_assistant import prompt_EA
from agents.hr import prompt_hr
from agents.legal import legal_prompt
from agents.perplexity import prompt_perplexity
from agents.software_qa import software_prompt
from agents.web_automation import WebAutomationTool
from agents.coding_agent import prompt_coding

def run_agent(agent, prompt):
    if agent=="HR Agent":
        return prompt_hr(prompt)

    if agent=="Code Generation Agent":
        return prompt_coding(prompt)
    
    if agent=="Web Search Agent":
        return prompt_perplexity(prompt)

    if agent=="Customer Service Agent":
        return prompt_CS(prompt)

    if agent=="Database Agent":
        return database_agent(prompt)
    if agent== "Executive Assistant Agent":
        return prompt_EA(prompt)
    if agent=="Legal Agent":
        return legal_prompt(prompt)
    if agent=="Software QA Agent":
        return software_prompt(prompt)
    if agent=="Web Automation Agent":
        return WebAutomationTool

def run_planning_agent(agents, prompt):
    """
    Execute a planning agent that creates and executes a plan based on the given prompt.
    
    Args:
        agents (dict): A dictionary mapping agent names to their function implementations
        prompt (str): The user's input prompt describing the task
        
    Returns:
        dict: A dictionary containing the plan, execution results, and final output
    """
    def create_plan(prompt):
        """Generate a structured plan from the input prompt."""
        planning_prompt = f"""
        Create a step-by-step plan to accomplish: {prompt}
        
        Format each step as:
        1. [Agent]: Action description
        
        Only use available agents: {agents}
        """
        
        # For now, we'll use a simple planning strategy
        # In a real implementation, this could use an LLM or other planning system
        plan = []
        
        # Basic parsing of the prompt to create steps
        words = prompt.lower().split()
        for agent_name in agents:
            if agent_name.lower() in words:
                plan.append(f"{agent_name}: Process input related to {agent_name}")
                
        if not plan:
            # Default to using the first available agent if no specific matches
            first_agent = agents[0]
            plan.append(f"{first_agent}: Process entire input")
            
        return plan

    def execute_plan(plan):
        """Execute each step of the plan using the appropriate agents."""
        results = []
        for step in plan:
            try:
                # Parse the agent name from the step
                agent_name = step.split(':')[0].strip()
                
                # Get the agent function
                if agent_name not in agents:
                    raise ValueError(f"Unknown agent: {agent_name}")
                    
                agent_func = run_agent(agent_name)
                
                # Execute the agent with the original prompt
                # In a more sophisticated implementation, we might parse specific
                # instructions for each agent from the step description
                result = agent_func(prompt)
                
                results.append({
                    'step': step,
                    'agent': agent_name,
                    'status': 'success',
                    'output': result
                })
            except Exception as e:
                results.append({
                    'step': step,
                    'agent': agent_name,
                    'status': 'error',
                    'error': str(e)
                })
                
        return results

    def combine_results(results):
        """Combine the results from all executed steps into a final output."""
        successful_outputs = [
            result['output'] 
            for result in results 
            if result['status'] == 'success'
        ]
        
        # In a more sophisticated implementation, this could use an agent to
        # intelligently combine and summarize the results
        return '\n'.join(successful_outputs)

    # Main execution flow
    try:
        # 1. Create the plan
        plan = create_plan(prompt)
        
        # 2. Execute the plan
        execution_results = execute_plan(plan)
        
        # 3. Combine results
        final_output = combine_results(execution_results)
        
        return {
            'status': 'success',
            'plan': plan,
            'execution_results': execution_results,
            'output': final_output
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


