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
