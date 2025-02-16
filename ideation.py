"""
This file contains ideation training updates and test prompts.
For each category with more than 5 prompts, we define an agent with 10 training examples.
Additionally, we choose 5 prompts from each agent (here, the first five) as test prompts.
"""

TRAIN_UPDATES = [
    # HR Related Questions (10 examples)
    # ("I have an issue with Workday. My hours weren't updated properly.", "HR Agent"),
    ("What are the company's healthcare benefits?", "HR Agent"),
    ("How do I request time off?", "HR Agent"),
    ("Can you help me find my last three pay stubs?", "HR Agent"),
    ("What is the process for filing a workplace complaint?", "HR Agent"),
    ("How do I update my direct deposit information?", "HR Agent"),
    ("What training programs or career development resources are available?", "HR Agent"),
    ("Who do I contact for questions about my 401(k)?", "HR Agent"),
    ("Can you generate an employment verification letter for me?", "HR Agent"),
    ("What is the company's remote work policy?", "HR Agent"),

    # Code Generation (10 examples)
    # ("Generate a Python script to fetch data from a REST API and store it in a database.", "Code Generation Agent"),
    ("Write a function in JavaScript to validate email addresses using regex.", "Code Generation Agent"),
    ("Create a SQL query to retrieve the top 10 highest-paying customers from our database.", "Code Generation Agent"),
    ("Generate a Python script to automate file renaming in a directory based on a pattern.", "Code Generation Agent"),
    ("Write a Dockerfile for a Node.js application with Express and PostgreSQL.", "Code Generation Agent"),
    ("Generate a React component for a dynamic search bar with autocomplete.", "Code Generation Agent"),
    ("Create a Terraform script to provision an AWS EC2 instance and configure security groups.", "Code Generation Agent"),
    ("Write a unit test suite in Jest for a function that processes user authentication.", "Code Generation Agent"),
    ("Generate a Kubernetes deployment YAML file for a Flask web application.", "Code Generation Agent"),
    ("Write a Python script to scrape product prices from an e-commerce website and store them in a CSV file.", "Code Generation Agent"),

    # Search the web (perplexity) (10 examples)
    # ("Find the latest security vulnerabilities in Python libraries we use.", "Web Search Agent"),
    ("Get the most recent updates on AWS pricing changes.", "Web Search Agent"),
    ("Search for trending front-end frameworks in 2025 and compare their adoption rates.", "Web Search Agent"),
    ("Find documentation for the latest version of Kubernetes.", "Web Search Agent"),
    ("Look up salary benchmarks for software engineers in San Francisco.", "Web Search Agent"),
    ("Search for recent tech layoffs and industry hiring trends.", "Web Search Agent"),
    ("Find the best practices for optimizing large-scale database queries in PostgreSQL.", "Web Search Agent"),
    ("Check if there are any regulatory changes affecting fintech apps in Europe.", "Web Search Agent"),
    ("Look up competitor product releases and announcements from the past month.", "Web Search Agent"),
    ("Find recent blog posts or research papers on AI-powered code generation.", "Web Search Agent"),

    # Customer Service (10 examples)
    # ("Help me troubleshoot why my SaaS account is locked.", "Customer Service Agent"),
    ("Provide steps to integrate your API with my existing application.", "Customer Service Agent"),
    ("I'm experiencing latency issues with your cloud service—how can I fix this?", "Customer Service Agent"),
    ("Can you help me migrate my data from another platform to your service?", "Customer Service Agent"),
    ("I need to downgrade my subscription—what features will I lose?", "Customer Service Agent"),
    ("My software license key isn't working—how do I generate a new one?", "Customer Service Agent"),
    ("Explain the security measures in place for protecting my account data.", "Customer Service Agent"),
    ("Provide a compatibility check—will your product work with my tech stack?", "Customer Service Agent"),
    ("Send me the latest changelog and release notes for your product.", "Customer Service Agent"),
    ("How do I set up multi-factor authentication for better security?", "Customer Service Agent"),

    # Database Agent (10 examples)
    # ("Retrieve the contact details of all employees in the marketing department.", "Database Agent"),
    ("Find the total number of employees working in the company and provide a breakdown by department.", "Database Agent"),
    ("List all employees who joined the company after January 1, 2023.", "Database Agent"),
    ("Get the email addresses of all team leads in the engineering department.", "Database Agent"),
    ("Show me the employee with the highest salary in the company.", "Database Agent"),
    ("Provide a list of employees along with their job titles and phone numbers.", "Database Agent"),
    ("Fetch all employees who report to [Manager Name].", "Database Agent"),
    ("Generate a list of employees whose work anniversary is this month.", "Database Agent"),
    ("Find employees who have been with the company for more than five years.", "Database Agent"),
    ("Retrieve a list of employees along with their assigned projects.", "Database Agent"),
]

TEST_PROMPTS = [
    # HR Agent test prompts (first 5)
    "I have an issue with Workday. My hours weren't updated properly.",

    # Code Generation Agent test prompts (first 5)
    "Generate a Python script to fetch data from a REST API and store it in a database.",

    # Web Search Agent test prompts (first 5)
    "Find the latest security vulnerabilities in Python libraries we use.",

    # Customer Service Agent test prompts (first 5)
    "Help me troubleshoot why my SaaS account is locked.",

    # Database Agent test prompts (first 5)
    "Retrieve the contact details of all employees in the marketing department.",
] 