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

    # Executive Assistant (10 examples)
    # ("Schedule a meeting with the engineering and marketing teams next Wednesday at 10 AM and send out invites.", "Executive Assistant Agent"),
    ("Draft and send a follow-up email to [Client Name] thanking them for today's call and summarizing next steps.", "Executive Assistant Agent"),
    ("Reschedule my 3 PM meeting with [Person] to Friday at the same time and update the calendar invite.", "Executive Assistant Agent"),
    ("Check my calendar for conflicts and suggest a better time for my one-on-one with [Employee].", "Executive Assistant Agent"),
    ("Set a reminder for me to review the quarterly financial report before Monday's executive meeting.", "Executive Assistant Agent"),
    ("Book a conference room for the product launch meeting next Tuesday at 2 PM.", "Executive Assistant Agent"),
    ("Summarize my unread emails and highlight anything urgent.", "Executive Assistant Agent"),
    ("Automatically send birthday greetings to employees on their special day.", "Executive Assistant Agent"),
    ("Create a daily agenda summary with my meetings, deadlines, and important tasks.", "Executive Assistant Agent"),
    ("Draft a professional email to [Vendor] requesting an updated contract before the end of the month.", "Executive Assistant Agent"),

    # Legal (10 examples)
    # ("Review the terms of service for our new SaaS product and ensure compliance with GDPR and CCPA.", "Legal Agent"),
    ("Draft a non-disclosure agreement (NDA) for third-party contractors working on proprietary software.", "Legal Agent"),
    ("Analyze a software licensing agreement and highlight any potential legal risks.", "Legal Agent"),
    ("Generate a data privacy policy that aligns with global data protection regulations.", "Legal Agent"),
    ("Review an intellectual property (IP) agreement to ensure our company retains all rights to internally developed software.", "Legal Agent"),
    ("Draft an employee acceptable use policy (AUP) for company-issued laptops and software.", "Legal Agent"),
    ("Check an open-source software license for compatibility with our proprietary codebase.", "Legal Agent"),
    ("Generate a compliance checklist for software accessibility requirements under ADA and WCAG.", "Legal Agent"),
    ("Review a vendor contract to identify any unfavorable clauses before signing.", "Legal Agent"),
    ("Draft a terms of service and privacy policy for a mobile app, ensuring compliance with App Store and Google Play guidelines.", "Legal Agent"),

    # Software QA (10 examples)
    # ("Generate a set of test cases for the login functionality, including valid, invalid, and edge cases.", "Software QA Agent"),
    ("Write automated test scripts for form validation on the user registration page.", "Software QA Agent"),
    ("Create a test plan for the checkout process in an e-commerce application.", "Software QA Agent"),
    ("Generate test cases for API endpoint /getUserProfile, covering all possible response scenarios.", "Software QA Agent"),
    ("Write performance test scenarios to measure load handling for 10,000 concurrent users.", "Software QA Agent"),
    ("Identify possible security vulnerabilities and generate security test cases for user authentication.", "Software QA Agent"),
    ("Develop test cases for a file upload feature, including different file types and sizes.", "Software QA Agent"),
    ("Create test cases to verify responsiveness and cross-browser compatibility for the dashboard UI.", "Software QA Agent"),
    ("Write a set of regression test cases to verify that recent updates did not break existing functionality.", "Software QA Agent"),
    ("Generate a report summarizing test coverage for the latest software release.", "Software QA Agent"),

    # Web Automation (10 examples)
    # ("Check my system's CPU and memory usage and report any high resource consumption.", "Web Automation Agent"),
    ("Find and delete all duplicate files in my Downloads folder.", "Web Automation Agent"),
    ("List all running applications and their resource usage.", "Web Automation Agent"),
    ("Automatically organize all my desktop files into categorized folders.", "Web Automation Agent"),
    ("Scan my computer for large unused files and suggest which ones to delete.", "Web Automation Agent"),
    ("Backup all my important documents and pictures to an external drive.", "Web Automation Agent"),
    ("Find and close all unresponsive applications.", "Web Automation Agent"),
    ("Check my internet speed and troubleshoot any connection issues.", "Web Automation Agent"),
    ("List all installed software and highlight outdated or unnecessary programs.", "Web Automation Agent"),
    ("Generate a report of my most-used applications in the past month.", "Web Automation Agent"),
]

TEST_PROMPTS = [
    # HR Agent test prompt
    "I have an issue with Workday. My hours weren't updated properly.",

    # Code Generation Agent test prompt
    "Generate a Python script to fetch data from a REST API and store it in a database.",

    # Web Search Agent test prompt
    "Find the latest security vulnerabilities in Python libraries we use.",

    # Customer Service Agent test prompt
    "Help me troubleshoot why my SaaS account is locked.",

    # Database Agent test prompt
    "Retrieve the contact details of all employees in the marketing department.",

    # Executive Assistant Agent test prompt
    "Schedule a meeting with the engineering and marketing teams next Wednesday at 10 AM and send out invites.",

    # Legal Agent test prompt
    "Review the terms of service for our new SaaS product and ensure compliance with GDPR and CCPA.",

    # Software QA Agent test prompt
    "Generate a set of test cases for the login functionality, including valid, invalid, and edge cases.",

    # Web Automation Agent test prompt
    "Check my system's CPU and memory usage and report any high resource consumption.",
]