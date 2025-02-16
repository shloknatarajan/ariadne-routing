# This file stores the training updates (a list of prompt-agent pairs)
# and the test prompts for the StreamRouter.

TRAIN_UPDATES = [
    # Math Agent (8)
    ("Calculate the derivative of sin(x) * e^x using symbolic differentiation.", "Math Agent"),
    ("Solve the integral of x^2 * ln(x) dx and simplify the result.", "Math Agent"),
    ("Find the eigenvalues of the matrix [[1,2,3],[0,1,4],[5,6,0]].", "Math Agent"),
    ("Determine the limit of (1 + 1/n)^n as n approaches infinity.", "Math Agent"),
    ("Prove the Pythagorean theorem using geometric methods.", "Math Agent"),
    ("Calculate the volume of a torus using triple integrals.", "Math Agent"),
    ("Find the general solution to the differential equation dy/dx = x*y.", "Math Agent"),
    ("Determine the convergence of the infinite series Σ(1/n^2) from n=1 to ∞.", "Math Agent"),
    
    # Coding Agent (8)
    ("Develop a Python script that scrapes a webpage using BeautifulSoup.", "Coding Agent"),
    ("Write a JavaScript function to debounce user input effectively.", "Coding Agent"),
    ("Refactor legacy PHP code into a modern MVC framework.", "Coding Agent"),
    ("Implement a RESTful API in Node.js with comprehensive error handling.", "Coding Agent"),
    ("Create a Python decorator for caching function results.", "Coding Agent"),
    ("Implement a binary search tree with balancing in Java.", "Coding Agent"),
    ("Design a scalable microservices architecture using Docker.", "Coding Agent"),
    ("Write a React component for handling form validation.", "Coding Agent"),
    
    # HR Agent (8)
    ("Draft an email scheduling a job interview with a promising candidate.", "HR Agent"),
    ("Outline best practices for remote employee engagement and retention.", "HR Agent"),
    ("Describe effective conflict resolution strategies in the workplace.", "HR Agent"),
    ("Propose a comprehensive benefits package for mid-level tech staff.", "HR Agent"),
    ("Create an onboarding plan for new remote employees.", "HR Agent"),
    ("Design a performance review template for software engineers.", "HR Agent"),
    ("Develop guidelines for promoting diversity and inclusion in hiring.", "HR Agent"),
    ("Write a policy for handling workplace harassment complaints.", "HR Agent"),
    
    # Deep Research (8)
    ("Summarize the latest advancements in quantum computing and their implications.", "Deep Research"),
    ("Analyze the impact of climate change on global economic trends with recent studies.", "Deep Research"),
    ("Critically review recent literature on artificial general intelligence.", "Deep Research"),
    ("Examine the sociopolitical effects of cyber warfare in modern nation-states.", "Deep Research"),
    ("Review emerging technologies in renewable energy storage systems.", "Deep Research"),
    ("Analyze recent developments in CRISPR gene editing technology.", "Deep Research"),
    ("Investigate the impact of social media on democratic processes.", "Deep Research"),
    ("Study the effects of microplastics on marine ecosystems.", "Deep Research"),
    
    # Image Gen (8)
    ("Generate an image prompt for a surreal landscape with floating islands.", "Image Gen"),
    ("Describe a futuristic cityscape at sunset with neon lights and flying vehicles.", "Image Gen"),
    ("Create a prompt for an artistic cyberpunk-themed portrait.", "Image Gen"),
    ("Outline a scene featuring an enchanted forest with glowing flora and mythical creatures.", "Image Gen"),
    ("Design a prompt for a steampunk-inspired mechanical dragon.", "Image Gen"),
    ("Create a detailed description for an underwater city scene.", "Image Gen"),
    ("Generate a prompt for a post-apocalyptic urban landscape.", "Image Gen"),
    ("Describe an alien marketplace with exotic creatures and architecture.", "Image Gen"),
    
    # General Chat (8)
    ("What's the weather forecast for this weekend in New York City?", "General"),
    ("Can you recommend some must-visit attractions in Tokyo?", "General"),
    ("How do I write a professional email to reschedule a meeting?", "General"),
    ("What are some good indoor activities for a rainy day?", "General"),
    ("Could you suggest some popular restaurants in San Francisco?", "General"),
    ("What's the best time of year to visit Paris?", "General"),
    ("How should I format a formal business email signature?", "General"),
    ("What are some fun weekend activities to do with family?", "General"),
    
    # GPT o3-mini-high (6)
    ("Take your time to develop a comprehensive strategy for reducing carbon emissions in urban areas.", "GPT o3-mini-high"),
    ("Break down the complex challenge of improving public education outcomes into actionable steps.", "GPT o3-mini-high"),
    ("Think carefully about all aspects of implementing a city-wide composting program and outline a strategy.", "GPT o3-mini-high"),
    ("Develop a methodical approach to analyze and improve supply chain resilience.", "GPT o3-mini-high"),
    ("Take time to consider and break down the challenge of reducing hospital wait times.", "GPT o3-mini-high"),
    ("Think deeply about strategies to increase voter turnout and civic engagement.", "GPT o3-mini-high"),
    ("Carefully analyze and break down approaches to improve mental health services accessibility.", "GPT o3-mini-high"),
    ("Develop a systematic strategy for transitioning a large organization to renewable energy.", "GPT o3-mini-high"),
    
    # Chemistry RAG (8)
    ("Explain the reaction mechanism of esterification between acetic acid and ethanol.", "Chemistry RAG"),
    ("Describe the periodic trends observed among the halogen elements.", "Chemistry RAG"),
    ("Discuss the thermodynamic principles underlying chemical equilibria.", "Chemistry RAG"),
    ("Analyze the molecular structure of caffeine and its pharmacological effects.", "Chemistry RAG"),
    ("Explain the concept of chirality in organic molecules.", "Chemistry RAG"),
    ("Describe the mechanism of photosynthesis in detail.", "Chemistry RAG"),
    ("Discuss the principles of nuclear magnetic resonance spectroscopy.", "Chemistry RAG"),
    ("Analyze the chemistry behind lithium-ion battery technology.", "Chemistry RAG"),
]

TEST_PROMPTS = [
    "How would you rigorously prove the Fundamental Theorem of Calculus?",
    "Develop a concise Python script to extract data from HTML tables.",
    "What strategies can improve remote team productivity and employee morale?",
    "Provide an overview of the latest breakthroughs in AI-enabled drug discovery.",
    "Generate a detailed image prompt for a fantasy castle amid a stormy landscape."
] 