from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from lumaai import LumaAI
import requests
import time
import os

# Load environment variables
load_dotenv()

class TextToVideoTool:
    def __init__(self):
        self.api_key = os.getenv('LUMAAI_API_KEY')
        self.name = "convert_text_to_video"
        self.description = "Converts text descriptions into video using LumaAI API"
        self.client = LumaAI(auth_token=self.api_key)
        
    def func(self, text: str, duration: str = "5s", resolution: str = "720p"):
        """Convert text to video using LumaAI API
        Args:
            text (str): The text description to convert to video
            duration (str): Duration of the video (default: "5s")
            resolution (str): Video resolution (default: "720p")
        """
        try:
            # Create the generation
            generation = self.client.generations.create(
                prompt=text,
                model="ray-2",
                resolution=resolution,
                duration=duration
            )
            
            # Poll for completion
            completed = False
            while not completed:
                generation = self.client.generations.get(id=generation.id)
                if generation.state == "completed":
                    completed = True
                elif generation.state == "failed":
                    raise RuntimeError(f"Generation failed: {generation.failure_reason}")
                print("Generating video...")
                time.sleep(3)
            
            # Get video URL
            video_url = generation.assets.video
            
            # Download the video
            response = requests.get(video_url, stream=True)
            file_name = f'{generation.id}.mp4'
            with open(file_name, 'wb') as file:
                file.write(response.content)
            
            return f"Video generated successfully. URL: {video_url}\nFile downloaded as: {file_name}"
        except Exception as e:
            return f"Error generating video: {str(e)}"

tts_tool = TextToVideoTool()

text_to_video_agent = Agent(
    role="Text-to-Video Agent",
    goal="Convert text descriptions into high-quality videos using LumaAI API.",
    backstory="Video generation specialist with expertise in creating engaging visual content from textual descriptions.",
    verbose=True,
    allow_delegation=False,
    llm=ChatOpenAI(),
    tools=[tts_tool]
)

task = Task(
    description="""Convert the following text to video: 'A serene mountain landscape at sunset, with clouds rolling over snow-capped peaks'""",
    agent=text_to_video_agent,
    expected_output="Video file URL and local file path."
)

crew = Crew(agents=[text_to_video_agent], tasks=[task], verbose=True)
result = crew.kickoff()
print(result)
