from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from elevenlabs import ElevenLabs
import os

# Load environment variables
load_dotenv()

class TextToSpeechTool:
    def __init__(self):
        self.api_key = os.getenv('ELEVENLABS_API_KEY')
        self.name = "convert_text_to_speech"
        self.description = "Converts text to speech using ElevenLabs API"
        self.client = ElevenLabs(api_key=self.api_key)
        
    def func(self, text, voice_id="JBFqnCBsd6RMkjVDRZzb"):  # Default voice
        """Convert text to speech using ElevenLabs API
        Args:
            text (str): The text to convert to speech
            voice_id (str): The voice ID to use
        """
        try:
            audio = self.client.text_to_speech.convert(
                voice_id=voice_id,
                output_format="mp3_44100_128",
                text=text,
                model_id="eleven_multilingual_v2"
            )
            
            # Convert generator to bytes
            audio_bytes = b"".join(audio)
            
            # Save the audio output
            with open("output_speech.mp3", "wb") as audio_file:
                audio_file.write(audio_bytes)
            return "Audio file successfully saved as output_speech.mp3"
        except Exception as e:
            return f"Error converting text to speech: {str(e)}"

tts_tool = TextToSpeechTool()

text_to_speech_agent = Agent(
    role="Text-to-Speech Agent",
    goal="Convert text content into high-quality speech using ElevenLabs API.",
    backstory="Speech synthesis specialist with expertise in natural language processing and voice generation.",
    verbose=True,
    allow_delegation=False,
    llm=ChatOpenAI(),
    tools=[tts_tool]
)

task = Task(
    description="""Convert the following text to speech using voice_id 'JBFqnCBsd6RMkjVDRZzb': 'Welcome to our company! We're excited to have you join us.'""",
    agent=text_to_speech_agent,
    expected_output="Confirmation of audio file creation with the synthesized speech."
)

crew = Crew(agents=[text_to_speech_agent], tasks=[task], verbose=True)
result = crew.kickoff()
print(result)
