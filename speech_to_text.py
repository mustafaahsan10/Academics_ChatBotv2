import os
import tempfile
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

class SpeechTranscriber:
    def __init__(self,  model_size="whisper-1"):
        """Initialize the speech transcriber with the specified model."""
        # Initialize OpenAI client
        api_key = st.secrets.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables or Streamlit secrets")
            
        self.openai_client = OpenAI(api_key=api_key)
        self.model = model_size
    
    def transcribe_audio(self, audio_file, prompt="", response_format="text"):
        """
        Transcribe an audio file using OpenAI's speech-to-text API.
        
        Args:
            audio_file: Path to the audio file
            prompt: Optional prompt to guide the transcription
            response_format: Format of the response (text or json)
            
        Returns:
            Transcribed text
        """
        if not audio_file or not os.path.exists(audio_file):
            return ""
            
        try:
            print(f"Transcribing with OpenAI's {self.model}...")
            
            with open(audio_file, "rb") as file:
                transcription = self.openai_client.audio.transcriptions.create(
                    model=self.model,
                    file=file,
                    response_format=response_format,
                    prompt=prompt
                )
            
            # Clean up the temporary file
            if os.path.exists(audio_file):
                os.remove(audio_file)
            
            # Return the transcription text
            # If response format is "text", the response is already a string
            if response_format == "text":
                return transcription
            # Otherwise, access the text property
            return transcription.text
            
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            # Clean up the file even if there's an error
            if os.path.exists(audio_file):
                os.remove(audio_file)
            return ""

# For testing
if __name__ == "__main__":
    transcriber = SpeechTranscriber()
    
    # Test with a sample audio file
    test_audio_path = "test_audio.wav"
    if os.path.exists(test_audio_path):
        text = transcriber.transcribe_audio(test_audio_path)
        print(f"Transcription: {text}")
    else:
        print(f"Test audio file '{test_audio_path}' not found.")