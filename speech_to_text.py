import os
import wave
import pyaudio
import numpy as np
import time
from faster_whisper import WhisperModel

# Disable symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

class SpeechTranscriber:
    def __init__(self, model_size="medium.en"):
        """Initialize the speech transcriber with the specified model."""
        # Load the Whisper model
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        
    def record_audio(self, duration=7, temp_file="temp_audio.wav"):
        """
        Record audio from the microphone for the specified duration.
        
        Args:
            duration: Number of seconds to record (default: 7)
            temp_file: Path to save the temporary audio file
            
        Returns:
            Path to the recorded audio file
        """
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, 
                        channels=1, 
                        rate=16000, 
                        input=True, 
                        frames_per_buffer=1024)
        
        print(f"Recording for {duration} seconds...")
        
        frames = []
        # Calculate total chunks to record based on frame rate and buffer size
        chunk_size = 1024
        sample_rate = 16000
        total_chunks = int((sample_rate / chunk_size) * duration)
        
        # Record for the specified duration
        for _ in range(total_chunks):
            data = stream.read(chunk_size)
            frames.append(data)
        
        print("Recording finished.")
        
        # Close and clean up resources
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Save the recorded audio
        wf = wave.open(temp_file, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        return temp_file
    
    def transcribe_audio(self, audio_file):
        """
        Transcribe an audio file using the Whisper model.
        
        Args:
            audio_file: Path to the audio file
            
        Returns:
            Transcribed text
        """
        print("Transcribing...")
        segments, _ = self.model.transcribe(audio_file, beam_size=5)
        
        text = ""
        for segment in segments:
            text += segment.text
        
        # Clean up the temporary file
        if os.path.exists(audio_file):
            os.remove(audio_file)
            
        return text
    
    def listen_and_transcribe(self, duration=7):
        """
        Record audio and transcribe it in one step.
        
        Args:
            duration: Number of seconds to record
            
        Returns:
            Transcribed text
        """
        temp_file = "temp_audio.wav"
        self.record_audio(duration, temp_file)
        return self.transcribe_audio(temp_file)


# For testing
if __name__ == "__main__":
    transcriber = SpeechTranscriber()
    text = transcriber.listen_and_transcribe(7)
    print(f"Transcription: {text}") 