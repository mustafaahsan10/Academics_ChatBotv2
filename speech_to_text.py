import os
import wave
import pyaudio
import numpy as np
import time
import threading
from faster_whisper import WhisperModel

# Disable symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

class SpeechTranscriber:
    def __init__(self, model_size="medium.en"):
        """Initialize the speech transcriber with the specified model."""
        # Load the Whisper model
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.is_recording = False
        self.audio_data = []
        self.p = None
        self.stream = None
        
    def start_recording(self):
        """
        Start recording audio from the microphone.
        Returns True if recording started successfully, False otherwise.
        """
        if self.is_recording:
            print("Already recording")
            return False
            
        try:
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(
                format=pyaudio.paInt16, 
                channels=1, 
                rate=16000, 
                input=True, 
                frames_per_buffer=1024
            )
            
            self.audio_data = []
            self.is_recording = True
            
            # Start the recording thread
            self.recording_thread = threading.Thread(target=self._record_thread)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
            print("Recording started...")
            return True
            
        except Exception as e:
            print(f"Error starting recording: {e}")
            self._cleanup_resources()
            return False
    
    def _record_thread(self):
        """Background thread to record audio"""
        try:
            while self.is_recording and self.stream:
                data = self.stream.read(1024, exception_on_overflow=False)
                self.audio_data.append(data)
        except Exception as e:
            print(f"Error in recording thread: {e}")
            self.is_recording = False
        
    def stop_recording(self, temp_file="temp_audio.wav"):
        """
        Stop recording and save the audio to a file.
        
        Args:
            temp_file: Path to save the temporary audio file
            
        Returns:
            Path to the recorded audio file or None if no audio was recorded
        """
        if not self.is_recording:
            print("Not recording")
            return None
            
        try:
            # Stop recording
            self.is_recording = False
            
            # Wait for recording thread to finish
            if hasattr(self, 'recording_thread') and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=1.0)
            
            # Clean up PyAudio resources
            self._cleanup_resources()
            
            print("Recording stopped.")
            
            # Check if we got any audio data
            if not self.audio_data:
                print("No audio data captured")
                return None
                
            # Save the recorded audio
            wf = wave.open(temp_file, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit audio = 2 bytes
            wf.setframerate(16000)
            wf.writeframes(b''.join(self.audio_data))
            wf.close()
            
            return temp_file
            
        except Exception as e:
            print(f"Error stopping recording: {e}")
            self._cleanup_resources()
            return None
    
    def _cleanup_resources(self):
        """Clean up PyAudio resources"""
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
                
            if self.p:
                self.p.terminate()
                self.p = None
        except Exception as e:
            print(f"Error cleaning up resources: {e}")
    
    def transcribe_audio(self, audio_file):
        """
        Transcribe an audio file using the Whisper model.
        
        Args:
            audio_file: Path to the audio file
            
        Returns:
            Transcribed text
        """
        if not audio_file or not os.path.exists(audio_file):
            return ""
            
        print("Transcribing...")
        segments, _ = self.model.transcribe(audio_file, beam_size=5)
        
        text = ""
        for segment in segments:
            text += segment.text
        
        # Clean up the temporary file
        if os.path.exists(audio_file):
            os.remove(audio_file)
            
        return text
    
    def is_currently_recording(self):
        """Check if recording is currently in progress"""
        return self.is_recording

# For testing
if __name__ == "__main__":
    transcriber = SpeechTranscriber()
    print("Press Enter to start recording...")
    input()
    transcriber.start_recording()
    print("Press Enter to stop recording...")
    input()
    audio_file = transcriber.stop_recording()
    if audio_file:
        text = transcriber.transcribe_audio(audio_file)
        print(f"Transcription: {text}")
    else:
        print("No audio recorded.") 