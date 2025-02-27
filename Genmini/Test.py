import os
from google.cloud import speech_v1 as speech
from langchain.tools import BaseTool

def transcribe_audio_mp3(audio_path: str, service_account_path: str) -> str:
    """
    Transcribe an MP3 audio file using Google Speech-to-Text API.
    """
    client = speech.SpeechClient.from_service_account_file(service_account_path)

    with open(audio_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP3,
        sample_rate_hertz=44100,
        language_code="en-US"
    )

    response = client.long_running_recognize(config=config, audio=audio)
    return " ".join(
        result.alternatives[0].transcript 
        for result in response.results
    )

def save_transcript_to_file(transcript: str, output_path: str) -> None:
    """
    Save the transcribed text to a file.
    """
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(transcript)
    print(f"Transcription saved to: {output_path}")

class AudioToTextMP3Tool(BaseTool):
    """
    A LangChain tool that converts MP3 audio to text using 
    the Google Speech-to-Text API and saves the result to a file.
    """

    name = "GoogleSpeechToTextMP3"
    description = (
        "Convert an MP3 audio file to text using Google Speech-to-Text "
        "and save it to a specified text file."
    )

    def _run(
        self, 
        audio_path: str = None,
        api_key: str = None,
        output_path: str = None,
        **kwargs
    ) -> str:
        """
        Synchronously run the tool.

        :param audio_path: Path to the MP3 file.
        :param api_key: Path to service account JSON (Google API key).
        :param output_path: Path to the output text file.
        """
        # Basic validation
        if not audio_path or not os.path.exists(audio_path):
            return f"❌ Audio file not found: {audio_path}"
        if not api_key or not os.path.exists(api_key):
            return f"❌ Service account file not found: {api_key}"
        if not output_path:
            return "❌ Output path not specified."

        # Perform transcription
        transcript = transcribe_audio_mp3(audio_path, api_key)
        save_transcript_to_file(transcript, output_path)
        return f"✅ Transcription completed: {output_path}"

    async def _arun(self, *args, **kwargs) -> str:
        """
        Asynchronously run the tool. Not yet implemented.
        """
        raise NotImplementedError("Async method not implemented for this tool.")

# Example usage
if __name__ == "__main__":
    api_key = "./data/svcAcct.json"
    audio_path = "./data/A0106.mp3"
    output_path = "transcription_output.txt"

    tool = AudioToTextMP3Tool()

    # Create the dictionary of inputs
    tool_input = {
        "audio_path": audio_path,
        "api_key": api_key,
        "output_path": output_path
    }

    # Run the tool, which will pass each dict key-value pair as a keyword argument
    result = tool.run(tool_input)
    print(result)
