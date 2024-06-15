import pyaudio
from faster_whisper import WhisperModel
import response as resp
from AniBot import Assistant
from audio_handler import embedd_audio, transcribe_audio
import os

# Default model size for the WhisperModel
DEFAULT_MODEL_SIZE = "medium"

# Initialize the AI assistant
ai_assistant = Assistant()


def main():
    """
    Main function to continuously listen to audio input, transcribe it, and interact with an AI assistant.

    This function sets up the audio stream, listens for audio input, records chunks, transcribes them using
    a Whisper model, and then processes the transcription with an AI assistant to generate responses.
    The AI's responses are played using a text-to-speech system.

    The loop continues until interrupted by the user (e.g., by pressing Ctrl+C).
    """
    model_size = DEFAULT_MODEL_SIZE + ".en"

    # Initialize the Whisper model with specified parameters
    model = WhisperModel(model_size, device="cuda", compute_type="float16", num_workers=10)

    # Set up the PyAudio stream for recording audio
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

    # Initialize a variable to accumulate customer input transcriptions
    customer_input_transcription = ""

    try:
        while True:
            chunk_file = "inflight_audio.wav"

            # Record audio chunk
            print("Listening...")
            if not embedd_audio(audio, stream):
                # Transcribe the recorded audio chunk
                transcription = transcribe_audio(model, chunk_file)
                os.remove(chunk_file)
                print("Client :{}".format(transcription))

                # Add customer input to the accumulated transcriptions
                customer_input_transcription += "Client : " + transcription + "\n"

                # Process customer input and get a response from the AI assistant
                output = ai_assistant.interact_with_llm(transcription)
                if output:
                    output = output.lstrip()
                    resp.play_text_to_speech(output)
                    print("AI Assistant :{}".format(output))

    except KeyboardInterrupt:
        print("\nHave a good one ! Take Care...")

    finally:
        # Ensure the audio stream is properly closed upon exit
        stream.stop_stream()
        stream.close()
        audio.terminate()


if __name__ == "__main__":
    main()
