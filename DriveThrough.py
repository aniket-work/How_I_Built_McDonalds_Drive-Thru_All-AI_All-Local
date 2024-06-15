import pyaudio
from faster_whisper import WhisperModel
import response as resp
from AniBot import Assistant
from audio_handler import embedd_audio, transcribe_audio
import os
DEFAULT_MODEL_SIZE = "medium"


ai_assistant = Assistant()

def main():
    model_size = DEFAULT_MODEL_SIZE + ".en"
    model = WhisperModel(model_size, device="cuda", compute_type="float16", num_workers=10)

    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    customer_input_transcription = ""

    try:
        while True:
            chunk_file = "temp_audio_chunk.wav"

            # Record audio chunk
            print("Listening...")
            if not embedd_audio(audio, stream):
                # Transcribe audio
                transcription = transcribe_audio(model, chunk_file)
                os.remove(chunk_file)
                print("Client :{}".format(transcription))

                # Add customer input to transcript
                customer_input_transcription += "Client : " + transcription + "\n"

                # Process customer input and get response from AI assistant
                output = ai_assistant.interact_with_llm(transcription)
                if output:
                    output = output.lstrip()
                    resp.play_text_to_speech(output)
                    print("AI Assistant :{}".format(output))

    except KeyboardInterrupt:
        print("\nHave a good one ! Take Care...")

    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ == "__main__":
    main()