import streamlit as st
import os
import pyaudio
from faster_whisper import WhisperModel
import response as resp
from AniBot import Assistant
from audio_handler import embedd_audio, transcribe_audio

# Default model size for the WhisperModel
DEFAULT_MODEL_SIZE = "medium"

# Initialize the AI assistant
ai_assistant = Assistant()


def load_menu(file_path):
    """
    Load and return the menu from the specified file path.

    Args:
        file_path (str): The path to the menu file.

    Returns:
        list: The menu content as a list of lines.
    """
    with open(file_path, 'r') as file:
        return file.readlines()


def main():
    """
    Main function to set up the Streamlit UI for the McDonald's drive-thru simulation.
    """
    st.title("Welcome to AniDonald's Drive-Thru 🍔🍟")

    # Display the menu in two columns with McDonald's theme
    menu_file_path = "menu/menu.txt"
    menu_content = load_menu(menu_file_path)

    col1, col2 = st.columns(2)
    menu_html = """
    <style>
        .menu-item {
            background-color: #FFEB3B;
            padding: 10px;
            margin: 5px;
            border-radius: 8px;
            font-family: Arial, sans-serif;
        }
        .menu-header {
            font-size: 24px; /* Increased font size */
            font-weight: bold;
            text-align: center; /* Center align headers */
            margin-bottom: 10px; /* Add some space below headers */
        }
        .menu-subheader {
            font-size: 18px;
            font-weight: bold;
        }
        .menu-text {
            font-size: 16px;
        }
    </style>
    """

    st.markdown(menu_html, unsafe_allow_html=True)

    with col1:
        for line in menu_content[:len(menu_content) // 2]:
            line = line.strip()
            if "Menu:" in line or "Restaurant Name:" in line or "Location:" in line:
                st.markdown(f'<div class="menu-header">{line}</div>', unsafe_allow_html=True)
            elif ":" in line and not "-" in line:
                st.markdown(f'<div class="menu-subheader">🍴 {line}</div>', unsafe_allow_html=True)
            elif " - " in line:
                item_name, item_price = line.split(" - ")
                st.markdown(f'<div class="menu-item"><strong>{item_name}</strong> - {item_price}</div>',
                            unsafe_allow_html=True)

    with col2:
        for line in menu_content[len(menu_content) // 2:]:
            line = line.strip()
            if "Menu:" in line or "Restaurant Name:" in line or "Location:" in line:
                st.markdown(f'<div class="menu-header">{line}</div>', unsafe_allow_html=True)
            elif ":" in line and not "-" in line:
                st.markdown(f'<div class="menu-subheader">🍴 {line}</div>', unsafe_allow_html=True)
            elif " - " in line:
                item_name, item_price = line.split(" - ")
                st.markdown(f'<div class="menu-item"><strong>{item_name}</strong> - {item_price}</div>',
                            unsafe_allow_html=True)

    if st.button("Place Order"):
        # Initialize audio stream and model on button click
        model_size = DEFAULT_MODEL_SIZE + ".en"
        model = WhisperModel(model_size, device="cuda", compute_type="float16", num_workers=10)
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        customer_input_transcription = ""

        try:
            while True:
                chunk_file = "inflight_audio.wav"
                st.markdown("<p style='font-style: italic; font-weight: bold;'>Listening...</p>", unsafe_allow_html=True)
                # Record and process audio
                if not embedd_audio(audio, stream):
                    transcription = transcribe_audio(model, chunk_file)
                    os.remove(chunk_file)
                    st.text(f"Client: {transcription}")

                    # Process customer input and get a response from the AI assistant
                    output = ai_assistant.interact_with_llm(transcription)
                    if output:
                        output = output.lstrip()
                        resp.play_text_to_speech(output)
                        st.text(f"Ani Bot: {output}")

        except KeyboardInterrupt:
            st.text("Session ended. Have a good one! Take care...")

        finally:
            # Ensure the audio stream is properly closed upon exit
            stream.stop_stream()
            stream.close()
            audio.terminate()


if __name__ == "__main__":
    main()
