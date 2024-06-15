import os
import time
import pygame
from gtts import gTTS


def play_text_to_speech(text, language='en', slow=False):
    """
    Converts the provided text to speech and plays it using the gTTS and pygame libraries.

    Args:
        text (str): The text to be converted to speech.
        language (str): The language of the speech. Defaults to 'en' for English.
        slow (bool): If True, the speech will be slower than normal. Defaults to False.

    Process:
        - Converts the provided text to speech using gTTS (Google Text-to-Speech).
        - Saves the generated speech to a temporary audio file.
        - Initializes pygame mixer to play the audio file.
        - Plays the audio file and waits until it finishes playing.
        - Stops and quits the pygame mixer.
        - Deletes the temporary audio file after playback.

    Example:
        play_text_to_speech("Hello, world!")
    """
    # Convert text to speech
    tts = gTTS(text=text, lang=language, slow=slow)

    # Save the speech to a temporary file
    temp_audio_file = "audit_audio_record.mp3"
    tts.save(temp_audio_file)

    # Initialize pygame mixer
    pygame.mixer.init()

    # Load and play the audio file
    pygame.mixer.music.load(temp_audio_file)
    pygame.mixer.music.play()

    # Wait for the audio to finish playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    # Stop and quit the mixer
    pygame.mixer.music.stop()
    pygame.mixer.quit()

    # Pause briefly to ensure file is not in use and then delete the temporary file
    time.sleep(3)
    os.remove(temp_audio_file)
