import os
import wave
import pyaudio
import numpy as np
from scipy.io import wavfile

# Default length of audio chunks in seconds
DEFAULT_CHUNK_LENGTH = 10


def check_call_pause(data, max_amplitude_threshold=3000):
    """
    Check if audio data contains silence based on a maximum amplitude threshold.

    Args:
        data (numpy.ndarray): The audio data to be analyzed.
        max_amplitude_threshold (int, optional): The amplitude threshold below which
                                                 the audio is considered silent. Defaults to 3000.

    Returns:
        bool: True if the maximum amplitude is below or equal to the threshold, indicating silence.
              False otherwise.
    """
    max_amplitude = np.max(np.abs(data))
    return max_amplitude <= max_amplitude_threshold


def embedd_audio(audio, stream, chunk_length=DEFAULT_CHUNK_LENGTH):
    """
    Record a chunk of audio, save it to a temporary file, and check if it contains silence.

    Args:
        audio (pyaudio.PyAudio): The PyAudio instance used to get sample size.
        stream (pyaudio.Stream): The audio stream from which to read data.
        chunk_length (int, optional): The length of the audio chunk to record in seconds.
                                      Defaults to DEFAULT_CHUNK_LENGTH.

    Returns:
        bool: True if the recorded chunk contains silence, False otherwise.
    """
    frames = []

    # Read audio data in chunks and append to frames
    for _ in range(0, int(16000 / 1024 * chunk_length)):
        data = stream.read(1024)
        frames.append(data)

    temp_file_path = 'temp_audio_chunk.wav'

    # Save the recorded frames to a temporary WAV file
    with wave.open(temp_file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))

    # Check if the recorded chunk contains silence
    try:
        samplerate, data = wavfile.read(temp_file_path)
        if check_call_pause(data):
            os.remove(temp_file_path)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error while reading audio file: {e}")
        return False


def transcribe_audio(model, file_path):
    """
    Transcribe the audio file using a provided transcription model.

    Args:
        model: The transcription model to use.
        file_path (str): The path to the audio file to transcribe.

    Returns:
        str: The transcription of the audio file.
    """
    segments, info = model.transcribe(file_path, beam_size=7)
    transcription = ' '.join(segment.text for segment in segments)
    return transcription
