from pydub import AudioSegment
import numpy as np
from scipy.io import wavfile
import noisereduce as nr
import librosa
import soundfile as sf
import os
import json

FADE = librosa.ex('brahms')

def read_audio_file(filename):
    """
    Reads an audio file (.wav or .mp3) and returns it in a mono format,
    normalized between -1.0 and 1.0.
    """
    # Using librosa to load audio files. `sr=None` loads the file in its original sample rate
    # mono=True ensures audio is mono
    audio, sample_rate = librosa.load(filename, sr=None, mono=True)
    return audio, sample_rate


def numpy_to_wav(filename, samples, sample_rate):
    """
    Writes a numpy array of samples to a WAV file with the given sample rate.
    """
    # librosa outputs float32 arrays for audio, soundfile can directly write this to WAV
    sf.write(filename, samples, sample_rate)


def numpy_to_wav(filename, samples, sample_rate):
    """
    Writes a numpy array of samples to a WAV file with the given sample rate.
    """
    # librosa outputs float32 arrays for audio, soundfile can directly write this to WAV
    sf.write(filename, samples, sample_rate) 
    # perform noise reduction 

class NumpyMono:
    def __init__(self, audio_file_path):
        self.audio_file_path = audio_file_path
        self.np_array, self.sample_rate = read_audio_file(audio_file_path)
        self.output_file_name = audio_file_path.replace('.wav','') + "_cln.wav"

    def numpy_to_wav(self):
        # Exporting numpy array to a .wav file
        numpy_to_wav(self.output_file_name,
            self.np_array, self.sample_rate)
        print('File saved as:', self.output_file_name)


if __name__ == "__main__":
    audio_file_path = 'path/to/your/audiofile.mp3'  # or .wav
    output_file_name = 'output_filename'
    NumpyMono(audio_file_path, output_file_name)


class JSONLog:
    def __init__(self, wav_file):
        self.wav_file = wav_file
        self.log_folder = os.path.join(
            os.path.expanduser('~'), 'Documents', 'transcripter')
        self.log_file = os.path.join(self.log_folder, 'log.json')
        self.ensure_log_exists()

    def ensure_log_exists(self):
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)
        if not os.path.isfile(self.log_file):
            with open(self.log_file, 'w') as log_json:
                initial_content = {
                    'status': 'initialized', 'files_processed': []}
                json.dump(initial_content, log_json)

    def update_log(self, update_dict):
        with open(self.log_file, 'r+') as log_json:
            content = json.load(log_json)
            content.update(update_dict)
            log_json.seek(0)
            json.dump(content, log_json, indent=4)
            log_json.truncate()

    def check_value(self, key):
        with open(self.log_file, 'r') as log_json:
            content = json.load(log_json)
            return content.get(key, None)
        
    

"""def read_audio_file(file_path):
Reads an audio file(.wav or .mp3) and returns it in a mono format.
    if file_path.endswith(".mp3"):
        audio = AudioSegment.from_mp3(file_path)
    elif file_path.endswith(".wav"):
        audio = AudioSegment.from_wav(file_path)
    else:
        raise Exception("Unsupported file format. Please use WAV or MP3.")
    audio_mono = audio.set_channels(1)
    return audio_mono


def audio_to_numpy(audio_mono):
Converts mono audio data to a numpy array.
    samples = np.array(audio_mono.get_array_of_samples())
    return samples.astype(np.float32)


def numpy_to_wav(np_array, sample_rate, file_name):
Converts a numpy array to a .wav file.
    audio_bytes = np_array.astype(np.int16).tobytes()

    # Creating an audio segment with the raw audio data
    audio = AudioSegment(data=audio_bytes, sample_width=2,
                         frame_rate=sample_rate, channels=1)
    # Exporting the audio segment to a .wav file
    audio.export(file_name, format="wav")


class NumpyMono:
    def __init__(self, audio_file_path):
        self.audio_file_path = audio_file_path
        self.audio_mono = read_audio_file(audio_file_path)
        self.np_array = audio_to_numpy(self.audio_mono)
        # The sample_rate needs to be obtained from the original file to accurately convert back to .wav
        self.sample_rate = self.audio_mono.frame_rate
        self.output_file_name = audio_file_path + "_clean_.wav"

    def numpy_to_wav(self):
        # Exporting numpy array to a .wav file
        numpy_to_wav(
            self.np_array, self.sample_rate, self.output_file_name)"""
