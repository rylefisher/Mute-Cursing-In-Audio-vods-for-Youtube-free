# censorship.py
import csv
import numpy as np
import wave
import soundfile as sf
from pathlib import Path
from read_ import *
import noisereduce as nr
import threading
import os
import shutil
import json
from _globals import *



class PortableNoiseReduction:
    def __init__(
        self, array: np.ndarray, start_time: float, end_time: float, sample_rate: int
    ):
        self.array = array
        self.start_time = (start_time - 1) if start_time > 0 else 0
        self.end_time = end_time + 1
        self.sample_rate = sample_rate

    def apply_noise_reduction(self):
        # Calculate the start and end indices
        start_sample = int(self.start_time * self.sample_rate)
        end_sample = int(self.end_time * self.sample_rate)

        # Extract segment for noise reduction
        segment = (
            self.array[:, start_sample:end_sample]
            if self.array.ndim == 2
            else self.array[start_sample:end_sample]
        )

        # Apply noise reduction
        reduced_noise_segment = nr.reduce_noise(y=segment, sr=self.sample_rate)

        # Replace original segment with reduced noise segment
        if self.array.ndim == 2:  # Multi-channel
            self.array[:, start_sample:end_sample] = reduced_noise_segment
        else:  # Single channel
            self.array[start_sample:end_sample] = reduced_noise_segment

        return self.array


def read_curse_words_from_csv(CURSE_WORD_FILE):
    curse_words_list = []
    with open(CURSE_WORD_FILE, newline="") as csvfile:
        lines = [line for line in csvfile.readlines() if line != ""]
    lines_update = [line.lower().strip() for line in lines if line != ""]
    return lines_update


def load_wav_as_np_array(wav_file_path):
    # Ensure we handle stereo or mono consistently
    try:
        audio_data, sample_rate = sf.read(wav_file_path, dtype="float32")
        return audio_data, sample_rate
    except Exception as e:
        print(f"An error occurred while reading the WAV file: {e}")
        return None, None


def get_word_samples(word, sample_rate):
    start_time = word["start"]
    end_time = word["end"]
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    return (start_sample, end_sample)


def apply_combined_fades(
    audio, sample_rate, start_time, stop_time, tier, fade_duration=0.01
):
    global tier1_buffer, tier2_buffer
    original_start = start_time
    diff = stop_time - start_time
    if tier == 1:
        buffer = tier1_buffer
    else:
        buffer = tier2_buffer
    # Safeguard against negative durations
    if diff < 0:
        raise ValueError("stop_time must be greater than start_time")

    # Ensure min silence duration
    if diff < min_silence_duration:
        split_silence_minimum = round(min_silence_duration / 2, 3)
        start_time = stop_time - (diff + split_silence_minimum)
        stop_time = original_start + (diff + split_silence_minimum)
        diff = stop_time - start_time
    else:
        # Adjust start_time and stop_time with buff_ratio
        start_time = stop_time - (diff * buffer)
        stop_time = original_start + (diff * buffer)

    # Safeguard against negative start_time
    if start_time < 0:
        start_time = 0

    # Safeguard against exceeding audio length
    if stop_time > len(audio) / sample_rate:
        stop_time = len(audio) / sample_rate

    fade_length = int(fade_duration * sample_rate)
    start_sample = int(start_time * sample_rate)
    stop_sample = int(stop_time * sample_rate)

    # Ensure valid sample indices
    start_sample = max(0, start_sample)
    stop_sample = min(len(audio), stop_sample)

    # Apply fade out
    fade_out_end = start_sample + fade_length
    if fade_out_end > audio.shape[0]:
        fade_out_end = audio.shape[0]
    fade_out_curve = np.linspace(1.0, 0.0, fade_out_end - start_sample)
    audio[start_sample:fade_out_end] *= fade_out_curve

    # Apply fade in
    fade_in_start = stop_sample - fade_length
    if fade_in_start < 0:
        fade_in_start = 0
    fade_in_curve = np.linspace(0.0, 1.0, stop_sample - fade_in_start)
    if fade_in_start < stop_sample:  # Ensure valid range for multiplication
        audio[fade_in_start:stop_sample] *= fade_in_curve

    # Ensure silence between the fades
    if fade_out_end < fade_in_start:
        audio[fade_out_end:fade_in_start] = 0
    return audio


def logger(message):
    with open("log.txt", "w") as f:
        f.write(message + "\n")


import numpy as np


def mute_curse_words(
    audio_data,
    sample_rate,
    transcription_result,
    curse_words_tier1,
    curse_words_tier2,
    log=True,
):
    audio_data_muted = np.copy(audio_data)  # Create copy once for mutation
    any_cursing_found = False
    if log:
        print("\n\n\n\n\n")  # Keep logging as requested

    for word in transcription_result:
        word_text = word["word"].lower()

        if len(word_text) < 3:  # Skip short words early
            continue

        # Check tier1 curses first, then tier2
        matched_curse = next(
            (curse for curse in curse_words_tier1 if curse in word_text), None
        )
        tier = 1 if matched_curse else None

        if not matched_curse:
            matched_curse = next(
                (curse for curse in curse_words_tier2 if curse in word_text), None
            )
            tier = 2 if matched_curse else None

        if matched_curse:
            any_cursing_found = True
            if log:
                print(
                    f"\ncurse:{matched_curse} -> transcript word:{word['word']} -> prob {word['probability']}\n"
                )
            # Pass tier to apply_combined_fades
            audio_data_muted = apply_combined_fades(
                audio_data_muted, sample_rate, word["start"], word["end"], tier
            )

    return audio_data_muted, any_cursing_found


def convert_stereo(f):
    return NumpyMono(f)


curses_tier1 = read_curse_words_from_csv(CURSE_TIER1)
curses_tier1_list = set(curses_tier1)
curses_tier1_set = set(curse.lower() for curse in curses_tier1_list)

curses_tier2 = read_curse_words_from_csv(CURSE_TIER2)
curse_tier2_list = set(curses_tier2)
curse_tier2_set = set(curse.lower() for curse in curse_tier2_list)


def find_curse_words(audio_content, sample_rate, results):
    global curse_words_tier1, curse_tier2_set
    return mute_curse_words(
        audio_content, sample_rate, results, curses_tier1_set, curse_tier2_set
    )


def process_audio_batch(trans_audio):
    max_threads = 8
    threads = []
    processed_paths = {}

    def wait_for_threads(threads):
        for thread in threads:
            thread.join()
        threads.clear()

    threadnumb = 0
    for trans, audio in trans_audio.items():
        threadnumb += 1

        if len(threads) >= max_threads:
            wait_for_threads(threads)

        # Sample way to track processed paths. Implement according to the actual process_audio
        processed_paths[threadnumb] = f"{audio}_processed"

        thread = threading.Thread(target=process_audio, args=(audio, threadnumb, trans))
        threads.append(thread)
        thread.start()

    # Wait for the remaining threads
    wait_for_threads(threads)
    return processed_paths


def combine_wav_files(segment_paths):
    if not segment_paths:
        print("No paths provided!")
        return

    output_nam = Path(segment_paths[0]).name
    output_path = Path(segment_paths[0]).parent.parent / f"{output_nam}combined.wav"
    print(f"\n\ncombining!\n\n{segment_paths}\n\n")
    with wave.open(str(output_path), "w") as outfile:
        # Initialize parameters
        for _, segment_path in enumerate(segment_paths):
            with wave.open(segment_path, "r") as infile:
                if not outfile.getnframes():
                    outfile.setparams(infile.getparams())
                outfile.writeframes(infile.readframes(infile.getnframes()))
            try:
                os.remove(segment_path)
            except OSError as e:
                print(f"Error: {e.strerror}")
    home = os.path.expanduser("~")
    # Construct the path to the user's download folder based on the OS
    download_folder = os.path.join(home, "Downloads")
    # outfile_finished = os.path.join(download_folder, f"{output_nam}combined_output.wav")
    # shutil.copyfile(output_path, outfile_finished)
    return output_path


def convert_json_format(input_filename, output_filename):
    with open(input_filename, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    simplified_data = []
    for segment in data.get("segments", []):
        for word_info in segment.get("words", []):
            simplified_data.append(
                {
                    "word": word_info["word"].strip(r"',.\"-_/`?!; ").lower(),
                    "start": word_info["start"],
                    "end": word_info["end"],
                    "probability": word_info["probability"],
                }
            )

    with open(output_filename, "w", encoding="utf-8") as outfile:
        json.dump(simplified_data, outfile, indent=4)

    print(f"The data has been successfully converted and saved to: {output_filename}")
    return simplified_data, output_filename


def process_audio(audio_file, transcript_file=None):
    global processed_paths
    print("converting to stereo")
    print("reading audio")
    audio_obj = NumpyMono(audio_file)
    print("process json")
    results, clean_json = convert_json_format(
        transcript_file, f"{transcript_file}_new.json"
    )
    print("find curse words")
    audio_obj.np_array, any_cursing_found = find_curse_words(
        audio_obj.np_array, audio_obj.sample_rate, results
    )
    print("exporting file now....")
    audio_obj.numpy_to_wav()
    return audio_obj.output_file_name, clean_json, any_cursing_found
