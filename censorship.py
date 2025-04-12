# censorship.py
import csv
import numpy as np
import wave
import soundfile as sf
from pathlib import Path
from read_ import *
import threading
import os
import shutil
import json
from _globals import *
import re

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


def mute_curse_words(
  audio_data,
  sample_rate,
  transcription_result,
  curse_words_tier1,
  curse_words_tier2,
  curse_words_exact_match, # added parameter
  log=True,
 ):
  audio_data_muted = np.copy(audio_data)
  any_cursing_found = False
  if log:
   print("\n\n\n\n\n")

  for word in transcription_result:
   word_text = word["word"].lower().strip("'\\ - / ?!_ *`,\"")

   if len(word_text) < 3: # skip very short words
    continue

   matched_curse = next(
    (curse for curse in curse_words_tier1 if curse in word_text), None
   )
   tier = 1 if matched_curse else None

   if not matched_curse:
    matched_curse = next(
     (curse for curse in curse_words_tier2 if curse in word_text), None
    )
    tier = 2 if matched_curse else None

   # --- start change: add exact match check ---
   if not matched_curse:
    matched_curse = next(
     (curse for curse in curse_words_exact_match if curse == word_text), None # check for exact match
    )
    tier = 2 if matched_curse else None # using tier 2 fade for exact matches as well, adjust if needed
   # --- end change ---

   if matched_curse:
    any_cursing_found = True
    if log:
     length_temp = word["end"] - word["start"]
     print(
      f"\ncurse:{matched_curse} (Tier {tier}) -> transcript word:{word['word']} -> prob {word['score']} FOR {length_temp:.2f}s (time)\n" # adjusted log
     )
    audio_data_muted = apply_combined_fades(
     audio_data_muted, sample_rate, word["start"], word["end"], tier
    )

  return audio_data_muted, any_cursing_found

def convert_stereo(f):
    return NumpyMono(f)


curses_tier1 = read_curse_words_from_csv(CURSE_TIER1)
curses_tier1_list = set(curses_tier1)
CURSE_WORDS_T1 = set(curse.lower() for curse in curses_tier1_list)

curses_tier2 = read_curse_words_from_csv(CURSE_TIER2)
curse_tier2_list = set(curses_tier2)
CURSE_WORDS_T2 = set(curse.lower() for curse in curse_tier2_list)

curse_exact = read_curse_words_from_csv(CURSE_EXACT_MATCH)
curse_exact_list = set(curse_exact)
CURSE_WORDS_EXACT = set(curse.lower() for curse in curse_exact_list)


def find_curse_words(audio_content, sample_rate, results):
    global curse_words_tier1, curse_tier2_set, curse_exact_set
    return mute_curse_words(
        audio_content,
        sample_rate,
        results,
        CURSE_WORDS_T1,
        CURSE_WORDS_T2,
        CURSE_WORDS_EXACT,
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

        thread = threading.Thread(target=censor_cursewords, args=(audio, threadnumb, trans))
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
    output_path = Path(segment_paths[0]).parent / f"{output_nam}_final.wav"
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


def censor_cursewords(audio_file, results):
    global processed_paths
    print("converting to stereo")
    print("reading audio")
    audio_obj = NumpyMono(audio_file)
    print("process json") 
    print("find curse words")
    audio_obj.np_array, any_cursing_found = find_curse_words(
        audio_obj.np_array, audio_obj.sample_rate, results
    )
    print("exporting file now....")
    audio_obj.numpy_to_wav()
    return audio_obj.output_file_name, any_cursing_found


# ------------------------------------------------
# ------------------------------------------------
# ------------------------------------------------
# ------------------------------------------------
# ------------------------------------------------
# ------------------------------------------------
# ------------------------------------------------
class AudioCensorer:
    """Censors audio based on transcription."""
    def __init__(
        self,  
        censor_mode: str = "silence",  # 'silence' or 'wav_insert'
        output_suffix: str = "_censored",  # Suffix for output file
    ):
        """Initializes the censorer.""" 
        if censor_mode not in ["silence", "wav_insert"]:
            raise ValueError("censor_mode must be 'silence' or 'wav_insert'")
        if censor_mode == "wav_insert" and INSERT_WAV is None:
            raise ValueError("insert_wav_path required for 'wav_insert' mode")
        if censor_mode == "wav_insert" and not os.path.exists(INSERT_WAV):
            raise FileNotFoundError(f"Insert WAV not found: {INSERT_WAV}")

        self.curses_tier1 = CURSE_WORDS_T1
        self.curses_tier2 = CURSE_WORDS_T2
        self.curses_exact = CURSE_WORDS_EXACT
        self.tier1_buffer = tier1_buffer
        self.tier2_buffer = tier2_buffer
        self.min_silence_duration = min_silence_duration
        self.fade_duration = FADE_DURATION
        self.censor_mode = censor_mode
        self.insert_wav_path = CURSE_SOUND
        self.output_suffix = output_suffix
        self._insert_audio_data = None  # Loaded insert audio
        self._insert_audio_samplerate = None  # Loaded insert sr

        # Pre-compile regex for word cleaning
        self._clean_word_regex = re.compile(r"['\\ \- / ?!_ *`,.\"$]+")

        # Load insert WAV immediately if needed
        if self.censor_mode == "wav_insert":
            self._load_insert_wav()

    def _load_insert_wav(self):
        """Loads the insert WAV file."""
        if not self.insert_wav_path:
            return  # No path provided

        try:
            data, sr = sf.read(self.insert_wav_path, dtype="float32", always_2d=True)
            self._insert_audio_data = data
            self._insert_audio_samplerate = sr
            # Ensure mono if needed later, keep original channels for now
            # print(f"Loaded insert WAV: {self.insert_wav_path}, SR: {sr}, Shape: {data.shape}")

        except Exception as e:
            print(f"ERROR: Error loading insert WAV: {e}")
            self._insert_audio_data = None
            self._insert_audio_samplerate = None
            # Decide if this should raise or just prevent wav_insert mode
            # raise # Optionally re-raise

    def _get_resampled_insert_audio(self, target_sr: int) -> np.ndarray | None:
        """Gets insert audio, resamples if necessary."""
        if self._insert_audio_data is None or self._insert_audio_samplerate is None:
            # Attempt to load if not already loaded
            self._load_insert_wav()
            if self._insert_audio_data is None:
                print("WARN: Insert audio not loaded, cannot resample.")
                return None  # Failed to load

        if self._insert_audio_samplerate == target_sr:
            return self._insert_audio_data  # No resampling needed

        # Resample
        try:
            # Check if librosa is available without importing globally
            import importlib

            librosa_spec = importlib.util.find_spec("librosa")
            if librosa_spec is None:
                print(
                    "WARN: librosa not found, cannot resample insert audio. pip install librosa. Skipping insert."
                )
                return None

            import librosa  # Now import it

            # Librosa expects shape (channels, samples) or (samples,)
            # Convert from (samples, channels) [soundfile] to (channels, samples) [librosa]
            data_for_resample = self._insert_audio_data.T
            if data_for_resample.shape[0] == 1:  # If mono, use 1D array
                data_for_resample = data_for_resample[0]

            resampled_data = librosa.resample(
                data_for_resample,
                orig_sr=self._insert_audio_samplerate,
                target_sr=target_sr,
            )

            # Convert back to (samples, channels) or ensure 2D if needed
            if resampled_data.ndim == 1:
                resampled_data = resampled_data[:, np.newaxis]  # Make 2D (samples, 1)
            else:
                resampled_data = resampled_data.T  # Transpose back (samples, channels)

            print(
                f"Resampled insert audio from {self._insert_audio_samplerate} Hz to {target_sr} Hz"
            )
            # Note: This resamples every time if SR mismatches. Consider caching.
            return resampled_data

        except Exception as e:
            print(f"ERROR: Error resampling insert audio: {e}. Skipping insert.")
            return None

    def _apply_crossfade(
        self,
        audio: np.ndarray,
        sample_rate: int,
        start_time: float,
        stop_time: float,
        tier: int,
    ) -> np.ndarray:
        """Applies cross-fade censorship."""
        # 1. Calculate adjusted time range with buffer
        original_start = start_time
        original_stop = stop_time
        diff = original_stop - original_start

        if diff < 0:
            print(f"WARN: stop_time {stop_time} < start_time {start_time}. Skipping.")
            return audio

        buffer = self.tier1_buffer if tier == 1 else self.tier2_buffer
        target_diff = max(diff * buffer, self.min_silence_duration)
        expansion = max(0, (target_diff - diff) / 2)

        adj_start_time = max(0, start_time - expansion)
        adj_stop_time = stop_time + expansion  # Let validation handle audio end

        # 2. Convert times to samples and validate
        audio_duration_samples = audio.shape[0]
        audio_duration_secs = audio_duration_samples / sample_rate
        adj_stop_time = min(adj_stop_time, audio_duration_secs)  # Clamp stop time

        start_sample = int(adj_start_time * sample_rate)
        stop_sample = int(adj_stop_time * sample_rate)

        # Ensure valid range within audio bounds
        start_sample = max(0, min(start_sample, audio_duration_samples))
        stop_sample = max(0, min(stop_sample, audio_duration_samples))

        censor_duration_samples = stop_sample - start_sample
        if censor_duration_samples <= 0:
            # print("WARN: Calculated censor duration is zero or negative. Skipping.")
            return audio  # No change needed

        # 3. Calculate fade length in samples
        fade_length = int(self.fade_duration * sample_rate)
        # Fade cannot be longer than half the censored duration
        fade_length = max(0, min(fade_length, censor_duration_samples // 2))

        if fade_length <= 0:
            # print("WARN: Fade length is zero. Applying abrupt change.")
            # Fallback to simple replacement if no fade possible
            if self.censor_mode == "silence":
                audio[start_sample:stop_sample] = 0
            elif self.censor_mode == "wav_insert":
                insert_audio = self._get_resampled_insert_audio(sample_rate)
                if insert_audio is not None:
                    # Basic tiling/truncating for replacement
                    num_repeats = int(
                        np.ceil(censor_duration_samples / len(insert_audio))
                    )
                    tiled_insert = np.tile(insert_audio, (num_repeats, 1))[
                        :censor_duration_samples
                    ]

                    # Match channels
                    if audio.ndim == 1 and tiled_insert.shape[1] > 1:
                        tiled_insert = np.mean(tiled_insert, axis=1)
                    elif audio.ndim > 1 and tiled_insert.shape[1] == 1:
                        if audio.shape[1] == 2:  # Specific common case: stereo
                            tiled_insert = np.column_stack((tiled_insert, tiled_insert))
                        else:  # Generic channel duplication
                            tiled_insert = np.tile(tiled_insert, (1, audio.shape[1]))
                    elif audio.ndim > 1 and audio.shape[1] != tiled_insert.shape[1]:
                        print(
                            f"WARN: Channel mismatch ({audio.shape[1]} vs {tiled_insert.shape[1]}). Using silence."
                        )
                        tiled_insert = np.zeros(
                            (censor_duration_samples, audio.shape[1]), dtype=audio.dtype
                        )

                    if audio[start_sample:stop_sample].shape == tiled_insert.shape:
                        audio[start_sample:stop_sample] = tiled_insert
                    else:  # Fallback if shapes somehow mismatch
                        print(
                            f"WARN: Shape mismatch during abrupt replace. Audio:{audio[start_sample:stop_sample].shape}, Insert:{tiled_insert.shape}. Using silence."
                        )
                        audio[start_sample:stop_sample] = 0

                else:  # Insert failed to load/resample
                    audio[start_sample:stop_sample] = 0  # Use silence
            return audio

        # 4. Prepare the insert audio chunk (or silence)
        insert_chunk: np.ndarray | None = None
        if self.censor_mode == "silence":
            insert_chunk = np.zeros(
                (censor_duration_samples, audio.shape[1] if audio.ndim > 1 else 1),
                dtype=audio.dtype,
            )
        elif self.censor_mode == "wav_insert":
            base_insert_audio = self._get_resampled_insert_audio(sample_rate)
            if base_insert_audio is not None:
                # Tile or truncate the loaded/resampled insert audio
                insert_len_samples = base_insert_audio.shape[0]
                if insert_len_samples == 0:
                    print("WARN: Insert audio data is empty. Using silence.")
                    insert_chunk = np.zeros(
                        (
                            censor_duration_samples,
                            audio.shape[1] if audio.ndim > 1 else 1,
                        ),
                        dtype=audio.dtype,
                    )
                else:
                    num_repeats = int(
                        np.ceil(censor_duration_samples / insert_len_samples)
                    )
                    # Ensure base_insert_audio has correct channels before tiling
                    target_channels = audio.shape[1] if audio.ndim > 1 else 1

                    if base_insert_audio.shape[1] != target_channels:
                        if base_insert_audio.shape[1] == 1 and target_channels > 1:
                            # Duplicate mono to match target channels
                            base_insert_audio = np.tile(
                                base_insert_audio, (1, target_channels)
                            )
                        elif base_insert_audio.shape[1] > 1 and target_channels == 1:
                            # Mix down multi-channel insert to mono
                            base_insert_audio = np.mean(
                                base_insert_audio, axis=1, keepdims=True
                            )
                        else:  # Mismatched multi-channel (e.g., 5.1 vs stereo) - mix down for now
                            print(
                                f"WARN: Complex channel mismatch ({base_insert_audio.shape[1]} vs {target_channels}). Mixing insert to mono and duplicating."
                            )
                            mono_insert = np.mean(
                                base_insert_audio, axis=1, keepdims=True
                            )
                            base_insert_audio = np.tile(
                                mono_insert, (1, target_channels)
                            )

                    tiled_insert = np.tile(base_insert_audio, (num_repeats, 1))
                    insert_chunk = tiled_insert[:censor_duration_samples]

                    # Final shape check
                    expected_shape = (
                        (censor_duration_samples,)
                        if target_channels == 1
                        else (censor_duration_samples, target_channels)
                    )
                    if insert_chunk.shape != expected_shape:
                        print(
                            f"WARN: Insert chunk shape mismatch after tiling. Expected {expected_shape}, got {insert_chunk.shape}. Using silence."
                        )
                        insert_chunk = np.zeros(expected_shape, dtype=audio.dtype)

            else:
                # Failed to get insert audio, fallback to silence
                insert_chunk = np.zeros(
                    (censor_duration_samples, audio.shape[1] if audio.ndim > 1 else 1),
                    dtype=audio.dtype,
                )

        # Ensure insert_chunk matches audio ndim for broadcasting
        if audio.ndim == 1 and insert_chunk.ndim > 1:
            insert_chunk = insert_chunk[:, 0]  # Use first channel if audio is mono
        elif audio.ndim > 1 and insert_chunk.ndim == 1:
            insert_chunk = insert_chunk[
                :, np.newaxis
            ]  # Make it (samples, 1) for broadcasting

        # 5. Create fade curves
        fade_in_curve = np.linspace(0.0, 1.0, fade_length).astype(audio.dtype)
        fade_out_curve = np.linspace(1.0, 0.0, fade_length).astype(audio.dtype)

        # Add channel dimension if audio is multi-channel for broadcasting
        if audio.ndim > 1:
            fade_in_curve = fade_in_curve[:, np.newaxis]
            fade_out_curve = fade_out_curve[:, np.newaxis]

        # 6. Apply cross-fade at the start
        start_fade_slice = slice(start_sample, start_sample + fade_length)
        original_start_segment = audio[start_fade_slice]
        insert_start_segment = insert_chunk[0:fade_length]

        try:
            audio[start_fade_slice] = (
                original_start_segment * fade_out_curve
                + insert_start_segment * fade_in_curve
            )
        except ValueError as e:
            print(f"ERROR: ValueError during start crossfade: {e}")
            print(
                f"Shapes: original={original_start_segment.shape}, insert={insert_start_segment.shape}, fade_out={fade_out_curve.shape}, fade_in={fade_in_curve.shape}"
            )
            # Fallback or re-raise depending on desired robustness
            audio[start_fade_slice] = insert_start_segment  # Simple replace as fallback

        # 7. Replace the middle section (if any)
        middle_start = start_sample + fade_length
        middle_end = stop_sample - fade_length
        if middle_start < middle_end:
            middle_slice = slice(middle_start, middle_end)
            insert_middle_segment = insert_chunk[
                fade_length : censor_duration_samples - fade_length
            ]
            try:
                audio[middle_slice] = insert_middle_segment
            except ValueError as e:
                print(f"ERROR: ValueError during middle replace: {e}")
                print(
                    f"Shapes: audio_slice={audio[middle_slice].shape}, insert_middle={insert_middle_segment.shape}"
                )
                # Fallback
                audio[middle_slice] = 0

        # 8. Apply cross-fade at the end
        end_fade_slice = slice(stop_sample - fade_length, stop_sample)
        original_end_segment = audio[end_fade_slice]
        insert_end_segment = insert_chunk[
            censor_duration_samples - fade_length : censor_duration_samples
        ]

        try:
            audio[end_fade_slice] = (
                insert_end_segment * fade_out_curve
                + original_end_segment * fade_in_curve
            )
        except ValueError as e:
            print(f"ERROR: ValueError during end crossfade: {e}")
            print(
                f"Shapes: original={original_end_segment.shape}, insert={insert_end_segment.shape}, fade_out={fade_out_curve.shape}, fade_in={fade_in_curve.shape}"
            )
            # Fallback
            audio[end_fade_slice] = original_end_segment  # Keep original as fallback

        return audio

    def _clean_word(self, word: str) -> str:
        """Removes punctuation and lowers case."""
        return self._clean_word_regex.sub("", word.lower())

    def censor_audio_data(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        transcription_results: list[dict],
        log: bool = True,
    ) -> tuple[np.ndarray, bool]:
        """Censors curse words via cross-fade."""
        modified_audio = np.copy(audio_data)  # Work on a copy
        any_cursing_found = False

        # Pre-check insert audio compatibility once
        if self.censor_mode == "wav_insert":
            # Attempt load/resample check early to avoid repeated checks/errors
            _ = self._get_resampled_insert_audio(sample_rate)

        if log:
            print("\n--- Starting Censorship ---")

        # Sort results by start time to avoid potential issues with overlapping buffers
        # Although current logic processes independently, sorting is safer.
        sorted_results = sorted(transcription_results, key=lambda x: x.get("start", 0))

        for word_info in sorted_results:
            if not all(k in word_info for k in ["word", "start", "end"]):
                if log:
                    print(f"WARN: Skipping invalid transcript entry: {word_info}")
                continue

            word_text_original = word_info["word"]
            word_text_cleaned = self._clean_word(word_text_original)

            if len(word_text_cleaned) < 2:  # Skip very short/empty words
                continue

            matched_curse = None
            tier = None

            # Exact matches first (most specific)
            if word_text_cleaned in self.curses_exact:
                matched_curse = word_text_cleaned
                tier = 2  # Assign tier (adjust if needed)
            else:
                # Tier 1 substring match
                for curse in self.curses_tier1:
                    if curse in word_text_cleaned:
                        matched_curse = curse
                        tier = 1
                        break  # Found highest priority match

                # Tier 2 substring match (if no T1 found)
                if not matched_curse:
                    for curse in self.curses_tier2:
                        if curse in word_text_cleaned:
                            matched_curse = curse
                            tier = 2
                            break

            if matched_curse and tier is not None:
                any_cursing_found = True
                start_time = word_info["start"]
                end_time = word_info["end"]

                if log:
                    duration = end_time - start_time
                    score = word_info.get("score", "N/A")
                    score_str = f"{score:.2f}" if isinstance(score, float) else score
                    print(
                        f"Found: '{matched_curse}' (T{tier}) in '{word_text_original}' "
                        f"@{start_time:.2f}s-{end_time:.2f}s ({duration:.2f}s) "
                        f"[Score: {score_str}] -> Applying {self.censor_mode} with crossfade"
                    )

                # Apply the cross-fade logic
                modified_audio = self._apply_crossfade(
                    modified_audio, sample_rate, start_time, end_time, tier
                )
            # --- Removed redundant fade logic ---

        if log:
            print(f"--- Censorship Complete. Found Curses: {any_cursing_found} ---")

        return modified_audio, any_cursing_found

    def censor_audio_file(
        self,
        audio_file_path: str,
        transcription_results: list[dict],
        output_dir: str | None = None,
        log: bool = True,
    ) -> tuple[str | None, bool]:
        """Loads, censors (cross-fade), saves file."""
        if not os.path.exists(audio_file_path):
            print(f"ERROR: Audio file not found: {audio_file_path}")
            return None, False

        try:
            if log:
                print(f"\nReading audio: {audio_file_path}")
            # Read as float32, keep original channels (always_2d=True)
            audio_data, sample_rate = sf.read(
                audio_file_path, dtype="float32", always_2d=True
            )
            original_ndim = audio_data.ndim  # Store original dimensions if needed later
            # Handle mono files read as (samples, 1) if downstream expects 1D
            # For consistency within this class, keep as 2D and handle channel logic inside.

        except Exception as e:
            print(f"ERROR: Failed to read audio file {audio_file_path}: {e}")
            return None, False

        if log:
            print(f"Processing {len(transcription_results)} transcription words...")

        censored_audio, any_cursing_found = self.censor_audio_data(
            audio_data,
            sample_rate,
            transcription_results,
            log=log,
        )

        if not any_cursing_found:
            if log:
                print("No curse words found matching criteria. No file written.")
            return None, False  # Nothing to save

        # Save the censored audio
        output_path = None  # Initialize
        try:
            base, ext = os.path.splitext(os.path.basename(audio_file_path))
            # Ensure the original extension is preserved (e.g., .wav, .flac)
            output_filename = f"{base}{self.output_suffix}{ext}"

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, output_filename)
            else:
                # Place censored file in the same directory as the original
                output_path = os.path.join(
                    os.path.dirname(audio_file_path), output_filename
                )

            if log:
                print(f"Exporting censored file to: {output_path}...")

            # Get subtype info from original file for quality preservation
            try:
                info = sf.info(audio_file_path)
                subtype = info.subtype
            except Exception as info_e:
                print(
                    f"WARN: Could not read info from {audio_file_path}: {info_e}. Using default subtype."
                )
                subtype = None  # Let soundfile choose default

            # Write the file
            sf.write(output_path, censored_audio, sample_rate, subtype=subtype)

            if log:
                print("Export complete.")
            return output_path, any_cursing_found

        except Exception as e:
            # Ensure output_path is defined for the error message
            path_str = output_path if output_path else "intended output location"
            print(f"ERROR: Failed to write censored audio file {path_str}: {e}")
            # Return None for path, but still indicate if cursing was found
            return None, any_cursing_found


# Example Usage (replace with your actual data)
if __name__ == "__main__":
    # --- Configuration ---
    CURSE_WORDS_T1 = {"damn", "hell"}  # Substring matches, shorter fade/buffer
    CURSE_WORDS_T2 = {
        "fuck",
        "shit",
        "asshole",
    }  # Substring matches, longer fade/buffer
    CURSE_WORDS_EXACT = {"crap"}  # Exact word match only (Tier 2 buffer)

    # # Choose Censor Mode: 'silence' or 'wav_insert'
    # # MODE = 'silence'
    MODE = "wav_insert"

    INPUT_AUDIO = "D:\\1_YG - _3840_(Vocals).wav"
    OUTPUT_DIR = "D:\\"  # Optional output directory

    # Example transcription results (from Whisper, etc.)
    # Ensure 'start' and 'end' times are accurate
    TRANSCRIPTION = [
        {"word": "Well", "start": 0.5, "end": 0.8, "score": 0.99},
        {"word": "damn", "start": 1.0, "end": 1.4, "score": 0.95},
        {"word": "it", "start": 1.4, "end": 1.6, "score": 0.98},
        {"word": "this", "start": 2.0, "end": 2.3, "score": 0.99},
        {"word": "is", "start": 2.3, "end": 2.5, "score": 0.99},
        {
            "word": "fucking",
            "start": 2.8,
            "end": 3.3,
            "score": 0.85,
        },  # Substring match 'fuck'
        {"word": "annoying", "start": 3.3, "end": 4.0, "score": 0.92},
        {"word": "Oh", "start": 4.5, "end": 4.7, "score": 0.99},
        {
            "word": "crap.",
            "start": 5.0,
            "end": 5.5,
            "score": 0.91,
        },  # Exact match 'crap'
    ]

    # --- Create and Run Censorer ---
    try:
        # Create the censorer instance
        censorer = AudioCensorer(
            censor_mode=CENSOR_MODE,
        )

        # Process the audio file
        output_file, was_censored = censorer.censor_audio_file(
            INPUT_AUDIO, TRANSCRIPTION, output_dir=OUTPUT_DIR, log=True
        )

        if output_file:
            print(f"\nSuccess! Censored file saved to: {output_file}")
            print(f"Censorship applied: {was_censored}")
        else:
            print("\nCensorship process failed or no file was written.")

    except FileNotFoundError as e:
        print(f"\nERROR: Required file not found - {e}")
    except ValueError as e:
        print(f"\nERROR: Configuration error - {e}")
    except ImportError as e:
        print(f"\nERROR: Missing dependency (likely librosa for resampling) - {e}")
        print("Please install it: pip install librosa soundfile")
    except Exception as e:
        import traceback

        print(f"\nAn unexpected error occurred: {e}")
        print(traceback.format_exc())  # Print full traceback for debugging
