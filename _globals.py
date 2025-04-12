import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
MODEL_SIZE = r"turbo"
SPLIT_IN_MS = 1100

# censorship 

# Choose Censor Mode: 'silence' or 'wav_insert'
# MODE = 'silence'
CENSOR_MODE = "wav_insert"
INSERT_WAV = "insert.wav"  # MUST PROVIDE PATH for wav_insert

OUTPUT_DIR = "D:\\"  # Optional output directory

print("loading model")
with open("hf_token.txt", "r") as f:
    tmp = f.read()
HF_TOKEN = tmp.strip()
min_silence_duration = 0.08
segment_duration = 3000
tier1_buffer = 1.02
tier2_buffer = 1.003
CURSE_TIER1 = "curse_words_tier1.csv"
CURSE_TIER2 = "curse_words_tier2.csv"
CURSE_EXACT_MATCH = "curse_words_exact_match.csv"
FADE_DURATION = 0.02
CURSE_SOUND = "insert.wav"

sample_audio_path = "looperman.wav"
transcripts = ""
exports = ""
new_trans_path = Path.cwd()
new_trans_path = Path(str(new_trans_path) + "\\transcripts")

transcription_options = {
    "verbose": True,  # Show progress and details
    "temperature": (
         0.0,
         0.2,
         0.4,
         0.6, # added higher temperature steps for more variability if needed
         0.8,
         1.0,
    ),  # Lower temperatures for more accurate transcriptions
    "compression_ratio_threshold": None,  # Slightly more strict to avoid bad compression artifacts
    "logprob_threshold": -2,  # Adjust as needed for your audio characteristics. Lower values increase strictness.
    "no_speech_threshold": 0.1,  # Reduce false positives
    "condition_on_previous_text": True,  # Helps maintain context across segments.
    "word_timestamps": True,  # Necessary for precise timing adjustments.
    "regroup": True,  # Improves segment grouping after VAD.
    "suppress_silence": True,  # Removes silent sections
    "suppress_word_ts": True,  # Adjusts timestamps based on silence
    "use_word_position": True,  # helps with timestamps accuracy in silence suppression
    "vad": True,  # Use Voice Activity Detection for better silence handling.  Might require installing silero-vad separately
    "vad_threshold": 0.1,  # Lower threshold makes VAD more sensitive to quieter speech (experiment!)
    "min_word_dur": 0.02,  # Adjust to minimum length for valid words, avoids small noise bursts getting transcribed as words.
    "min_silence_dur": 0.05,  # Minimum length for silence sections to be suppressed (experiment!)
    "denoiser": "demucs",  # Experiment with this - rnnoise is pretty good. Check the supported_denoiser list in the docs.
    "mel_first": False,  # Avoids excessive memory usage for long files
    "language": "english",
    "nonspeech_skip": 0.3,
    "avg_prob_threshold": -10, # Lower the threshold to capture even low confidence words.
    "suppress_ts_tokens": False, # Ensure timestamp tokens are not suppressed to capture all words.
    "max_instant_words": 0.3, # allow segments composed entirely of instantaneous words (effectively disabling this check)
}
