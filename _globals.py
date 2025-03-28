import stable_whisper
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
MODEL_SIZE = r"turbo"
SPLIT_IN_MS = 20
print("loading model")
MODEL = stable_whisper.load_model(MODEL_SIZE, device="cuda")

min_silence_duration = 0.015
segment_duration = 3000
tier1_buffer = 1.01
tier2_buffer = 0.8
CURSE_TIER1 = "curse_words_tier1.csv"
CURSE_TIER2 = "curse_words_tier2.csv"

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
    ),  # Lower temperatures for more accurate transcriptions
    "compression_ratio_threshold": 2.0,  # Slightly more strict to avoid bad compression artifacts
    "logprob_threshold": -2,  # Adjust as needed for your audio characteristics. Lower values increase strictness.
    "no_speech_threshold": 0.3,  # Reduce false positives
    "condition_on_previous_text": True,  # Helps maintain context across segments.
    "word_timestamps": True,  # Necessary for precise timing adjustments.
    "regroup": True,  # Improves segment grouping after VAD.
    "suppress_silence": True,  # Removes silent sections
    "suppress_word_ts": True,  # Adjusts timestamps based on silence
    "use_word_position": True,  # helps with timestamps accuracy in silence suppression
    "vad": True,  # Use Voice Activity Detection for better silence handling.  Might require installing silero-vad separately
    "vad_threshold": 0.14,  # Lower threshold makes VAD more sensitive to quieter speech (experiment!)
    "min_word_dur": 0.05,  # Adjust to minimum length for valid words, avoids small noise bursts getting transcribed as words.
    "min_silence_dur": 0.15,  # Minimum length for silence sections to be suppressed (experiment!)
    "denoiser": "demucs",  # Experiment with this - rnnoise is pretty good. Check the supported_denoiser list in the docs.
    "mel_first": False,  # Avoids excessive memory usage for long files
    "language": "english",
    "nonspeech_skip": 1
}
