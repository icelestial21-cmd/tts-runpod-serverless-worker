"""
rp_handler.py

This is the main handler for the runpod TTS worker with multi-speaker support.
The API now expects the "text" field as a list of pairs where each pair contains a speaker ID
and its corresponding text to synthesize. For each speaker, TTS synthesis is performed using
the RUSynth library (with long_text mode always enabled), and the resulting audio segments are
concatenated with short silence intervals. Optionally, audio enhancement is applied using the
AudioEnhancer from resemble_enhance. All models are loaded from local directories specified via
environment variables. GPU is used if available.
"""

import io
import os
import base64
import pathlib
from typing import Any, Dict, List, Tuple

import runpod
import torch
import numpy as np
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils.rp_upload import upload_in_memory_object
from runpod.serverless.utils import rp_cleanup
from scipy.io.wavfile import write
from pydub import AudioSegment

# Import the TTS synthesizer from the rusynth library
from rusynth import RUSynth
# Import the AudioEnhancer from resemble_enhance
from audio_enhancer import AudioEnhancer
# Import the input schema for validation
from rp_schema import INPUT_SCHEMA

# Set model directories from environment variables.
TTS_MODEL_DIR: str = os.getenv("WORKER_TTS_MODEL_DIR", "/app/model/tts")
AUDIO_ENHANCER_DIR: str = os.getenv("WORKER_AUDIO_ENHANCER_DIR", "/app/model/audio_enhancer")

# Load the TTS model using RUSynth from the local directory.
TTS_MODEL: RUSynth = RUSynth(TTS_MODEL_DIR)

# Determine the device: use CUDA if enabled and available.
USE_CUDA: bool = os.environ.get('WORKER_USE_CUDA', 'True').lower() == 'true'
DEVICE: str = "cuda" if USE_CUDA and torch.cuda.is_available() else "cpu"

# Load the audio enhancer from the local directory.
AUDIO_ENHANCER: AudioEnhancer = AudioEnhancer.from_pretrained(
    pathlib.Path(AUDIO_ENHANCER_DIR) / "enhancer_stage2",
    device=DEVICE
)

def upload_audio(wav: np.ndarray, sample_rate: int, key: str) -> str:
    """
    Converts the audio numpy array to bytes and uploads it to S3 (if configured),
    otherwise returns a base64-encoded string.

    Args:
        wav: Audio data as a numpy array.
        sample_rate: Sampling rate of the audio.
        key: The key or filename for the uploaded object.

    Returns:
        A string representing the uploaded file URL or the base64 encoded audio.
    """
    wav_io = io.BytesIO()
    write(wav_io, sample_rate, wav)
    wav_bytes: bytes = wav_io.getvalue()
    if os.environ.get('BUCKET_ENDPOINT_URL', False):
        return upload_in_memory_object(
            key,
            wav_bytes,
            bucket_creds={
                "endpointUrl": os.environ.get('BUCKET_ENDPOINT_URL', None),
                "accessId": os.environ.get('BUCKET_ACCESS_KEY_ID', None),
                "accessSecret": os.environ.get('BUCKET_SECRET_ACCESS_KEY', None)
            }
        )
    return base64.b64encode(wav_bytes).decode('utf-8')

def upload_bytes(bytes_array: bytes, key: str) -> str:
    """
    Upload bytes to S3 (if configured), otherwise returns a base64-encoded string.

    Args:
        bytes_array: bytes to upload.
        key: The key or filename for the uploaded object.

    Returns:
        A string representing the uploaded file URL or the base64 encoded audio.
    """
    if os.environ.get('BUCKET_ENDPOINT_URL', False):
        return upload_in_memory_object(
            key,
            bytes_array,
            bucket_creds={
                "endpointUrl": os.environ.get('BUCKET_ENDPOINT_URL', None),
                "accessId": os.environ.get('BUCKET_ACCESS_KEY_ID', None),
                "accessSecret": os.environ.get('BUCKET_SECRET_ACCESS_KEY', None)
            }
        )
    return base64.b64encode(wav_bytes).decode('utf-8')

def wav_to_bytes(wav: np.ndarray, sample_rate: int) -> str:
    """
    Converts the audio numpy array to bytes.

    Args:
        wav: Audio data as a numpy array.
        sample_rate: Sampling rate of the audio.

    Returns:
        A string representing the uploaded file URL or the base64 encoded audio.
    """
    wav_io = io.BytesIO()
    write(wav_io, sample_rate, wav)
    wav_bytes: bytes = wav_io.getvalue()
    return wav_bytes

def wav_to_mp3_bytes(wav: np.ndarray, sample_rate: int) -> bytes:
    """
    Converts a WAV numpy array (int16) to MP3 format using pydub.

    Args:
        wav: Audio data as a numpy array in int16 format.
        sample_rate: Sampling rate of the audio.

    Returns:
        Audio data in MP3 format as bytes.
    """
    # Create an AudioSegment from raw audio data.
    audio_segment = AudioSegment(
        data=wav.tobytes(),
        sample_width=2,   # int16 -> 2 bytes
        frame_rate=sample_rate,
        channels=1
    )
    mp3_io = io.BytesIO()
    audio_segment.export(mp3_io, format="mp3")
    return mp3_io.getvalue()

def concatenate_audios(audios: List[np.ndarray], sample_rate: int, silence_duration: float = 0.2) -> np.ndarray:
    """
    Concatenates a list of audio segments with a silence gap between them.

    Args:
        audios: List of numpy arrays representing audio segments.
        sample_rate: Sampling rate of the audio.
        silence_duration: Duration of silence (in seconds) to insert between segments.

    Returns:
        A single numpy array containing the concatenated audio.
    """
    silence = np.zeros(int(silence_duration * sample_rate), dtype=audios[0].dtype)
    segments: List[np.ndarray] = []
    for audio in audios:
        segments.append(audio)
        segments.append(silence)
    if segments:
        segments.pop()  # Remove the last silence
    return np.concatenate(segments)

def run(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main handler function for the multi-speaker TTS worker.

    This function validates the input JSON, synthesizes speech for each speaker from the
    provided list using RUSynth (with long_text mode always enabled), concatenates the
    resulting audio segments with silence, optionally applies audio enhancement using the
    AudioEnhancer, and returns the resulting audio.

    Args:
        job: A dictionary containing the job input data.

    Returns:
        A dictionary containing the output audio (base64-encoded or URL).
    """
    job_input: Dict[str, Any] = job['input']

    # Validate the input using the schema
    validated: Dict[str, Any] = validate(job_input, INPUT_SCHEMA)
    if 'errors' in validated:
        return {"error": validated['errors']}
    params: Dict[str, Any] = validated['validated_input']

    # The "text" field is expected to be a list of pairs [speaker_id, text].
    text_list: List[Tuple[Any, str]] = params["text"]

    # Common TTS parameters (applied to all segments)
    speed: float = params.get("speed", 1.0)
    accentize: bool = params.get("accentize", True)
    volume: float = params.get("volume", 0.3)
    low_pass_filter_cutoff: Any = params.get("low_pass_filter_cutoff", None)
    # long_text mode is always enabled
    long_text: bool = True
    enhance_audio: bool = params.get("enhance_audio", False)
    output_format: str = params.get("output_format", "wav").lower()

    audio_segments: List[np.ndarray] = []
    sample_rate: int = None

    # Iterate over the list of pairs.
    for pair in text_list:
        try:
            # Each pair should contain speaker_id and text.
            speaker_id, text = pair
            speaker_id = int(speaker_id)
        except (ValueError, TypeError):
            continue  # Skip invalid entries

        # Synthesize audio for the current speaker using RUSynth with long_text mode enabled.
        segment_audio, sr = TTS_MODEL.synthesize_long(
            text=text,
            speaker_id=speaker_id,
            speed=speed,
            accentize=accentize,
            volume=volume,
            low_pass_filter_cutoff=low_pass_filter_cutoff,
            verbose=False
        )
        if sample_rate is None:
            sample_rate = sr
        elif sample_rate != sr:
            sample_rate = sr  # Ensure consistency (or perform resampling if necessary)
        audio_segments.append(segment_audio)

    # Concatenate all audio segments with a silence gap between them
    final_audio: np.ndarray = concatenate_audios(audio_segments, sample_rate, silence_duration=0.2)
    final_audio = final_audio.astype(np.int16)

    # Optionally, apply audio enhancement using AudioEnhancer with default parameters.
    if enhance_audio:
        final_audio = final_audio.astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(final_audio).float()
        # Default parameters: nfe=64, solver="midpoint", lambd=1.0, tau=0.5
        audio_tensor, sample_rate = AUDIO_ENHANCER(audio_tensor, sample_rate=sample_rate)
        audio_tensor= (audio_tensor * 32768.0).clamp(-32768.0, 32767.0)
        final_audio = audio_tensor.detach().cpu().numpy().astype(np.int16)

    key = f"{job['id']}"
    if output_format == "mp3":
        bytes_array = wav_to_mp3_bytes(final_audio, sample_rate)
        key += ".mp3"
    else:
        bytes_array = wav_to_bytes(final_audio, sample_rate)
        key += ".wav"
    audio_return: str = upload_bytes(bytes_array, key)

    job_output: Dict[str, Any] = {"audio": audio_return}

    # Clean up temporary files if necessary
    rp_cleanup.clean(['input_objects'])
    return job_output

if __name__ == "__main__":
    runpod.serverless.start({"handler": run})
