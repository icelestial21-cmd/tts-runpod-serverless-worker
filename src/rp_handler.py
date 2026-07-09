import io
import os
import argparse
import base64
# runpod utils
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils.rp_upload import upload_in_memory_object
from runpod.serverless.utils import rp_download, rp_cleanup
# predictor
import predict
from rp_schema import INPUT_SCHEMA
# utils
from scipy.io.wavfile import write


# Model params
model_dir = os.getenv("WORKER_MODEL_DIR", "/model")


def upload_audio(wav, sample_rate, key):
    """ Uploads audio to S3 bucket if it is available, otherwise returns base64 encoded audio. """
    # Convert wav to bytes
    wav_io = io.BytesIO()
    write(wav_io, sample_rate, wav)

    # Upload to S3
    if os.environ.get('BUCKET_ENDPOINT_URL', False):
        return upload_in_memory_object(
            key,
            wav_io.read(),
            bucket_creds = {
                "endpointUrl": os.environ.get('BUCKET_ENDPOINT_URL', None),
                "accessId": os.environ.get('BUCKET_ACCESS_KEY_ID', None),
                "accessSecret": os.environ.get('BUCKET_SECRET_ACCESS_KEY', None)
            }
        )
    # Base64 encode
    return base64.b64encode(wav_io.getvalue()).decode('utf-8')


import numpy as np

def run(job):
    job_input = job['input']

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        yield {"error": validated_input['errors']}
        return
    
    validated_input = validated_input['validated_input']

    # Download input objects
    for k, v in validated_input["voice"].items():
        validated_input["voice"][k] = rp_download.download_files_from_urls(
            job['id'],
            [v]
        )

    is_stream = validated_input.get("stream", False)

    if is_stream:
        # Streaming Inference text-to-audio
        chunks = MODEL.predict_stream(
            language=validated_input["language"],
            speaker_wav=validated_input["voice"],
            text=validated_input["text"],
            gpt_cond_len=validated_input.get("gpt_cond_len", 7),
            max_ref_len=validated_input.get("max_ref_len", 10),
            speed=validated_input.get("speed", 1.0)
        )
        
        for wave_chunk, sr in chunks:
            # Convert float32 wave chunk to PCM 16-bit
            pcm16_chunk = (wave_chunk * 32767).astype(np.int16).tobytes()
            chunk_b64 = base64.b64encode(pcm16_chunk).decode('utf-8')
            yield {
                "audio_chunk": chunk_b64,
                "sample_rate": sr,
                "is_final": False
            }
            
        yield {"is_final": True}
        
    else:
        # Blocking Inference text-to-audio
        wave, sr = MODEL.predict(
            language=validated_input["language"],
            speaker_wav=validated_input["voice"],
            text=validated_input["text"],
            gpt_cond_len=validated_input.get("gpt_cond_len", 7),
            max_ref_len=validated_input.get("max_ref_len", 10),
            speed=validated_input.get("speed", 1.0),
            enhance_audio=validated_input.get("enhance_audio", True)
        )

        # Upload output object
        audio_return = upload_audio(wave, sr, f"{job['id']}.wav")
        job_output = {
            "audio": audio_return
        }
        
        yield job_output

    # Remove downloaded input objects
    rp_cleanup.clean(['input_objects'])


if __name__ == "__main__":
    MODEL = predict.Predictor(model_dir=model_dir)
    MODEL.setup()

    runpod.serverless.start({"handler": run})
