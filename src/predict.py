import os
import numpy as np
# torch
import torch
# xtts
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from audio_enhancer import AudioEnhancer

use_cuda = os.environ.get('WORKER_USE_CUDA', 'True').lower() == 'true'

class Predictor:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.is_setup = False
        self.config = None
        self.model = None
        self.audio_enhancer = None

    def setup(self):
        # We now lazy-load the models inside predict() so the container boots instantly
        # and RunPod's health checker doesn't kill us for taking too long to start.
        pass

    def _lazy_load_models(self):
        if self.is_setup:
            return

        from huggingface_hub import snapshot_download
        
        # Download models
        if not os.path.exists(os.path.join(self.model_dir, "xttsv2", "config.json")):
            print("Downloading XTTSv2 model weights...", flush=True)
            snapshot_download(repo_id="coqui/XTTS-v2", local_dir=os.path.join(self.model_dir, "xttsv2"))
            
        if not os.path.exists(os.path.join(self.model_dir, "audio_enhancer", "enhancer_stage2")):
            print("Downloading Resemble-Enhance model weights...", flush=True)
            snapshot_download(repo_id="ResembleAI/resemble-enhance", local_dir=os.path.join(self.model_dir, "audio_enhancer"))

        print("Loading XTTSv2 into VRAM...", flush=True)
        self.config = XttsConfig()
        self.config.load_json(os.path.join(self.model_dir, "xttsv2", "config.json"))
        self.model = Xtts.init_from_config(self.config)
        
        # PyTorch 2.6+ defaults to weights_only=True, which breaks Coqui-TTS unpickling.
        # We monkeypatch torch.load to bypass this breaking change safely.
        original_load = torch.load
        def _patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        
        torch.load = _patched_load
        try:
            self.model.load_checkpoint(
                self.config,
                checkpoint_dir=os.path.join(self.model_dir, "xttsv2"),
                use_deepspeed=True,
                eval=True
            )
        finally:
            torch.load = original_load

        if use_cuda:
            self.model.cuda()

        print("Loading Audio Enhancer into VRAM...", flush=True)
        self.audio_enhancer = AudioEnhancer.from_pretrained(
            os.path.join(self.model_dir, "audio_enhancer", "enhancer_stage2"),
            "cuda" if use_cuda else "cpu"
        )
        
        self.is_setup = True
        print("Models successfully loaded!", flush=True)

    @torch.inference_mode()
    def predict(
            self,
            text: list,
            speaker_wav: dict,
            gpt_cond_len: int,
            max_ref_len: int,
            language: str,
            speed: float,
            enhance_audio: bool
    ):
        self._lazy_load_models()
        silence = np.zeros(int(0.10 * 24000))
        wave, sr = None, None
        for line in text:
            voice = speaker_wav[line[0]]
            outputs = self.model.synthesize(
                line[1],
                self.config,
                speaker_wav=voice,
                gpt_cond_len=gpt_cond_len,
                language=language,
                enable_text_splitting=True,
                max_ref_len=max_ref_len,
                speed=speed
            )
            _wave, _sr = outputs['wav'], 24000
            if wave is None:
                wave = _wave
                sr = _sr
            else:
                wave = torch.cat([wave, silence.copy(), _wave], dim=1)
        if enhance_audio:
            wave, sr = self.audio_enhancer(
                torch.from_numpy(wave),
                sr
            )
            wave = wave.detach().cpu().numpy()
        return wave, sr

    @torch.inference_mode()
    def predict_stream(
            self,
            text: list,
            speaker_wav: dict,
            gpt_cond_len: int,
            max_ref_len: int,
            language: str,
            speed: float
    ):
        self._lazy_load_models()
        
        for line in text:
            voice = speaker_wav[line[0]]
            raw_text = line[1]
            
            chunks = self.model.inference_stream(
                raw_text,
                language=language,
                speaker_wav=voice,
                gpt_cond_len=gpt_cond_len,
                max_ref_len=max_ref_len,
                speed=speed,
                enable_text_splitting=True
            )
            
            for chunk in chunks:
                wave = chunk.detach().cpu().numpy()
                yield wave, 24000
