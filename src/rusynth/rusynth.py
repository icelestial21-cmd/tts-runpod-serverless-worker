import json
import pathlib
import numpy as np
from numpy.typing import NDArray
# accentizer
from ruaccent import RUAccent
# sentence splitter
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
# progress bar
from tqdm.auto import tqdm
# model downloader
from huggingface_hub import snapshot_download
# inference engine
from .engine import EngineORT
# utils
from .utils import (
    Tokenizer,
    intersperse,
    TextSimplifier,
    volume_to_dbfs,
    normalize_audio,
    low_pass_filter
)


class RUSynth:
    def __init__(self, model_path: pathlib.Path or str):
        if isinstance(model_path, str):
            model_path = pathlib.Path(model_path)

        # config
        self.config = json.loads((model_path / 'config.json').read_text())

        # inference engine
        self.engine = EngineORT()
        self.engine.load(model_path)

        # tokenizer
        self.tokenizer = Tokenizer(model_path)

        # accentizer
        self.accentizer = RUAccent()
        self.accentizer.load(
            omograph_model_size='turbo2',
            use_dictionary=True,
            tiny_mode=False
        )

        # text simplifier
        self.simplifier = TextSimplifier()

    @staticmethod
    def download(
            repo_id: str = 'bes-dev/rusynth',
            local_dir: pathlib.Path or str = '/tmp/'
    ):
        return snapshot_download(repo_id, local_dir = local_dir)

    @classmethod
    def from_pretrained(
            cls,
            repo_id: str = 'bes-dev/rusynth',
            local_dir: pathlib.Path or str = '/tmp/'
    ):
        model_path = cls.download(repo_id, local_dir)
        return cls(model_path)

    def tokenize(self, text: str, accentize: bool = True) -> NDArray:
        text = self.simplifier(text)
        if accentize:
            text = self.accentizer.process_all(text)
        text = self.tokenizer.tokenize(text)
        if self.config['add_blank']:
            text = intersperse(text, 0)
        return np.array(text, dtype=np.int64)

    def inference(self, text: str, speaker_id: int, speed: float = 1.0) -> NDArray:
        text = self.tokenize(text)
        text_lengths = np.array([text.shape[0]], dtype=np.int64)
        scales = np.array([
            self.config['scale_noise'],
            self.config['scale_noise'] / speed,
            self.config['scale_noise_w']
        ], dtype=np.float32)
        sid = np.array([speaker_id], dtype=np.int64)
        audio = self.engine.run(np.expand_dims(text, axis=0), text_lengths, scales, sid)
        audio *= self.config['volume_scale']
        return audio.astype(np.int16), self.config['sampling_rate']

    def synthesize(
            self,
            text: str,
            speaker_id: int,
            speed: float = 1.0,
            accentize: bool = True,
            volume: float = 0.3,
            low_pass_filter_cutoff: int = 3000
    ) -> NDArray:
        audio, sr = self.inference(text, speaker_id, speed)
        audio = normalize_audio(
            audio,
            sampling_rate = sr,
            channels = 1,
            target_dBFS = volume_to_dbfs(volume)
        )
        if low_pass_filter:
            audio = low_pass_filter(audio, sr, 1, low_pass_filter_cutoff)
        return audio, sr

    def synthesize_long(
            self,
            text: str,
            speaker_id: int,
            speed: float = 1.0,
            silence_duration: float = 0.2,
            accentize: bool = True,
            volume: float = 0.3,
            low_pass_filter_cutoff: int = 3000,
            verbose: bool = False
    ) -> NDArray:
        # progress bar
        if verbose:
            pb = tqdm
        else:
            pb = lambda x: x

        # prepare silence
        sampling_rate = self.config['sampling_rate']
        silence = np.zeros(int(silence_duration * sampling_rate))

        # split text into sentences
        sentences = nltk.sent_tokenize(text)

        output = []
        for sentence in pb(sentences):
            audio, sr = self.synthesize(sentence, speaker_id, speed, accentize, volume, low_pass_filter_cutoff)
            output.extend([audio, silence])

        output = np.concatenate(output)
        return output, sampling_rate

    def __call__(
            self,
            text: str,
            speaker_id: int,
            speed: float = 1.0,
            accentize: bool = True,
            volume: float = 0.3,
            low_pass_filter_cutoff: int = None,
            long_text: bool = False,
            verbose: bool = False
    ) -> NDArray:
        if long_text:
            return self.synthesize_long(
                text = text,
                speaker_id = speaker_id,
                speed = speed,
                accentize = accentize,
                volume = volume,
                low_pass_filter_cutoff = low_pass_filter_cutoff,
                verbose = verbose
            )
        return self.synthesize(
            text = text,
            speaker_id = speaker_id,
            speed = speed,
            accentize = accentize,
            volume = volume,
            low_pass_filter_cutoff = low_pass_filter_cutoff
        )
