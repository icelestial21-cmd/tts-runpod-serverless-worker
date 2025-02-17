import abc
import json
import pathlib
import numpy as np
import onnxruntime as rt


class EngineBase(abc.ABC):
    @abc.abstractmethod
    def load(self, model_path):
        pass

    @abc.abstractmethod
    def run(self, text, text_lengths, scales, sid=None):
        pass


class EngineORT(EngineBase):
    def load(self, model_path):
        self.sess = rt.InferenceSession(pathlib.Path(model_path) / "model.onnx")

    def run(self, text, text_lengths, scales, sid=None):
        return self.sess.run(None, {
            'input': text,
            'input_lengths': text_lengths,
            'scales': scales,
            'sid': sid
        })[0]
