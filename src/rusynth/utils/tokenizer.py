import re
import pathlib
from typing import List
from .ru_dictionary import convert
from .utils import load_dictionary
from .symbols import symbols


class Tokenizer:
    def __init__(self, model_path: pathlib.Path):
        self.dictionary = load_dictionary(model_path / "dictionary.txt")
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}

    def tokenize(self, text: str) -> List[int]:
        pattern = "([,.?!;:\"() ])"
        phonemes = ["^"]
        for word in re.split(pattern, text.lower()):
            if word == "":
                continue
            if re.match(pattern, word) or word == '-':
                phonemes.append(word)
            elif word in self.dictionary:
                phonemes.extend(self.dictionary[word])
            else:
                phonemes.extend(convert(word).split())
        phonemes.append("$")

        sequence = [
            self.symbol_to_id.get(symbol, self.symbol_to_id["_"])
            for symbol in phonemes
        ]
        # sequence = [self.symbol_to_id[symbol] for symbol in phonemes]
        # sequence = []
        # for symbol in phonemes:
        #     if symbol in self.symbol_to_id:
        #         sequence.append(self.symbol_to_id[symbol])
        #     else:
        #         sequence.append(self.symbol_to_id["_"])
        return sequence
