import json
from pathlib import Path


class CharacterTokenizer:
    def __init__(self, chars=None):
        self.chars = list(chars or [])
        self._refresh()

    def _refresh(self):
        self.char_to_id = {char: index for index, char in enumerate(self.chars)}
        self.id_to_char = dict(enumerate(self.chars))

    def fit(self, text):
        self.chars = sorted(set(text))
        self._refresh()
        return self

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        unknown = sorted(set(text) - set(self.char_to_id))
        if unknown:
            raise ValueError(f"Unknown characters: {unknown!r}")
        return [self.char_to_id[char] for char in text]

    def decode(self, ids):
        return "".join(self.id_to_char[int(index)] for index in ids)

    def to_dict(self):
        return {"type": "character", "chars": self.chars}

    @classmethod
    def from_dict(cls, value):
        return cls(value["chars"])

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

