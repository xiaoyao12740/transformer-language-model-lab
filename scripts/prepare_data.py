import hashlib
import json
import urllib.request
from pathlib import Path

from src.tokenizer import CharacterTokenizer

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main():
    root = Path(__file__).resolve().parents[1]
    raw = urllib.request.urlopen(URL, timeout=30).read().decode("utf-8")
    text = raw.replace("\r\n", "\n")
    path = root / "data/formal_corpus.txt"
    path.write_text(text, encoding="utf-8", newline="\n")
    sha = hashlib.sha256(text.encode()).hexdigest()
    provenance = {"source_url": URL, "work": "The Complete Works of William Shakespeare", "status": "Shakespeare's underlying works are public domain; source transcription attributed by URL", "normalization": "LF line endings; complete source file", "sha256": sha, "character_count": len(text), "vocabulary_size": CharacterTokenizer().fit(text).vocab_size}
    (root / "data/provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
