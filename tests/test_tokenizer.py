from src.tokenizer import CharacterTokenizer


def test_tokenizer_roundtrip(tmp_path):
    tokenizer = CharacterTokenizer().fit("causal text\n")
    assert tokenizer.decode(tokenizer.encode("causal text\n")) == "causal text\n"
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    assert CharacterTokenizer.load(path).chars == tokenizer.chars


def test_unknown_character_rejected():
    tokenizer = CharacterTokenizer().fit("abc")
    try:
        tokenizer.encode("abcd")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

