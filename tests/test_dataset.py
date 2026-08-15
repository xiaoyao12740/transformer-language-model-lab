from src.dataset import AutoregressiveDataset, contiguous_split


def test_dataset_shift():
    x, y = AutoregressiveDataset(list(range(8)), 4)[2]
    assert x.tolist() == [2, 3, 4, 5]
    assert y.tolist() == [3, 4, 5, 6]


def test_contiguous_split_and_no_cross_boundary_window():
    text = "".join(chr(0x400 + i) for i in range(100))
    splits = contiguous_split(text, 0.8, 0.1)
    assert "".join(splits.values()) == text
    assert set(splits["train"]).isdisjoint(splits["val"])
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert len(AutoregressiveDataset(list(range(len(splits["train"]))), 4)) == 76


def test_invalid_split_rejected():
    try:
        contiguous_split("abc", 0.9, 0.2)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

