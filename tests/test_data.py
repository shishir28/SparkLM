import pytest


from sparklm.data import make_batch, make_example


def test_make_example_shifts_target_by_one():
    ids = [1, 0, 2, 0, 2, 0]

    x, y = make_example(ids, start=0, block_size=4)

    assert x == [1, 0, 2, 0]
    assert y == [0, 2, 0, 2]


def test_make_example_with_different_start():
    ids = [1, 0, 2, 0, 2, 0]

    x, y = make_example(ids, start=1, block_size=3)

    assert x == [0, 2, 0]
    assert y == [2, 0, 2]


def test_make_example_with_insufficient_tokens():
    ids = [1, 0, 2, 0, 2, 0]

    with pytest.raises(ValueError, match="Not enough tokens"):
        make_example(ids, start=3, block_size=4)


def test_make_example_with_negative_start():
    ids = [1, 0, 2, 0, 2, 0]

    with pytest.raises(ValueError, match="start.*non-negative|Start.*non-negative"):
        make_example(ids, start=-1, block_size=3)


def test_make_example_with_non_positive_block_size():
    ids = [1, 0, 2, 0, 2, 0]

    with pytest.raises(ValueError, match="Block size.*positive"):
        make_example(ids, start=0, block_size=0)

def test_make_batch():
    ids = [1, 0, 2, 0, 2, 0]
    starts = [0, 1]
    block_size = 3

    x_batch, y_batch = make_batch(ids, starts, block_size)

    assert x_batch.tolist() == [[1, 0, 2], [0, 2, 0]]
    assert y_batch.tolist() == [[0, 2, 0], [2, 0, 2]]