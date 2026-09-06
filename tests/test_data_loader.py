from sparklm.data_loader import create_dataloader_v1


def test_create_dataloader_v1_builds_valid_loader():
    text = "hello world hello world hello world"
    loader = create_dataloader_v1(
        text=text,
        batch_size=2,
        max_length=3,
        stride=2,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape[0] == 2
    assert batch_y.shape[0] == 2
    assert batch_x.shape[1] == 3
    assert batch_y.shape[1] == 3


def test_create_dataloader_v1_validates_inputs():
    try:
        create_dataloader_v1("abc", batch_size=0)
        assert False, "Expected ValueError for batch_size <= 0"
    except ValueError:
        pass

    try:
        create_dataloader_v1("abc", max_length=0)
        assert False, "Expected ValueError for max_length <= 0"
    except ValueError:
        pass
