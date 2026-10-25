import io
from pathlib import Path
import urllib.error
import zipfile

import pandas as pd
import pytest

from sparklm.classification.spam_data import (
    create_balanced_dataset,
    download_and_unzip_spam_data,
    load_spam_tsv,
    map_spam_labels,
    random_split,
)


def sample_dataframe():
    return pd.DataFrame(
        {
            "Id": range(6),
            "Label": ["ham", "ham", "ham", "ham", "spam", "spam"],
            "Text": [f"message {index}" for index in range(6)],
        }
    )


def zip_payload(filename="SMSSpamCollection", content=b"ham\thello\n"):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zip_file:
        zip_file.writestr(filename, content)
    return buffer.getvalue()


def test_balancing_is_equal_deterministic_and_keeps_every_spam_row():
    dataframe = sample_dataframe()

    first = create_balanced_dataset(dataframe)
    second = create_balanced_dataset(dataframe)

    assert first["Label"].value_counts().to_dict() == {"ham": 2, "spam": 2}
    assert set(first[first["Label"] == "spam"]["Id"]) == {4, 5}
    pd.testing.assert_frame_equal(first, second)


def test_map_spam_labels_produces_only_zero_and_one():
    mapped = map_spam_labels(create_balanced_dataset(sample_dataframe()))

    assert set(mapped["Label"]) == {0, 1}
    assert pd.api.types.is_integer_dtype(mapped["Label"])


def test_map_spam_labels_rejects_unexpected_labels():
    dataframe = pd.DataFrame({"Label": ["ham", "other"], "Text": ["a", "b"]})

    with pytest.raises(ValueError, match="Expected labels"):
        map_spam_labels(dataframe)


def test_random_split_is_deterministic_disjoint_and_complete():
    dataframe = pd.DataFrame({"Id": range(10)})

    first = random_split(dataframe, train_frac=0.6, val_frac=0.2)
    second = random_split(dataframe, train_frac=0.6, val_frac=0.2)

    assert [len(split) for split in first] == [6, 2, 2]
    for first_split, second_split in zip(first, second):
        pd.testing.assert_frame_equal(first_split, second_split)
    split_ids = [set(split["Id"]) for split in first]
    assert split_ids[0].isdisjoint(split_ids[1])
    assert split_ids[0].isdisjoint(split_ids[2])
    assert split_ids[1].isdisjoint(split_ids[2])
    assert set().union(*split_ids) == set(range(10))


@pytest.mark.parametrize(
    ("train_frac", "val_frac"),
    [(-0.1, 0.2), (0.7, -0.1), (0.8, 0.2)],
)
def test_random_split_rejects_invalid_fractions(train_frac, val_frac):
    with pytest.raises(ValueError, match="sum to less than 1"):
        random_split(pd.DataFrame({"Id": range(3)}), train_frac, val_frac)


def test_load_spam_tsv_validates_rows_and_labels(tmp_path):
    path = tmp_path / "spam.tsv"
    path.write_text("ham\thello\nspam\twin now\n", encoding="utf-8")

    dataframe = load_spam_tsv(path, expected_rows=2)

    assert dataframe.to_dict("records") == [
        {"Label": "ham", "Text": "hello"},
        {"Label": "spam", "Text": "win now"},
    ]


def test_download_skips_when_target_exists(tmp_path, monkeypatch):
    target = tmp_path / "extracted" / "SMSSpamCollection.tsv"
    target.parent.mkdir()
    target.write_text("existing", encoding="utf-8")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("network should not be called")

    monkeypatch.setattr("urllib.request.urlopen", fail_if_called)
    download_and_unzip_spam_data(
        "https://example.invalid/spam.zip",
        tmp_path / "spam.zip",
        target.parent,
        target,
    )

    assert target.read_text(encoding="utf-8") == "existing"


def test_download_extracts_and_renames_dataset(tmp_path, monkeypatch):
    payload = zip_payload()
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda url, timeout: io.BytesIO(payload),
    )
    target = tmp_path / "raw" / "SMSSpamCollection.tsv"

    download_and_unzip_spam_data(
        "https://example.test/spam.zip",
        tmp_path / "downloads" / "spam.zip",
        target.parent,
        target,
    )

    assert target.read_bytes() == b"ham\thello\n"
    assert (tmp_path / "downloads" / "spam.zip").is_file()


def test_download_rejects_unsafe_zip_entries(tmp_path, monkeypatch):
    payload = zip_payload("../outside.txt", b"unsafe")
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda url, timeout: io.BytesIO(payload),
    )

    with pytest.raises(ValueError, match="Unsafe zip entry"):
        download_and_unzip_spam_data(
            "https://example.test/spam.zip",
            tmp_path / "spam.zip",
            tmp_path / "raw",
            tmp_path / "raw" / "SMSSpamCollection.tsv",
        )
    assert not (tmp_path / "outside.txt").exists()


def test_download_propagates_network_errors_for_caller_fallback(
    tmp_path,
    monkeypatch,
):
    def raise_network_error(*args, **kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("urllib.request.urlopen", raise_network_error)

    with pytest.raises(urllib.error.URLError):
        download_and_unzip_spam_data(
            "https://example.invalid/spam.zip",
            tmp_path / "spam.zip",
            tmp_path / "raw",
            tmp_path / "raw" / "SMSSpamCollection.tsv",
        )
    assert not (tmp_path / "spam.zip.part").exists()
