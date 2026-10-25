from pathlib import Path
import shutil
import urllib.request
import zipfile

import pandas as pd


EXPECTED_LABELS = {"ham", "spam"}


def download_and_unzip_spam_data(
    url,
    zip_path,
    extracted_file_path,
    extracted_data_file_path,
):
    zip_path = Path(zip_path)
    extracted_file_path = Path(extracted_file_path)
    extracted_data_file_path = Path(extracted_data_file_path)

    if extracted_data_file_path.exists():
        print(
            f"Data file {extracted_data_file_path} already exists. "
            "Skipping download."
        )
        return

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    extracted_file_path.mkdir(parents=True, exist_ok=True)
    extracted_data_file_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = zip_path.with_suffix(f"{zip_path.suffix}.part")

    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            with partial_path.open("wb") as output_file:
                shutil.copyfileobj(response, output_file)
        partial_path.replace(zip_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise

    extraction_root = extracted_file_path.resolve()
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        for member in zip_file.infolist():
            member_path = (extraction_root / member.filename).resolve()
            if not member_path.is_relative_to(extraction_root):
                raise ValueError(f"Unsafe zip entry: {member.filename}")
        zip_file.extractall(extracted_file_path)

    original_file_path = extracted_file_path / "SMSSpamCollection"
    if not original_file_path.is_file():
        raise FileNotFoundError(
            f"Expected extracted file was not found: {original_file_path}"
        )
    original_file_path.rename(extracted_data_file_path)
    print(f"Downloaded and extracted data to {extracted_data_file_path}")


def load_spam_tsv(path, expected_rows=5572):
    dataframe = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["Label", "Text"],
    )
    if expected_rows is not None and len(dataframe) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows, found {len(dataframe)}"
        )

    observed_labels = set(dataframe["Label"].unique())
    if observed_labels != EXPECTED_LABELS:
        raise ValueError(
            f"Expected labels {EXPECTED_LABELS}, found {observed_labels}"
        )
    return dataframe


def create_balanced_dataset(dataframe):
    num_spam = (dataframe["Label"] == "spam").sum()
    ham_rows = dataframe[dataframe["Label"] == "ham"]
    spam_rows = dataframe[dataframe["Label"] == "spam"]
    if num_spam == 0:
        raise ValueError("Dataset contains no spam rows")
    if len(ham_rows) < num_spam:
        raise ValueError("Dataset does not contain enough ham rows to balance")

    ham_subset = ham_rows.sample(n=num_spam, random_state=123)
    return pd.concat([ham_subset, spam_rows]).copy()


def map_spam_labels(dataframe):
    observed_labels = set(dataframe["Label"].unique())
    if observed_labels != EXPECTED_LABELS:
        raise ValueError(
            f"Expected labels {EXPECTED_LABELS}, found {observed_labels}"
        )

    mapped = dataframe.copy()
    mapped["Label"] = mapped["Label"].map({"ham": 0, "spam": 1})
    return mapped


def random_split(dataframe, train_frac, val_frac):
    if train_frac < 0 or val_frac < 0 or train_frac + val_frac >= 1:
        raise ValueError(
            "train_frac and val_frac must be nonnegative and sum to less than 1"
        )

    shuffled = dataframe.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(shuffled) * train_frac)
    val_end = train_end + int(len(shuffled) * val_frac)
    return (
        shuffled.iloc[:train_end].copy(),
        shuffled.iloc[train_end:val_end].copy(),
        shuffled.iloc[val_end:].copy(),
    )
