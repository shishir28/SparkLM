from pathlib import Path
import urllib.error

from sparklm.classification.spam_data import (
    create_balanced_dataset,
    download_and_unzip_spam_data,
    load_spam_tsv,
    map_spam_labels,
    random_split,
)


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "sms_spam_collection"
ZIP_PATH = ROOT / "data" / "raw" / "sms_spam_collection.zip"
DATA_FILE_PATH = RAW_DIR / "SMSSpamCollection.tsv"
PROCESSED_DIR = ROOT / "data" / "processed" / "sms_spam"
PRIMARY_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
FALLBACK_URL = (
    "https://f001.backblazeb2.com/file/"
    "LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
)
EXPECTED_RAW_COUNTS = {"ham": 4825, "spam": 747}


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    try:
        download_and_unzip_spam_data(
            url=PRIMARY_URL,
            zip_path=ZIP_PATH,
            extracted_file_path=RAW_DIR,
            extracted_data_file_path=DATA_FILE_PATH,
        )
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
        print(f"Primary URL failed: {error}. Trying backup URL...")
        download_and_unzip_spam_data(
            url=FALLBACK_URL,
            zip_path=ZIP_PATH,
            extracted_file_path=RAW_DIR,
            extracted_data_file_path=DATA_FILE_PATH,
        )

    dataframe = load_spam_tsv(DATA_FILE_PATH)
    raw_counts = dataframe["Label"].value_counts().to_dict()
    if raw_counts != EXPECTED_RAW_COUNTS:
        raise ValueError(
            f"Expected raw counts {EXPECTED_RAW_COUNTS}, found {raw_counts}"
        )

    balanced_dataframe = map_spam_labels(create_balanced_dataset(dataframe))
    balanced_counts = balanced_dataframe["Label"].value_counts().to_dict()
    if balanced_counts != {0: 747, 1: 747}:
        raise ValueError(f"Unexpected balanced counts: {balanced_counts}")

    train_dataframe, validation_dataframe, test_dataframe = random_split(
        balanced_dataframe,
        train_frac=0.7,
        val_frac=0.1,
    )
    split_sizes = (
        len(train_dataframe),
        len(validation_dataframe),
        len(test_dataframe),
    )
    if split_sizes != (1045, 149, 300):
        raise ValueError(f"Unexpected split sizes: {split_sizes}")

    train_dataframe.to_csv(PROCESSED_DIR / "train.csv", index=False)
    validation_dataframe.to_csv(PROCESSED_DIR / "validation.csv", index=False)
    test_dataframe.to_csv(PROCESSED_DIR / "test.csv", index=False)

    print(f"Raw data: {DATA_FILE_PATH}")
    print(f"Processed data: {PROCESSED_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
