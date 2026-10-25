import urllib.error
from pathlib import Path

from sparklm.classification.spam_data import (
    create_balanced_dataset,
    download_and_unzip_spam_data,
    load_spam_tsv,
    map_spam_labels,
    random_split,
)
from sparklm.classification.spam_dataset import (
    create_spam_data_loaders,
    create_spam_datasets,
)
from sparklm.tokenizers.bpe_tokenizer import BPETokenizer


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "sms_spam_collection"
ZIP_PATH = ROOT / "data" / "raw" / "sms_spam_collection.zip"
DATA_FILE_PATH = RAW_DIR / "SMSSpamCollection.tsv"
PROCESSED_DIR = ROOT / "data" / "processed" / "sms_spam"
TRAIN_CSV = PROCESSED_DIR / "train.csv"
VALIDATION_CSV = PROCESSED_DIR / "validation.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"
PRIMARY_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
FALLBACK_URL = (
    "https://f001.backblazeb2.com/file/"
    "LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
)
EXPECTED_RAW_COUNTS = {"ham": 4825, "spam": 747}
EXPECTED_SPLIT_SIZES = (1045, 149, 300)
EXPECTED_LOADER_COUNTS = (130, 19, 38)
GPT2_CONTEXT_LENGTH = 1024


def download_spam_data():
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


def prepare_spam_csv_files():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    download_spam_data()

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

    dataframes = random_split(
        balanced_dataframe,
        train_frac=0.7,
        val_frac=0.1,
    )
    split_sizes = tuple(len(split) for split in dataframes)
    if split_sizes != EXPECTED_SPLIT_SIZES:
        raise ValueError(f"Unexpected split sizes: {split_sizes}")

    csv_paths = (TRAIN_CSV, VALIDATION_CSV, TEST_CSV)
    for dataframe_split, csv_path in zip(dataframes, csv_paths):
        dataframe_split.to_csv(csv_path, index=False)
    return csv_paths


def main() -> int:
    csv_paths = prepare_spam_csv_files()
    tokenizer = BPETokenizer("gpt2")
    datasets = create_spam_datasets(
        *csv_paths,
        tokenizer=tokenizer,
        context_length=GPT2_CONTEXT_LENGTH,
    )
    if datasets[0].max_length != 120:
        raise ValueError(
            f"Expected training max_length 120, found {datasets[0].max_length}"
        )

    loaders = create_spam_data_loaders(*datasets)
    loader_counts = tuple(len(loader) for loader in loaders)
    if loader_counts != EXPECTED_LOADER_COUNTS:
        raise ValueError(f"Unexpected loader counts: {loader_counts}")

    print(f"Raw data: {DATA_FILE_PATH}")
    print(f"Processed data: {PROCESSED_DIR}")
    print(f"Dataset sizes: {tuple(len(dataset) for dataset in datasets)}")
    print(f"Maximum sequence length: {datasets[0].max_length}")
    print(f"Loader counts: {loader_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
