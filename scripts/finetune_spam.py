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
from sparklm.config import GPT2_MODEL_CONFIGS
from sparklm.generation import generate_text_simple
from sparklm.gpt2_weights import load_weights_into_gpt
from sparklm.model import GPTModel
from sparklm.text_utils import text_to_token_ids, token_ids_to_text
from sparklm.tokenizers.bpe_tokenizer import BPETokenizer


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "sms_spam_collection"
ZIP_PATH = ROOT / "data" / "raw" / "sms_spam_collection.zip"
DATA_FILE_PATH = RAW_DIR / "SMSSpamCollection.tsv"
PROCESSED_DIR = ROOT / "data" / "processed" / "sms_spam"
TRAIN_CSV = PROCESSED_DIR / "train.csv"
VALIDATION_CSV = PROCESSED_DIR / "validation.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"
GPT2_MODELS_DIR = ROOT / "gpt2"
PRIMARY_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
FALLBACK_URL = (
    "https://f001.backblazeb2.com/file/"
    "LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
)
CHOOSE_MODEL = "gpt2-small (124M)"
CLASSIFICATION_PROMPT = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.'"
)
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
    **GPT2_MODEL_CONFIGS[CHOOSE_MODEL],
}
EXPECTED_RAW_COUNTS = {"ham": 4825, "spam": 747}
EXPECTED_SPLIT_SIZES = (1045, 149, 300)
EXPECTED_LOADER_COUNTS = (130, 19, 38)


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


def initialize_pretrained_gpt2(tokenizer):
    from sparklm.gpt_download import download_and_load_gpt2

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    _, params = download_and_load_gpt2(model_size, GPT2_MODELS_DIR)

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()

    prompts = (
        ("Every effort moves you", 15),
        (CLASSIFICATION_PROMPT, 23),
    )
    for prompt, max_new_tokens in prompts:
        token_ids = generate_text_simple(
            model,
            idx=text_to_token_ids(prompt, tokenizer),
            max_new_tokens=max_new_tokens,
            context_size=BASE_CONFIG["context_length"],
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        print(f"Generated text for '{prompt}': {generated_text}")
    return model


def main() -> int:
    csv_paths = prepare_spam_csv_files()
    tokenizer = BPETokenizer("gpt2")
    datasets = create_spam_datasets(
        *csv_paths,
        tokenizer=tokenizer,
        context_length=BASE_CONFIG["context_length"],
    )
    if datasets[0].max_length != 120:
        raise ValueError(
            f"Expected training max_length 120, found {datasets[0].max_length}"
        )

    loaders = create_spam_data_loaders(*datasets)
    loader_counts = tuple(len(loader) for loader in loaders)
    if loader_counts != EXPECTED_LOADER_COUNTS:
        raise ValueError(f"Unexpected loader counts: {loader_counts}")

    initialize_pretrained_gpt2(tokenizer)

    print(f"Raw data: {DATA_FILE_PATH}")
    print(f"Processed data: {PROCESSED_DIR}")
    print(f"Dataset sizes: {tuple(len(dataset) for dataset in datasets)}")
    print(f"Maximum sequence length: {datasets[0].max_length}")
    print(f"Loader counts: {loader_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
