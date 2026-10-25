import time
import urllib.error
from pathlib import Path

import torch

from sparklm.classification.classifier_model import (
    BASE_CONFIG,
    add_classification_head,
    initialize_pretrained_gpt2,
    run_pretrained_sanity_checks,
)
from sparklm.classification.inference import classify_review
from sparklm.classification.metrics import calc_accuracy_loader, calc_loss_loader
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
from sparklm.classification.train_classifier import train_classifier_simple
from sparklm.device import move_model_to_device
from sparklm.plotting import plot_values
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
EXPECTED_RAW_COUNTS = {"ham": 4825, "spam": 747}
EXPECTED_SPLIT_SIZES = (1045, 149, 300)
EXPECTED_LOADER_COUNTS = (130, 19, 38)
NUM_EPOCHS = 5
RUNS_DIR = ROOT / "experiments" / "runs"
LOSS_PLOT_PATH = RUNS_DIR / "spam_classifier_loss.png"
ACCURACY_PLOT_PATH = RUNS_DIR / "spam_classifier_accuracy.png"
MODEL_PATH = RUNS_DIR / "spam_classifier_model.pt"
EXAMPLE_TEXTS = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.",
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!",
)


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


def create_spam_loaders(tokenizer):
    csv_paths = prepare_spam_csv_files()
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
    return loaders


def build_spam_classifier(tokenizer):
    model = initialize_pretrained_gpt2(GPT2_MODELS_DIR)
    run_pretrained_sanity_checks(model, tokenizer)
    model = add_classification_head(model, num_classes=2, trainable_last_blocks=1)
    device = move_model_to_device(model)
    print(f"Using device: {device}")
    return model, device


def print_baseline_metrics(model, train_loader, validation_loader, device):
    train_accuracy = calc_accuracy_loader(
        train_loader, model, device, num_batches=10
    )
    validation_accuracy = calc_accuracy_loader(
        validation_loader, model, device, num_batches=10
    )

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(validation_loader, model, device, num_batches=5)

    print(
        f"Initial Train Accuracy: {train_accuracy:.4f}, "
        f"Validation Accuracy: {validation_accuracy:.4f}, "
        f"Train Loss: {train_loss:.4f}, "
        f"Validation Loss: {val_loss:.4f}"
    )


def fine_tune(model, train_loader, validation_loader, device):
    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=0.1,
    )

    history = train_classifier_simple(
        model,
        train_dataloader=train_loader,
        val_dataloader=validation_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=NUM_EPOCHS,
        eval_freq=50,
        eval_iter=5,
    )
    end_time = time.time()
    print(
        f"Training completed in {end_time - start_time:.2f} seconds "
        f"on device {device}."
    )
    return history


def plot_history(train_values, val_values, examples_seen, label, path):
    plot_values(
        epochs_seen=torch.linspace(0, NUM_EPOCHS, len(train_values)),
        examples_seen=torch.linspace(0, examples_seen, len(train_values)),
        train_values=train_values,
        val_values=val_values,
        label=label,
        path=path,
    )


def save_training_plots(history):
    train_losses, val_losses, train_accs, val_accs, examples_seen = history
    plot_history(
        train_accs, val_accs, examples_seen, "accuracy", ACCURACY_PLOT_PATH
    )
    plot_history(train_losses, val_losses, examples_seen, "loss", LOSS_PLOT_PATH)


def print_final_accuracy(model, loaders, device):
    train_loader, validation_loader, test_loader = loaders
    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    validation_accuracy = calc_accuracy_loader(validation_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)
    print(
        f"Final Train Accuracy: {train_accuracy:.4f}, "
        f"Validation Accuracy: {validation_accuracy:.4f}, "
        f"Test Accuracy: {test_accuracy:.4f}"
    )


def print_example_predictions(model, tokenizer, device, max_length):
    for text in EXAMPLE_TEXTS:
        prediction = classify_review(
            text, model, tokenizer, device, max_length=max_length
        )
        print(f"Text: {text}\nClassified as: {prediction}")


def save_classifier(model):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved classifier to {MODEL_PATH}")
    return MODEL_PATH


def main() -> int:
    tokenizer = BPETokenizer("gpt2")
    loaders = create_spam_loaders(tokenizer)
    model, device = build_spam_classifier(tokenizer)

    # The test loader stays unused until the final evaluation after training.
    train_loader, validation_loader, _ = loaders

    print_baseline_metrics(model, train_loader, validation_loader, device)
    history = fine_tune(model, train_loader, validation_loader, device)
    save_training_plots(history)
    print_final_accuracy(model, loaders, device)

    max_length = train_loader.dataset.max_length
    print_example_predictions(model, tokenizer, device, max_length)
    save_classifier(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
