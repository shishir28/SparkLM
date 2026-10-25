import os
import json
import requests
# Triton (loaded lazily by torch._dynamo, e.g. in AdamW) segfaults if TensorFlow
# is imported first, so load torch._dynamo before TensorFlow.
import torch._dynamo  # noqa: F401
import tensorflow as tf
import numpy as np
from tqdm import tqdm

def download_and_load_gpt2(model_size, models_dir):
    tf.config.set_visible_devices([], "GPU")
    allowed_sizes = ["124M", "355M", "774M", "1558M"]
    if model_size not in allowed_sizes:
        raise ValueError(f"Invalid model size '{model_size}'. Allowed sizes: {allowed_sizes}")

    model_dir = os.path.join(models_dir, f"{model_size}")
    os.makedirs(model_dir, exist_ok=True)

    base_url = f"https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = ["checkpoint", # small text file naming the latest checkpoint (model.ckpt) but tf.train.latest_checkpoint reads it
                 "encoder.json", # holds token to Id vacabulary but titoken BPEtokenizer already has it
                 "hparams.json", # holds model hyperparameters (n_vocab, n_ctx, n_embd, n_head, n_layer) and is needed to build the model
                 "model.ckpt.data-00000-of-00001", # holds actual weight values (about 500 MB for 124M) but we don't need it
                 "model.ckpt.index", # table of contents for the weights file. For each tensor it records the name, shape and byte location, so list_variables and load_variable can find it.
                 "model.ckpt.meta", # holds the TensorFlow computation graph but we don't need it
                 "vocab.bpe"] # holds merge rule but titoken BPEtokenizer already has it

    for filename in filenames:
        url = f"{base_url}/{model_size}/{filename}"
        local_path = os.path.join(model_dir, filename)
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            continue
        download_file(url, local_path)


    tf_checkpoint_path = tf.train.latest_checkpoint(model_dir)
    if tf_checkpoint_path is None:
        raise FileNotFoundError(f"No TensorFlow checkpoint found in {model_dir}")

    settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r",encoding="utf-8"))
    params = load_gpt2_params_from_tf_checkpoint(tf_checkpoint_path, settings)
    return settings, params

def download_file(url, local_path):
    def _attempt_download(url):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))

        if (os.path.exists(local_path) and os.path.getsize(local_path) == total_size):
            return True

        block_size = 1024
        desc = os.path.basename(local_path)
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as progress_bar:
            with open(local_path, 'wb') as f:
                for data in response.iter_content(block_size):
                    f.write(data)
                    progress_bar.update(len(data))
        return True

    try:
        if _attempt_download(url):
            return
    except  requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        raise

def load_gpt2_params_from_tf_checkpoint(checkpoint_path, settings):
    # initialize an empty dictionary to hold the parameters
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    for var_name, _ in tf.train.list_variables(checkpoint_path):
        variable_array = np.squeeze(tf.train.load_variable(checkpoint_path, var_name))

        var_name_parts = var_name.split('/')[1:]  # remove the 'model/' prefix

        target_dict = params
        if var_name_parts[0].startswith("h"):
            layer_number = int(var_name_parts[0][1:])  # extract the block index
            target_dict = params["blocks"][layer_number]

        for key in var_name_parts[1:-1]:  # traverse the path except for the last part
            target_dict = target_dict.setdefault(key, {})

        last_key = var_name_parts[-1]
        target_dict[last_key] = variable_array

    return params
