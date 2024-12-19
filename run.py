import os
import torch
import sys
from gpt import download_file_if_missing, parse_arguments, text_generator

if __name__ == '__main__':
    """
    Main script for downloading the GPT-2 model files (if missing), 
    parsing command-line arguments, loading the model, and generating text.
    """

    # Parse arguments provided by the user (e.g., input text, batch size, etc.)
    args = parse_arguments()

    model_size = args.param
    # Ensure the model size is valid

    # for weight refer https://huggingface.co/openai-community/
    # also refer the https://github.com/openai/gpt-2 repository for information on the parameters
    # Dictionary containing URLs for different model files (e.g., weight, encoder, vocab)
    weights_parameters = {
        "124M": {
            "weight_url": "https://huggingface.co/openai-community/gpt2/resolve/main/pytorch_model.bin",
            "encoder_url": "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json",
            "vocab_url": "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe",
        },
        "355M": {
            "weight_url": "https://huggingface.co/openai-community/gpt2-medium/resolve/main/pytorch_model.bin",
            "encoder_url": "https://openaipublic.blob.core.windows.net/gpt-2/models/355M/encoder.json",
            "vocab_url": "https://openaipublic.blob.core.windows.net/gpt-2/models/355M/vocab.bpe",
        },
        "774M": {
            "weight_url": "https://huggingface.co/openai-community/gpt2-large/resolve/main/pytorch_model.bin",
            "encoder_url": "https://openaipublic.blob.core.windows.net/gpt-2/models/774M/encoder.json",
            "vocab_url": "https://openaipublic.blob.core.windows.net/gpt-2/models/774M/vocab.bpe",
        },
        "1558M": {
            "weight_url": "https://huggingface.co/openai-community/gpt2-xl/resolve/main/pytorch_model.bin",
            "encoder_url": "https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/encoder.json",
            "vocab_url": "https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/vocab.bpe",
        }
    }

    # Ensure the model size is valid
    if model_size not in weights_parameters:
        print(f"Model size {model_size} is not supported.")
        sys.exit(1)

    # Get the URLs for the specified model size
    model_files = weights_parameters[model_size]

    # Ensure the model files are available
    download_file_if_missing(model_files["weight_url"], f"models/{model_size}/pytorch_model.bin")
    download_file_if_missing(model_files["encoder_url"], f"models/{model_size}/encoder.json")
    download_file_if_missing(model_files["vocab_url"], f"models/{model_size}/vocab.bpe")

    # Load the model's state dictionary from the specified file
    state_dict = torch.load(f"models/{model_size}/pytorch_model.bin", map_location='cpu' if not torch.cuda.is_available() else None, weights_only=True)

    # Generate text based on the loaded model and parsed arguments
    text_generator(state_dict, args)
