import argparse


def parse_arguments():
    """
    Parses command-line arguments for text generation using the GPT-2 model.

    This function uses the argparse library to handle various command-line options
    that control the text generation process. It supports specifying input text,
    output length, sampling temperature, number of samples to generate, batch size,
    and more. The returned arguments are used for controlling the model's behavior.

    Returns:
        argparse.Namespace: The parsed command-line arguments.

    Command-line arguments:
        --text (str): The input text that serves as the prompt for text generation.
        --quiet (bool): If True, silences the output. Default is False.
        --nsamples (int): The number of samples to generate. Default is 1.
        --unconditional (bool): If specified, the model generates text without context (unconditional generation).
        --batch_size (int): The batch size for generation. Default is -1, indicating no batching.
        --length (int): The length of the generated text. Default is -1, which means the model decides the length.
        --temperature (float): The sampling temperature that controls the randomness of predictions. Default is 0.7.
        --top_k (int): The number of top-k tokens to sample from for each prediction. Default is 40.
        --param (str): The model parameter to use (e.g., '124M', '355M', '774M', '1558M'). Default is '124M'.
    """
    parser = argparse.ArgumentParser(description="Text generation using GPT-2 model")

    # Command-line arguments for text generation
    parser.add_argument("--text", type=str, required=True, help="Input text to generate output")
    parser.add_argument("--quiet", type=bool, default=False, help="Silence the output")
    parser.add_argument("--nsamples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')
    parser.add_argument("--batch_size", type=int, default=-1, help="Batch size for generation")
    parser.add_argument("--length", type=int, default=-1, help="Length of generated text")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--top_k", type=int, default=40, help="Top k tokens to sample from")
    parser.add_argument("--param", type=str, default="124M", help="Custom parameter (e.g., model size)")

    return parser.parse_args()
