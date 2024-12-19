import torch
import random
import numpy as np

from gpt.modules.encoder import get_encoder
from gpt.modules.model import GPT2LMHeadModel
from gpt.modules.utils import load_weight
from gpt.modules.config import GPT2Config
from gpt.modules.sample import sample_sequence


def text_generator(state_dict, args):
    """
    Generates text samples using the GPT-2 model based on provided arguments.

    This function sets up the model, processes the input context, generates text, and outputs the results.
    It allows for both conditional and unconditional text generation based on the provided arguments.

    Args:
        state_dict (dict): A dictionary containing the model's pre-trained weights.
        args (argparse.Namespace): The command-line arguments containing configuration for text generation.
            Expected arguments:
                - text (str): Input text to generate text from.
                - quiet (bool): If False, outputs the generated text to the console.
                - nsamples (int): Number of samples to generate.
                - batch_size (int): Batch size for generation.
                - length (int): Length of the generated text.
                - temperature (float): Temperature for sampling.
                - top_k (int): Number of top-k tokens to sample from.
                - unconditional (bool): If True, performs unconditional generation.

    Returns:
        None: Prints the generated text samples to the console.
    """
    if args.quiet is False:
        print(args)

    # Default batch size to 1 if not specified
    if args.batch_size == -1:
        args.batch_size = 1
    assert args.nsamples % args.batch_size == 0

    # Set random seed for reproducibility
    seed = random.randint(0, 2147483647)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Set device for model (cuda if available, otherwise cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the encoder and model configuration
    enc = get_encoder()
    config = GPT2Config()
    model = GPT2LMHeadModel(config)

    # Load the pre-trained model weights
    model = load_weight(model, state_dict)
    model.to(device)
    model.eval()

    # Set the default generation length if not specified
    if args.length == -1:
        args.length = config.n_ctx // 2
    elif args.length > config.n_ctx:
        raise ValueError(f"Can't generate samples longer than window size: {config.n_ctx}")

    # Print the input prompt text
    print(args.text)

    # Encode the input context text into tokens
    context_tokens = enc.encode(args.text)

    generated = 0
    for _ in range(args.nsamples // args.batch_size):
        # Generate text based on the given context and parameters
        out = sample_sequence(
            model=model, length=args.length,
            context=context_tokens if not args.unconditional else None,
            start_token=enc.encoder['<|endoftext|>'] if args.unconditional else None,
            batch_size=args.batch_size,
            temperature=args.temperature, top_k=args.top_k, device=device
        )

        # Trim the output to only include generated tokens beyond the input context
        out = out[:, len(context_tokens):].tolist()

        # Decode and print the generated text
        for i in range(args.batch_size):
            generated += 1
            text = enc.decode(out[i])
            if args.quiet is False:
                print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
            print(text)
