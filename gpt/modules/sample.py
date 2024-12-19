import torch
import torch.nn.functional as F
from tqdm import trange
from gpt.modules.encoder import get_encoder

def top_k_logits(logits, k):
    """
    Filters the logits to retain only the top-k highest values.

    This function ensures that only the top-k logits are considered, with the rest replaced by a very small
    value (`-1e10`) to make them effectively ignored during sampling. It is used to limit the number of
    possible next tokens when sampling from the model's output.

    Args:
        logits (torch.Tensor): The raw logits (unnormalized predictions) from the model for each token in the vocabulary.
        k (int): The number of top logits to retain. All logits outside the top-k are set to a very small value.

    Returns:
        torch.Tensor: The filtered logits, with only the top-k logits preserved and the rest masked out.
    """
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)

def sample_sequence(model, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, device='cuda', sample=True):
    """
    Generates a sequence of tokens from the model by sampling or selecting based on top-k logits.

    This function generates a sequence of tokens starting from a provided context or start token. It uses the
    model to predict the next token iteratively, either sampling from the logits or selecting the most probable token.

    Args:
        model (torch.nn.Module): The model to generate tokens from.
        length (int): The length of the sequence to generate.
        start_token (int, optional): The token to start the generation from. If None, `context` must be specified.
        batch_size (int, optional): The number of sequences to generate in parallel.
        context (list or torch.Tensor, optional): A sequence of token ids to condition the generation on.
            It is required if `start_token` is not provided.
        temperature (float, optional): A scaling factor for the logits before applying softmax. Higher values make the model more "creative".
        top_k (int, optional): The number of top logits to retain when sampling. Set to 0 to disable.
        device (str, optional): The device to run the model on (e.g., 'cuda' or 'cpu').
        sample (bool, optional): If True, samples the next token; otherwise, selects the most probable token.

    Returns:
        torch.Tensor: The generated sequence of tokens, including the initial context.
    """
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
    prev = context
    output = context
    past = None
    with torch.no_grad():
        for i in trange(length):
            logits, past = model(prev, past=past)
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            output = torch.cat((output, prev), dim=1)
    return output

def generate_from_prompt(model, prompt, length, temperature=1.0, top_k=0, device='cuda'):
    """
    Generates text from a given prompt by encoding the prompt and then generating additional tokens using the model.

    This function encodes the provided prompt using a byte pair encoding (BPE) encoder, generates a sequence
    of tokens using the model, and then decodes the generated tokens back into text.

    Args:
        model (torch.nn.Module): The model to generate text from.
        prompt (str): The initial text prompt to condition the generation on.
        length (int): The number of tokens to generate after the prompt.
        temperature (float, optional): A scaling factor for the logits before applying softmax.
        top_k (int, optional): The number of top logits to retain when sampling. Set to 0 to disable.
        device (str, optional): The device to run the model on (e.g., 'cuda' or 'cpu').

    Returns:
        str: The generated text, including the original prompt and the newly generated tokens.
    """
    # Initialize the encoder
    encoder = get_encoder()  # Assumes a `get_encoder()` function initializes the BPE utilities

    # Encode the prompt to tokens
    context_tokens = encoder.encode(prompt)

    # Generate tokens
    generated_tokens = sample_sequence(
        model=model,
        length=length,
        context=context_tokens,
        temperature=temperature,
        top_k=top_k,
        device=device
    )

    # Decode the tokens back to text
    generated_text = encoder.decode(generated_tokens[0].tolist())
    return generated_text
