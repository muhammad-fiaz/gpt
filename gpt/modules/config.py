class GPT2Config(object):
    """
    A configuration class for the GPT-2 model.

    This class holds various hyperparameters used to configure the GPT-2 model architecture, including the 
    vocabulary size, number of positions, embedding dimensions, number of layers, and other parameters. 
    These parameters are essential for initializing and training the model.

    Attributes:
        vocab_size (int): The size of the vocabulary (number of tokens).
        n_ctx (int): The context size (length of input sequence).
        n_positions (int): The maximum number of positions (used for positional embeddings).
        n_embd (int): The size of the hidden states (embedding dimension).
        n_layer (int): The number of layers (depth of the model).
        n_head (int): The number of attention heads.
        layer_norm_epsilon (float): The epsilon value used in layer normalization.
        initializer_range (float): The standard deviation of the initializer for the model weights.

    Args:
        vocab_size_or_config_json_file (int or str, optional): If an integer, sets the vocabulary size (default is 50257).
            If a string, it is treated as the path to a configuration file.
        n_positions (int, optional): The maximum number of positions (default is 1024).
        n_ctx (int, optional): The context size (default is 1024).
        n_embd (int, optional): The embedding dimension (default is 768).
        n_layer (int, optional): The number of layers (default is 12).
        n_head (int, optional): The number of attention heads (default is 12).
        layer_norm_epsilon (float, optional): The epsilon value for layer normalization (default is 1e-5).
        initializer_range (float, optional): The standard deviation for weight initialization (default is 0.02).
    """

    def __init__(
            self,
            vocab_size_or_config_json_file=50257,
            n_positions=1024,
            n_ctx=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
    ):
        """
        Initializes a GPT2Config object with the given parameters.

        Args:
            vocab_size_or_config_json_file (int or str, optional): The vocabulary size or the path to a config file (default 50257).
            n_positions (int, optional): The maximum number of positions (default 1024).
            n_ctx (int, optional): The context size (default 1024).
            n_embd (int, optional): The embedding dimension (default 768).
            n_layer (int, optional): The number of layers (default 12).
            n_head (int, optional): The number of attention heads (default 12).
            layer_norm_epsilon (float, optional): The epsilon value for layer normalization (default 1e-5).
            initializer_range (float, optional): The standard deviation for weight initialization (default 0.02).
        """
        self.vocab_size = vocab_size_or_config_json_file
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
