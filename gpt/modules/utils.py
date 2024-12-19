


def load_weight(model, state_dict):
    """
    Loads weights into the model from the provided state dictionary, mapping old weight names to new ones if necessary.

    This function handles renaming the keys in the state dictionary to match the expected names in the model, 
    such as changing `.g` to `.weight` and `.b` to `.bias`. It then loads the weights into the model while 
    handling any potential mismatched or missing keys.

    Args:
        model (torch.nn.Module): The model to load the weights into.
        state_dict (dict): A dictionary containing the model's weights.

    Returns:
        torch.nn.Module: The model with the weights loaded.
    """
    old_keys = []
    new_keys = []

    # Renaming keys in state_dict to match model's expected key names
    for key in state_dict.keys():
        new_key = None
        if key.endswith(".g"):
            new_key = key[:-2] + ".weight"
        elif key.endswith(".b"):
            new_key = key[:-2] + ".bias"
        elif key.endswith(".w"):
            new_key = key[:-2] + ".weight"
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)

    # Updating state_dict keys
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    # Copy state_dict so that _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=""):
        """
        Recursively loads the state_dict into each module of the model.

        Args:
            module (torch.nn.Module): The current module to load weights into.
            prefix (str, optional): The prefix for the module’s keys in the state_dict.
        """
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    start_model = model
    # Handling the transformer model (if present) separately
    if hasattr(model, "transformer") and all(not s.startswith('transformer.') for s in state_dict.keys()):
        start_model = model.transformer

    # Start loading weights into the model
    load(start_model, prefix="")

    # Make sure the input and output embeddings are still tied after loading weights
    model.set_tied()

    return model
