import copy
import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter

def gelu(x):
    """
    Applies the GELU (Gaussian Error Linear Unit) activation function.

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor: The result of applying the GELU activation function.
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class LayerNorm(nn.Module):
    """
    Layer normalization module that normalizes the input tensor across the last dimension.
    """
    def __init__(self, hidden_size, eps=1e-12):
        """
        Initialize the LayerNorm module.

        Args:
            hidden_size (int): The size of the hidden dimension.
            eps (float, optional): A small value added to the variance for numerical stability.
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        """
        Forward pass through the LayerNorm module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The normalized output tensor.
        """
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class Conv1D(nn.Module):
    """
    A 1D convolutional layer implemented as a linear transformation.
    """
    def __init__(self, nf, nx):
        """
        Initialize the 1D convolutional layer.

        Args:
            nf (int): The number of output features.
            nx (int): The number of input features.
        """
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = Parameter(w)
        self.bias = Parameter(torch.zeros(nf))

    def forward(self, x):
        """
        Forward pass through the Conv1D layer.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying the convolutional transformation.
        """
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x

class Attention(nn.Module):
    """
    Multi-headed self-attention mechanism.
    """
    def __init__(self, nx, n_ctx, config, scale=False):
        """
        Initialize the attention mechanism.

        Args:
            nx (int): The number of input features.
            n_ctx (int): The number of context tokens.
            config (object): Configuration object containing hyperparameters.
            scale (bool, optional): Whether to scale the attention logits.
        """
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)

    def _attn(self, q, k, v):
        """
        Perform the attention operation.

        Args:
            q (Tensor): The query tensor.
            k (Tensor): The key tensor.
            v (Tensor): The value tensor.

        Returns:
            Tensor: The output tensor after applying attention.
        """
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]
        w = w * b - 1e10 * (1 - b)
        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        """
        Merge the heads of the multi-head attention into a single tensor.

        Args:
            x (Tensor): The input tensor with separated heads.

        Returns:
            Tensor: The output tensor with merged heads.
        """
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        """
        Split the input tensor into multiple attention heads.

        Args:
            x (Tensor): The input tensor.
            k (bool, optional): Whether to transpose for key attention (default is False).

        Returns:
            Tensor: The output tensor with separated heads.
        """
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x, layer_past=None):
        """
        Forward pass through the attention mechanism.

        Args:
            x (Tensor): The input tensor.
            layer_past (tuple, optional): Past key and value tensors for caching.

        Returns:
            Tensor: The output tensor after applying attention.
            Tensor: The present key-value pairs for caching.
        """
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present

class MLP(nn.Module):
    """
    A feed-forward multi-layer perceptron (MLP).
    """
    def __init__(self, n_state, config):
        """
        Initialize the MLP module.

        Args:
            n_state (int): The number of output features.
            config (object): Configuration object containing hyperparameters.
        """
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu

    def forward(self, x):
        """
        Forward pass through the MLP.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying the MLP transformation.
        """
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2

class Block(nn.Module):
    """
    A transformer block that includes attention and a feed-forward MLP.
    """
    def __init__(self, n_ctx, config, scale=False):
        """
        Initialize the transformer block.

        Args:
            n_ctx (int): The number of context tokens.
            config (object): Configuration object containing hyperparameters.
            scale (bool, optional): Whether to scale the attention logits.
        """
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None):
        """
        Forward pass through the transformer block.

        Args:
            x (Tensor): The input tensor.
            layer_past (tuple, optional): Past key and value tensors for caching.

        Returns:
            Tensor: The output tensor after applying the transformer block.
            Tensor: The present key-value pairs for caching.
        """
        a, present = self.attn(self.ln_1(x), layer_past=layer_past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present

class GPT2Model(nn.Module):
    """
    The GPT-2 model that consists of multiple transformer blocks.
    """
    def __init__(self, config):
        """
        Initialize the GPT-2 model.

        Args:
            config (object): Configuration object containing hyperparameters.
        """
        super(GPT2Model, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.vocab_size

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def set_embeddings_weights(self, model_embeddings_weights):
        """
        Set the weights of the embeddings layer.

        Args:
            model_embeddings_weights (Tensor): The pre-trained weights for the embeddings.
        """
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights

    def forward(self, input_ids, position_ids=None, token_type_ids=None, past=None):
        """
        Forward pass through the GPT-2 model.

        Args:
            input_ids (Tensor): The input token IDs.
            position_ids (Tensor, optional): The position IDs for the tokens.
            token_type_ids (Tensor, optional): The token type IDs.
            past (list, optional): Past key and value tensors for caching.

        Returns:
            Tensor: The output hidden states after applying the model.
            Tensor: The present key-value pairs for caching.
        """
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long,
                                        device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        presents = []
        for block, layer_past in zip(self.h, past):
            hidden_states, present = block(hidden_states, layer_past)
            presents.append(present)
        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape), presents

class GPT2LMHead(nn.Module):
    """
    The language modeling head that projects the output of the GPT-2 model to the vocabulary space.
    """
    def __init__(self, model_embeddings_weights, config):
        """
        Initialize the language modeling head.

        Args:
            model_embeddings_weights (Tensor): The pre-trained embeddings weights.
            config (object): Configuration object containing hyperparameters.
        """
        super(GPT2LMHead, self).__init__()
        self.n_embd = config.n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        """
        Set the weights of the embeddings layer for the language modeling head.

        Args:
            model_embeddings_weights (Tensor): The pre-trained weights for the embeddings.
        """
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights

    def forward(self, hidden_state):
        """
        Forward pass through the language modeling head.

        Args:
            hidden_state (Tensor): The hidden states from the GPT-2 model.

        Returns:
            Tensor: The logits representing the predicted token probabilities.
        """
        lm_logits = self.decoder(hidden_state)
        return lm_logits

class GPT2LMHeadModel(nn.Module):
    """
    The complete GPT-2 model for language modeling that includes both the transformer and language modeling head.
    """
    def __init__(self, config):
        """
        Initialize the GPT-2 language modeling head model.

        Args:
            config (object): Configuration object containing hyperparameters.
        """
        super(GPT2LMHeadModel, self).__init__()
        self.transformer = GPT2Model(config)
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, config)

    def set_tied(self):
        """
        Tie the weights of the language modeling head with the transformer model.
        """
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)

    def forward(self, input_ids, position_ids=None, token_type_ids=None, lm_labels=None, past=None):
        """
        Forward pass through the GPT-2 model and language modeling head.

        Args:
            input_ids (Tensor): The input token IDs.
            position_ids (Tensor, optional): The position IDs for the tokens.
            token_type_ids (Tensor, optional): The token type IDs.
            lm_labels (Tensor, optional): The labels for calculating the loss.
            past (list, optional): Past key and value tensors for caching.

        Returns:
            Tensor: The logits for language modeling.
            Tensor: The present key-value pairs for caching.
        """
        hidden_states, presents = self.transformer(input_ids, position_ids, token_type_ids, past)
        lm_logits = self.lm_head(hidden_states)
        if lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            return loss
        return lm_logits, presents
