"""
Byte Pair Encoding (BPE) Utilities.

This module provides utilities for Byte Pair Encoding (BPE) used in tokenizing and encoding textual data,
particularly for neural network models like GPT-based models. It includes an Encoder class that handles
BPE encoding and decoding, along with helper functions and utilities to convert between byte sequences
and Unicode characters for efficient tokenization.

Functions:
- bytes_to_unicode: Creates a mapping between UTF-8 byte sequences and Unicode characters for efficient
  tokenization and encoding.
- get_pairs: Returns a set of symbol pairs in a word represented as a tuple of symbols.
- get_encoder: Loads encoder and BPE merge rules from files and returns an instance of the Encoder class.

Classes:
- Encoder: Handles the encoding and decoding of text using Byte Pair Encoding. This class includes methods
  for performing BPE encoding (`bpe`), tokenizing text (`encode`), and decoding tokens back to text (`decode`).

Details:
The BPE encoding splits words into subword units, iteratively merging the most frequent pairs of characters
into a single symbol until no more merges can be made. This helps handle out-of-vocabulary words and reduces
the overall vocabulary size required for a model, improving generalization for unseen data.
"""

import os
import json
import regex as re
from functools import lru_cache

@lru_cache()
def bytes_to_unicode():
    """
    Returns a dictionary mapping UTF-8 byte values to Unicode characters.

    This function creates a large list of Unicode characters that cover a significant number of bytes
    (including characters such as punctuation and symbols) to ensure that Byte Pair Encoding (BPE) can
    handle most textual data efficiently. This avoids mappings to whitespace/control characters that may
    interfere with BPE encoding.

    Returns:
        dict: A dictionary mapping byte values to corresponding Unicode characters.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    """
    Returns a set of symbol pairs in a word.

    The word is represented as a tuple of symbols, and this function identifies and returns all consecutive
    symbol pairs in the word. This is a key step in applying the BPE algorithm, where the most frequent symbol
    pair is merged iteratively.

    Args:
        word (tuple): A tuple of symbols (strings) representing a word.

    Returns:
        set: A set containing the consecutive symbol pairs from the word.
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class Encoder:
    """
    Encoder for Byte Pair Encoding (BPE).

    This class provides the encoding and decoding functionality using the BPE algorithm. It is responsible
    for transforming text into a sequence of subword tokens using BPE merges and converting token sequences
    back into readable text.

    Attributes:
        encoder (dict): A dictionary mapping subword tokens to integers.
        decoder (dict): A dictionary mapping integers back to subword tokens.
        errors (str): Error handling strategy for decoding.
        byte_encoder (dict): A mapping from byte values to Unicode characters.
        byte_decoder (dict): A mapping from Unicode characters back to byte values.
        bpe_ranks (dict): A dictionary of BPE merge operations with their corresponding rank.
        cache (dict): A cache for storing already encoded tokens.
        pat (re.Pattern): Regular expression pattern for tokenizing text.
    """
    def __init__(self, encoder, bpe_merges, errors='replace'):
        """
        Initializes the Encoder with the given encoder, BPE merges, and error handling strategy.

        Args:
            encoder (dict): A dictionary mapping subword tokens to integers.
            bpe_merges (list): A list of tuple pairs representing BPE merge operations.
            errors (str, optional): Error handling strategy for decoding. Defaults to 'replace'.
        """
        self.encoder = encoder
        self.decoder = {v:k for k,v in self.encoder.items()}
        self.errors = errors
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Regular expression pattern for matching tokens
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token):
        """
        Encodes a token using the Byte Pair Encoding algorithm.

        This method applies the BPE algorithm iteratively, merging the most frequent symbol pairs in the token
        based on the provided BPE merge rules. It continues the merging process until no further merges are possible.

        Args:
            token (str): The token to be encoded.

        Returns:
            str: The encoded token after applying BPE.
        """
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        """
        Encodes a string of text into subword tokens using BPE.

        This method first tokenizes the input text using a regular expression pattern, then applies BPE encoding
        to each token, and finally returns the resulting subword tokens as a list.

        Args:
            text (str): The text to be tokenized and encoded.

        Returns:
            list: A list of subword tokens after applying BPE encoding.
        """
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        """
        Decodes a sequence of subword tokens back into the original text.

        This method takes a list of subword tokens and decodes them into a readable string by reversing the
        encoding process. It handles the conversion of token IDs back to characters using the decoder mapping.

        Args:
            tokens (list): A list of subword tokens to be decoded.

        Returns:
            str: The decoded text.
        """
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text

def get_encoder():
    """
    Loads the encoder and BPE merge rules from files and returns an instance of the Encoder class.

    This function loads the necessary files (`encoder.json` and `vocab.bpe`) that contain the encoder mappings
    and the BPE merge rules. It then initializes and returns an `Encoder` object with the loaded data.

    Returns:
        Encoder: An instance of the Encoder class with the loaded encoder and BPE data.
    """
    with open('./models/encoder.json', 'r') as f:
        encoder = json.load(f)
    with open('./models/vocab.bpe', 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    return Encoder(
        encoder=encoder,
        bpe_merges=bpe_merges,
    )
