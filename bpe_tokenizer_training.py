import re
import collections
import psutil
import time
import os
from datasets import load_dataset

def get_pair_frequencies(tokenized_texts):
    """
    Compute the frequency of adjacent tokens paris
    :param tokenized_texts: list of lists of tokenized texts
    :return: Counter of frequencies
    """
    pair_freqs = collections.Counter()
    for tokens in tokenized_texts:
        for i in range(len(tokens)-1):
            pair = (tokens[i], tokens[i+1])
            pair_freqs[pair] += 1
    return pair_freqs


def train_bpe_tokenizer(input_path: str, vocab_size: int, special_tokens: list):
    """Given the path to an input corpus, run train a BPE tokenizer and
        output its vocabulary and merges.

        Args:
            input_path (str | os.PathLike): Path to BPE tokenizer training data.
            vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
            special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
                These strings will never be split into multiple tokens, and will always be
                kept as a single token. If these special tokens occur in the `input_path`,
                they are treated as any other string.

        Returns:
            tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
                vocab:
                    The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                    to bytes (token bytes)
                merges:
                    BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                    representing that <token1> was merged with <token2>.
                    Merges are ordered by order of creation.
    """
    # Read the input text file
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Temporarily remove special tokens from text before tokenization
    for token in special_tokens:
        text = text.replace(token, '')

    # Convert into bytes
    text_bytes = text.encode('utf-8')

    # Initialize Vocabulary
    byte_vocab = {i: bytes([i]) for i in set(text_bytes)}

    # Initialize tokenized text
    tokenized_texts = [list(line.encode('utf-8')) for line in text.splitlines() if line]

    merges = []

    # Add special tokens to the vocab
    for token in special_tokens:
        byte_token = token.encode('utf-8') if isinstance(token, str) else token
        if byte_token not in byte_vocab.values():
            byte_vocab[max(byte_vocab.keys()) + 1] = byte_token

    # BPE merging loop
    current_vocab_size = len(byte_vocab)
    while current_vocab_size < vocab_size:
        pair_freqs = get_pair_frequencies(tokenized_texts)
        if not pair_freqs:
            break # if there is no more pairs to merge

        # Get most frequent adjacent token pair
        most_common_pair, _ = pair_freqs.most_common(1)[0]
        # Form a new token by merging the pair
        new_token = most_common_pair[0] + most_common_pair[1]
        # Add pair into merges
        merges.append(most_common_pair)
        # Add the new token in the vocabulary
        byte_vocab[max(byte_vocab.keys()) + 1] = new_token
        current_vocab_size += 1

    # Return updated vocabulary and merges
    return (byte_vocab,), merges

def main():
    # Initialize time and memory calculation
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024
    start_time = time.time()

    # Save dataset as file
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    input_file = "tinystories.txt"
    with open(input_file, 'w', encoding='utf-8') as f:
        for example in dataset:
            f.write(example['text'] + '\n')

    vocab_size = 10000
    special_tokens = ["<endoftext>"]

    vocab, merges = train_bpe_tokenizer(input_file, vocab_size, special_tokens)

    end_time = time.time()
    mem_after = process.memory_info().rss / 1024 / 1024

    # Outputting results
    print(f"Running time: {end_time - start_time: .2f} seconds")
    print(f"Memory usage: {mem_after - mem_before: .2f} MB")

    print("Vocab size: {}".format(len(vocab)))
    print("Sample vocab: ")
    for k, v in list(vocab[0].items())[:10]:
        print(f"ID: {k}, Token: {v}")

if __name__ == '__main__':
    main()
