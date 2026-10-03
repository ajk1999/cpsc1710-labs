import unicodedata
from pathlib import Path


def tokenize(text):
    # Step 1: lowercase all letters
    text = text.lower()

    # Step 2: remove all punctuation (any Unicode "P" category character)
    text = "".join(c for c in text if not unicodedata.category(c).startswith("P"))

    # Step 3: split on whitespace
    return text.split()


def build_vocab(tokens):
    # Start with an empty vocabulary
    stoi = {}  # token -> number
    itos = {}  # number -> token
    for token in tokens:
        if token not in stoi:
            # Give the token the next unused positional number
            index = len(stoi)
            stoi[token] = index
            itos[index] = token
    return stoi, itos


def encode(tokens, stoi):
    # Look up each token's number in the vocabulary and collect them in a new list
    return [stoi[token] for token in tokens]


def decode(indices, itos):
    # Look up which token each number belongs to and collect them in a new list
    return [itos[index] for index in indices]


def detokenize(decoded_tokens):
    # Put a space between each token to produce the final sentence
    return " ".join(decoded_tokens)


if __name__ == "__main__":
    text = (Path(__file__).parent / "synthpara_tokenizer_test.txt").read_text(encoding="utf-8")
    tokens = tokenize(text)
    print(tokens)

    stoi, itos = build_vocab(tokens)
    print("Vocab size:", len(stoi))
    print(stoi)
    print(itos)

    indices = encode(tokens, stoi)
    print(indices)

    decoded_tokens = decode(indices, itos)
    print(decoded_tokens)

    sentence = detokenize(decoded_tokens)
    print(sentence)
