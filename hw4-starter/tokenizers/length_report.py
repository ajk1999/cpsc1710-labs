from pathlib import Path

text = (Path(__file__).parent / "synthpara_tokenizer_test.txt").read_text(encoding="utf-8")

print("Corpus length:", len(text))
chars = sorted(list(set(text)))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
vocab_size = len(chars)
print("Vocab:", chars)
