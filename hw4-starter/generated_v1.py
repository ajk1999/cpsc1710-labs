# Tiny character-level RNN text generator (fresh implementation from prompt.txt)

import numpy as np
import tensorflow as tf

SEED = 1710  # set to None for non-reproducible runs
SEQ_LEN = 40
EMBED_DIM = 32
LSTM_UNITS = 128
EPOCHS = 20
BATCH_SIZE = 64
TEMPERATURES = [0.1, 0.5, 0.7, 1.0]

if SEED is not None:
    tf.keras.utils.set_random_seed(SEED)
rng = np.random.default_rng(SEED)

# ---------- 1) Corpus ----------
WORDS = ["cats", "dogs", "noodles", "tacos", "books",
         "robots", "music", "puzzles", "pizza", "coding"]
text = "\n".join(f"I like {w}." for w in WORDS)

# ---------- 2) Characters <-> numbers ----------
chars = sorted(set(text))
char_to_id = {c: i for i, c in enumerate(chars)}
id_to_char = dict(enumerate(chars))
vocab_size = len(chars)

print("Corpus length:", len(text))
print("Vocab:", chars)

# ---------- 3) Sliding-window question/answer pairs ----------
ids = np.array([char_to_id[c] for c in text], dtype=np.int32)
n_samples = len(ids) - SEQ_LEN
X = np.stack([ids[i : i + SEQ_LEN] for i in range(n_samples)])
y = ids[SEQ_LEN : SEQ_LEN + n_samples]
print("Num training samples:", len(X))

# ---------- 4) Model: Embedding -> LSTM -> Dense ----------
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, EMBED_DIM),
    tf.keras.layers.LSTM(LSTM_UNITS),
    tf.keras.layers.Dense(vocab_size),
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-2),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)

# ---------- 5) Train ----------
history = model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0)
print("Final loss:", history.history["loss"][-1])


# ---------- 6) Temperature sampling ----------
def to_probabilities(logits, temperature):
    """Softmax over logits scaled by temperature (<= 0 means greedy)."""
    if temperature <= 0:
        probs = np.zeros(len(logits), dtype=np.float64)
        probs[np.argmax(logits)] = 1.0
        return probs
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled -= scaled.max()  # numerical stability
    exp = np.exp(scaled)
    return exp / exp.sum()


def make_context(seed):
    """Left-pad with spaces to SEQ_LEN and encode the last SEQ_LEN characters."""
    seed = seed.rjust(SEQ_LEN)
    return [char_to_id.get(c, 0) for c in seed[-SEQ_LEN:]]


def predict_logits(context):
    return model.predict(np.array([context], dtype=np.int32), verbose=0)[0]


def top_next_chars(seed="I like ", temperature=1.0, top_n=5):
    probs = to_probabilities(predict_logits(make_context(seed)), temperature)
    best = np.argsort(probs)[::-1][:top_n]
    return [(id_to_char[int(i)], float(probs[i])) for i in best]


# ---------- 7) Generation ----------
def generate(seed="I like ", n_chars=180, temperature=0.7):
    context = make_context(seed)
    output = [seed]
    for _ in range(n_chars):
        probs = to_probabilities(predict_logits(context), temperature)
        idx = int(rng.choice(vocab_size, p=probs))
        output.append(id_to_char[idx])
        context = context[1:] + [idx]  # slide the window forward
    return "".join(output)


# ---------- 8) Demo ----------
for temperature in TEMPERATURES:
    print(f"\n=== Temperature {temperature} ===")
    print("Top probabilities for the first generated character:")
    for char, p in top_next_chars("I like ", temperature):
        print(f"  {char!r}: {p:.1%}")
    print(generate("I like ", n_chars=180, temperature=temperature))
