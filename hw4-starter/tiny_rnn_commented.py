# Tiny RNN text generator
# Big idea: show the network 40 characters and ask it what character comes next
# repeat that until we have a whole string of text

# numpy = fast math on arrays.  tensorflow = builds & trains the neural network
import numpy as np
import tensorflow as tf


# Keep the first run reproducible so classmates can compare results.
# Change this number—or set it to None—to experiment with different runs.
# Seed fixes results, so same seed gives the same results every time.
SEED = 1710
if SEED is not None:
    # fix the randomness inside TensorFlow to make the first run reproducible
    tf.keras.utils.set_random_seed(SEED)
# random number generator we'll use to pick characters when generating text
rng = np.random.default_rng(SEED)


# ---------- 1) Tiny corpus (swap this block to change tasks) ----------
# Training data.
tiny_lines = [
    "I like cats.",
    "I like dogs.",
    "I like noodles.",
    "I like tacos.",
    "I like books.",
    "I like robots.",
    "I like music.",
    "I like puzzles.",
    "I like pizza.",
    "I like coding.",
]
# concatenates the 10 sentences into a single string
# ("I like cats.\nI like dogs.\n...")
text = "\n".join(tiny_lines)

# Alternative corpora (uncomment one):
# DNA: text = "TATAAA\nCGCGCG\nATG...TAA\nACGTACGTACGT\n"
# Emoji: text = "☀️🌤️⛅🌧️⛈️🌈\n🍞🧈🍯\n🥚🍳🍞\n🙂➡️😊\n"
# Nursery: text = "Twinkle twinkle little star,\nHow I wonder what you are.\n"

print("Corpus length:", len(text))
# we build a lookup table to help the model convert characters to numbers and then back to letters
# chars = gives us the set of characters we can choose from (deduplicated and sorted alphabetically)
chars = sorted(list(set(text)))
# stoi = "string to integer": character mapped to -> number
stoi = {c: i for i, c in enumerate(chars)}
# itos = "integer to string": number mapped to -> character
itos = {i: c for c, i in stoi.items()}
# vocab_size = how many distinct characters exist fior the model to choose from
vocab_size = len(chars)
print("Vocab:", chars)

# ---------- 2) Vectorize to (input sequence -> next char) pairs ----------
# QA pairs for training and validation.  A question: a window of 40 characters.  answer: the character right after it
seq_len = 40  # how many chars model sees at once
step = 1      # advances the window forward 1 char each time
X_idx, y_idx = [], []  # # of questions and # of answers respectively 
# for loop through the text, stopping 40 characters before the end
for i in range(0, len(text) - seq_len, step):
    seq = text[i : i + seq_len]    # 40 characters starting at position i
    nxt = text[i + seq_len]        # the single character that comes right after it
    X_idx.append([stoi[c] for c in seq])  # store the window as a list of 40 numbers
    y_idx.append(stoi[nxt])               # store the answer as just one number

# convert the plain Python lists into numpy arrays
X = np.array(X_idx, dtype=np.int32)
y = np.array(y_idx, dtype=np.int32)
print("Num training samples:", len(X))

# ---------- 3) Model ----------
# Sequential = output of one layer is input into next layer
model = tf.keras.Sequential(
    [
        # Embedding encodes each character as a vector so similar characters can end up with similar vectors
        tf.keras.layers.Embedding(vocab_size, 32),
        # LSTM reads the 40 characters and builds a memory with 128 characters, but...
        # ... only the final memory is passed on as a summary of what was just read
        tf.keras.layers.LSTM(128),
        # Dense turns the final memory into one raw score per possible next character
        # Higher score for a character the model thinks is more likely
        tf.keras.layers.Dense(vocab_size),
    ]
)
model.compile(
    # optimizer = the method for adjusting the weights
    # 1e-2 (0.01) is the learning rate
    optimizer=tf.keras.optimizers.Adam(1e-2),
    # typical loss minimization with tf and keras 
    # from_logits=True means raw scores not yet probabilities
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)

# ---------- 4) Train (tiny & fast) ----------
# fit is the loop of guessing, measuring error, adjust the weights
# batch_size=64: look at 64 pairs before each adjustment
# epochs=20: go through the entire set of pairs 20 times
# verbose=0: don't print progress for every epoch
history = model.fit(X, y, batch_size=64, epochs=20, verbose=0)
# history remembers the loss after each epoch
print("Final loss:", history.history["loss"][-1])


# ---------- 5) Sampling helper with temperature ----------
# Temperature controls how likely a low probability pick is:
#   low almost always choose the top guess and high gives less likely characters more shares
def sample_logits(logits, temperature=1.0):
    # Pick a next character 
    if temperature <= 0:  # greedy
        # just takse the highest-scoring character
        return int(np.argmax(logits))
    # dividing by temperature makes the differences between scores more (low temp) or less (high temp) extreme
    logits = logits / temperature
    # softmax converts raw logit into probabilities that are positive and add up to 1
    probabilities = tf.nn.softmax(logits).numpy()
    # character i is chosen with probability probabilities[i]
    return int(rng.choice(len(probabilities), p=probabilities))


def next_char_probabilities(seed="I like ", temperature=1.0, top_n=5):
    """Return the most likely next characters for a prompt."""
    # The model needs exactly 40 characters. If the seed is shorter, add spaces.
    seed = seed if len(seed) >= seq_len else (" " * (seq_len - len(seed)) + seed)
    # convert the last 40 characters to numbers (unknown characters become 0)
    context = [stoi.get(c, 0) for c in seed[-seq_len:]]
    # the model expects a batch of inputs 
    x = np.array([context], dtype=np.int32)
    # ask the model for its raw scores
    logits = model.predict(x, verbose=0)[0]

    if temperature <= 0:
        # gives the top character 100% and everything else 0%
        probabilities = np.zeros_like(logits, dtype=np.float64)
        probabilities[np.argmax(logits)] = 1.0
    else:
        # same temperature + softmax steps as above 
        probabilities = tf.nn.softmax(logits / temperature).numpy()

    # argsort ranks characters lowest -> highest probability
    # only keep the top_n highest probability characters, and reverse to get highest -> lowest
    top_indices = np.argsort(probabilities)[-top_n:][::-1]
    # return (character, probability) pairs. converts numbers back into characters
    return [(itos[int(i)], float(probabilities[i])) for i in top_indices]


def generate(seed="I like ", n_chars=200, temperature=0.7):
    # Ensure seed length is at least seq_len by left-padding with spaces.
    seed = seed if len(seed) >= seq_len else (" " * (seq_len - len(seed)) + seed)
    # context = the sliding 40-character window
    context = [stoi.get(c, 0) for c in seed[-seq_len:]]
    # output collects everything we print
    output = list(seed)
    for _ in range(n_chars):
        x = np.array([context], dtype=np.int32)
        # 1. ask the model for scores on the next character
        logits = model.predict(x, verbose=0)[0]
        # 2. pick one character with the set temperature
        idx = sample_logits(logits, temperature)
        char = itos[idx]
        # 3. add it to the output
        output.append(char)
        # 4. moves the window by drop the oldest character, adding the new one at the end,
        #    then repeat 
        context = context[1:] + [idx]
    # combine the list of characters back into a single string
    return "".join(output)


# ---------- 6) Try a few temperatures ----------
# Run the same demo at four temperatures, from cautious (0.1) to wild (1.0)
for temperature in [0.1, 0.5, 0.7, 1.0]:
    print("\n=== Temperature", temperature, "===")
    # show what the model is calculating for the very first new character
    print("Top probabilities for the first generated character:")
    for char, probability in next_char_probabilities(
        seed="I like ", temperature=temperature
    ):
        print(f"  {char!r}: {probability:.1%}")
    # generates 180 characters of text starting from "I like "
    print(generate(seed="I like ", n_chars=180, temperature=temperature))
