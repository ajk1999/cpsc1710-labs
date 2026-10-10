import random

# Load the training text
text = open("input.txt", encoding="utf-8").read()

# Tokenize the text into words
tokens = text.split()

# Count how often each word follows each other word
counts = {}
for i in range(len(tokens) - 1):
    w1 = tokens[i]
    w2 = tokens[i + 1]
    if w1 not in counts:
        counts[w1] = {}
    if w2 not in counts[w1]:
        counts[w1][w2] = 0
    counts[w1][w2] += 1

# Show part of the model
print("Tokens:", len(tokens), "| Unique words:", len(set(tokens)))
print("Words that followed", repr(tokens[0]) + ":", counts[tokens[0]])

# Choose the next word at weighted random
def choose_next(w1):
    options = counts[w1]
    total = sum(options.values())
    r = random.uniform(0, total)
    running = 0
    for word in options:
        running = running + options[word]
        if r < running:
            return word

# Generate about 100 words
current = random.choice(list(counts))
output = [current]
while len(output) < 100:
    if current not in counts:
        current = random.choice(list(counts))
    else:
        current = choose_next(current)
    output.append(current)

# Show the result
print(" ".join(output))
