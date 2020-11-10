import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# GPU SETUP
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device, "\n")
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1),    'GB')


# PREPROCESSING
file = open("data/meta.txt", "rt")
text = file.read().lower()
file.close()

def f(x):
    if x in "\n.,!?;:/1234567890\"":
        return " "
    return x
text = "".join(list(map(f, text)))
corpus = text.split()


# EMBEDDINGS
embed_vectors, context_vectors = {}, {}
for word in corpus:
    if word not in embed_vectors.keys():
        embed_vectors[word]   = torch.rand(50, requires_grad=True)
        context_vectors[word] = torch.rand(50, requires_grad=True)

# TRAINING DATA
train_set = [] # (embed_vector_word, context_vector_word, similarity classification)
for i in range(2, len(corpus)-2):
    for j in [-2, -1, 1, 2]:
        train_set.append((corpus[i], corpus[i+j], 1))
    # add some negative sampling
    train_set.append((corpus[i], corpus[(i+1000) % len(corpus)], 0))

with open("data/train_set.txt", "wt") as f:
    for example in train_set:
        f.write(str(example) + "\n")


# LEARN
learning_rate = 1e-5
for epoch in range(20):
    print(f"Epoch {epoch + 1} in progress.")
    total_loss = 0
    for (i, (embed, context, sim_class)) in enumerate(train_set):
        loss = (
                sim_class -
                torch.sigmoid(torch.dot(embed_vectors[embed], context_vectors[context]))
                ) ** 2
        loss.backward()

        total_loss += loss
        if i % 999 == 0:
            print(f"Average Loss: {total_loss/i}")
        
        with torch.no_grad():
            embed_vectors[embed]     -= learning_rate * embed_vectors[embed].grad
            context_vectors[context] -= learning_rate * context_vectors[context].grad
            
            embed_vectors[embed].grad.zero_()
            context_vectors[context].grad.zero_()

print(embed_vectors["something"])

with open("data/embedding.txt", "wt") as f:
    for (word, tensor) in embed_vectors.items():
        f.write(f"{word} => {tensor.data.tolist()}\n")
