# Code 1: Best Movie Recommendation
# (Chooses the movie with highest score for the user)

import torch

# User preference vector
user = torch.tensor([3, 4, 5])

# Each row is a movie with [action, drama, comedy] scores
movies = torch.tensor([
    [4, 6, 7],
    [8, 5, 4],
    [1, 2, 1]
])

# Multiply user preference with each movie and find best match
scores = movies @ user
print(scores.argmax().item())

# Code 2: Product Choice
# (Recommends which product suits the user best)

import torch

# User preferences
user = torch.tensor([1, 2, 3])

# Product features
products = torch.tensor([
    [2, 3, 4],
    [9, 8, 1],
    [4, 5, 6]
])

scores = products @ user
print(scores.argmax().item())

# Code 3: Music Recommendation
# (Finds which song best matches user's taste)

import torch

# User taste vector
user = torch.tensor([5, 7, 8])

# Song feature vectors
songs = torch.tensor([
    [4, 6, 7],
    [9, 7, 6],
    [3, 1, 1]
])

scores = songs @ user
print(scores.argmax().item())
