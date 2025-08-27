import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


samples_per_digit = 50     
flip_fraction = 0.2        
iters = 500               


transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST('data', train=True, download=True, transform=transform)


patterns = []
for digit in range(10):
    idxs = (train_data.targets == digit).nonzero(as_tuple=False).squeeze()[:samples_per_digit]
    imgs = train_data.data[idxs].numpy()
    imgs = np.where(imgs > 127, 1, -1)  # Binarize to ±1
    prototype = np.sign(np.sum(imgs.reshape(samples_per_digit, -1), axis=0))
    prototype[prototype == 0] = 1
    patterns.append(prototype)
patterns = np.stack(patterns)

N = patterns.shape[1]


W = np.zeros((N, N))
for p in patterns:
    W += np.outer(p, p)
np.fill_diagonal(W, 0)
W /= N   

