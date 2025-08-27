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


def update_state(state, W, iters=iters):
    for _ in range(iters):
        i = np.random.randint(0, N)
        h = np.dot(W[i], state)
        state[i] = 1 if h > 0 else -1 if h < 0 else state[i]
    return state


def distort(pattern, flip_fraction=flip_fraction):
    distorted = pattern.copy()
    num_flip = int(flip_fraction * N)
    indices = np.random.choice(N, num_flip, replace=False)
    distorted[indices] = -distorted[indices]
    return distorted


correct = 0
for digit in range(10):
    original = patterns[digit]
    distorted = distort(original)
    retrieved = update_state(distorted.copy(), W)

  
    similarities = [np.dot(retrieved, p) for p in patterns]
    predicted_digit = np.argmax(similarities)
    if predicted_digit == digit:
        correct += 1

 
    plt.figure(figsize=(6,2))
    plt.suptitle(f"True: {digit}, Predicted: {predicted_digit}")
    plt.subplot(1,3,1)
    plt.imshow(original.reshape(28,28), cmap='gray')
    plt.title("Original")
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(distorted.reshape(28,28), cmap='gray')
    plt.title("Noisy")
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(retrieved.reshape(28,28), cmap='gray')
    plt.title("Retrieved")
    plt.axis('off')
    plt.show()

print(f"Accuracy: {correct}/10 digits ({correct*10}%)")
