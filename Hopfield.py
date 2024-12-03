import numpy as np
import matplotlib.pyplot as plt

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))
    
    def train(self, patterns):
        for p in patterns:
            self.weights += np.outer(p, p)
        np.fill_diagonal(self.weights, 0)
    
    def recall(self, pattern, steps=10):
        for _ in range(steps):
            for i in np.random.permutation(self.size):
                pattern[i] = 1 if np.dot(self.weights[i], pattern) > 0 else -1
        return pattern

def to_binary(image):
    return np.where(image.flatten() > 0, 1, -1)

def to_image(binary, shape):
    return np.where(np.array(binary).reshape(shape) > 0, 1, 0)

def corrupt(image, fraction=0.25):
    corrupted = image.copy().flatten()
    size = len(corrupted)
    corrupted[:size // 4] = 1 - corrupted[:size // 4]
    return corrupted

face = np.zeros((10, 10), int)
face[9, 0] = face[8, 1] = face[7, 2] = face[6, 3] = face[6, 4] = face[6, 5] = face[7, 6] = face[8, 7] = face[8, 8] = face[9, 9] = 1
face[1, 3] = face[1, 7] = face[4, 4] = 1

tree = np.zeros((10, 10), int)
for i in range(10):
    tree[i, 4] = tree[i, 5] = 1
tree[0, 3:7] = tree[1, 3:7] = 1
tree[4, 2:9] = 1
tree[5, 8] = 1

patterns = [to_binary(face), to_binary(tree)]
network = HopfieldNetwork(100)
network.train(patterns)

corrupted_face = corrupt(face)
recalled_face = network.recall(to_binary(corrupted_face))

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title("Original Face")
plt.imshow(face, cmap='binary')
plt.subplot(1, 3, 2)
plt.title("Corrupted Face")
plt.imshow(to_image(corrupted_face, face.shape), cmap='binary')
plt.subplot(1, 3, 3)
plt.title("Recalled Face")
plt.imshow(to_image(recalled_face, face.shape), cmap='binary')
plt.show()
