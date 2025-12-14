import numpy as np

TEST_MIDI = "16b9a230fb007c0009feee532c3c4686.mid.npz"

node = np.load(f"out/np/omega/task1/{TEST_MIDI}")

for i in range(len(node) - 1):
    print(node[f'array{i + 1}'])