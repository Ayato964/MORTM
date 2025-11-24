import numpy as np

TEST_MIDI = "000c004a21a44e2c80f3f549f4abc8b5.mid"

node = np.load(f"out/np/research/chord/{TEST_MIDI}.npz")

for i in range(len(node) - 1):
    print(node[f'array{i + 1}'])