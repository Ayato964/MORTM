import numpy as np


load = np.load("./out/np/datasets_small_decoder/bl4alice.mid.npz", allow_pickle=True)['arr_3']

print(load)