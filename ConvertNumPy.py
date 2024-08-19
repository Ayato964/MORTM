from convert.ConvertMidi import ConvertMidi
from transformer.tokenizer import Tokenizer
import os

datasets = "data/other/"
files = os.listdir(datasets)

BRASS = [57, 58, 65, 66, 67, 68]
PIANO = [1, 2, 3, 4, 5, 6, 7, 8]
GUITAR = [25, 26, 27, 28, 29, 30, 31, 32]


ALL = [1, 2, 3, 4, 5, 6, 7, 8, 25, 26, 27, 28, 29, 30, 31, 32, 57, 58, 65, 66, 67, 68]
tokenizer = Tokenizer()
for file in files:
    con = ConvertMidi(tokenizer, datasets + file, BRASS, 120)
    con.convert()
    con.save()

tokenizer.save()
print(len(tokenizer.tokens))
