import torch
import urllib.request import Request, urlopen

# https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing#scrollTo=h5hjCcLDr2WC

class GPT():
    def __init__(self) -> None:
        pass

    def load_save_tinyshakespeare(self):
        FILE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        request = Request(FILE_URL)
        with urlopen(request) as response:
            data = response.read().decode('utf-8')
            with open("tinyshakespeare.txt", "w", encoding="utf-8") as f:
                f.write(data)

    def data():
        with open('input.txt', 'r', encoding='utf-8') as f:
            text = f.read()

        chars = sorted(list(set(text)))
        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = { i:ch for i,ch in enumerate(chars) }
        encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
        decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string