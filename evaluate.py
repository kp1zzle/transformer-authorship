import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
#import matplotlib.pyplot as plt
#import seaborn
from torchtext import data, datasets
import spacy
#seaborn.set_context(context="talk")
import pdb
import pandas as pd

spacy_en = spacy.load('en')
def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"
SRC = data.Field(tokenize=tokenize_en, pad_token=BLANK_WORD)
TGT = data.Field(init_token = BOS_WORD, eos_token = EOS_WORD, pad_token=BLANK_WORD) 

train, test = data.TabularDataset.splits(path='../data', train=train_file, test=test_file, format='csv',
    fields=[('src', SRC), ('trg', TGT)])

SRC.build_vocab(train)
TGT.build_vocab(train)
print("Built vocabulary")

torch.open('model.pt')
print("Opened model")
model.eval()
# Evaluate model
# Load train data
with open('../data/C50-testData.csv', newline='', encoding="utf-8") as csvfile:
    dataset = pd.read_csv(csvfile, delimiter=',', names = ['text', 'author'], encoding='utf-8')
    
    texts = dataset['text'].tolist()
    labels = dataset['author'].tolist()

    correct = 0

    for sentence, label in zip(texts, labels):
        sent = tokenize_en(sentence)
        src = torch.LongTensor([[SRC.vocab.stoi[w] for w in sent]])
        src = Variable(src)
        src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
        out = greedy_decode(model, src, src_mask, 
                            max_len=3, start_symbol=TGT.stoi["<s>"])
        print("Translation:", end="\t")
        trans = ""
        for i in range(1, out.size(1)):
            sym = TGT.itos[out[0, i]]
            if sym == "</s>": break
            trans += sym + " "
        print(trans)

        if atoi(trans) == label:
            correct += 1

    acc = float(correct)/float(len(labels))
    print("Test acc: " + acc)

