
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data import get_mus_label_class, generate_input_seqs
from transformer import Transformer

epochs = 500

K = 512
L = 32
S = 10000
N = 8
Nmax = 32
eps = 0

D = 63
P = 65

alpha = 0.01

P = 1.0/(np.arange(1,K+1)**alpha)
P /= np.sum(P)

B = 2
p_B = 0.5
p_C = 0

batchsize = 128
no_repeats = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def criterion(model, inputs, labels):
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    logits = F.softmax(outputs, dim=-1)
    loss = F.cross_entropy(logits, labels)
    return loss

def accuracy(model, inputs, labels, mask=None, flip_labels=False):
    outputs = model(inputs)
    label_preds = F.softmax(outputs, dim=-1)
    
    if mask is not None:
        label_preds += mask
    label_preds_inds = torch.argmax(label_preds, dim=1)
    
    label_inds = torch.argmax(labels, dim=1)
    if flip_labels:
        label_inds = (label_inds + 1) % labels.size(-1)
    
    correct = (label_preds_inds == label_inds).float()
    return correct.mean().item()

model = Transformer(L).to(device)
optim = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(epochs):

    optim.zero_grad()
    mus_label, mus_class, labels_class = get_mus_label_class(K,L,D)

    test_inputs, test_labels  = generate_input_seqs(mus_label,mus_class,labels_class,S,N, Nmax,eps = eps, P = P, B = B, p_B = p_B, p_C = p_C, no_repeats = no_repeats)
    test_inputs_ic, test_labels_ic =  generate_input_seqs(mus_label,mus_class,labels_class,S,N, Nmax,eps = eps, P = P, B = B, p_B = 1, p_C = 1, no_repeats = no_repeats)
    test_inputs_ic2, test_labels_ic2 =  generate_input_seqs(mus_label,mus_class,labels_class,S,N, Nmax,eps = eps, P = P, B = B, p_B = 1, p_C = 0, flip_labels = True, no_repeats = no_repeats)
    test_inputs_iw, test_labels_iw =  generate_input_seqs(mus_label,mus_class,labels_class,S,N, Nmax,eps = eps, P = P, B = 0, p_B = 0, p_C = 0, no_repeats = no_repeats)

    inputs_batch, labels_batch, target_classes = generate_input_seqs(mus_label, mus_class, labels_class, batchsize, N, Nmax, eps=eps, P=P, B=B, p_B=p_B, p_C=p_C, output_target_labels=True, no_repeats=no_repeats)
    
    loss = criterion(model, inputs_batch, labels_batch)
    loss.backward()
    optim.step()

    print(f"Epoch: {epoch}, Loss: {loss.item()}")