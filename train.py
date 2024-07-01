
import sys
import wandb

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data import get_mus_label_class, generate_input_seqs
from transformer import Transformer, MLP, Readout

wandb.init(
    # Set the wandb project where this run will be logged
    project="icl-data",
)

def plot_grad_flow(named_parameters):
    
    ave_grads = []
    layers = []

    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            # If this is a transformer block
            if n.split(".")[-3] == "causal_block":
                layers.append(n.split(".")[-2] + "," + n.split(".")[0][-1])
            else:
                layers.append(n.split(".")[-2])
            if p.grad is not None:
                ave_grads.append(p.grad.abs().mean().cpu())
            else:
                ave_grads.append(-1)
    
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers)
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)

epochs = 25000

K = 512
L = 128
S = 10000
N = 8
Nmax = 32
eps = 0

D = 63
P = 65

alpha = 0

P = 1.0/(np.arange(1,K+1)**alpha)
P /= np.sum(P)

B = 1
p_B = 0.25
p_C = 0.25

batchsize = 128
no_repeats = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = nn.CrossEntropyLoss()

def criterion(model, inputs, labels, epoch):
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs, epoch=epoch)
    loss = loss_fn(outputs, labels)
    return loss

def accuracy(model, inputs, labels, epoch=-1, vis_mode=-1, flip_labels=False):

    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs, epoch=epoch, vis_mode=vis_mode)

    label_preds = F.softmax(outputs, dim=-1)
    label_preds_inds = torch.argmax(label_preds, dim=1)
    label_inds = torch.argmax(labels, dim=1)

    if flip_labels:
        label_inds = (label_inds + 1) % labels.size(-1)
    
    correct = (label_preds_inds == label_inds).float()
    return correct.mean().item()

mlp_readout = Readout(L)
model = Transformer(L, mlp=mlp_readout).to(device)
# model = Transformer(L).to(device)
model.train()

optim = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
# optim = optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-6)
mus_label, mus_class, labels_class = get_mus_label_class(K,L,D)

test_inputs, test_labels  = generate_input_seqs(mus_label,mus_class,labels_class,S,N, Nmax,eps = eps, P = P, B = B, p_B = p_B, p_C = p_C, no_repeats = no_repeats)
test_inputs_ic, test_labels_ic =  generate_input_seqs(mus_label,mus_class,labels_class,S,N, Nmax,eps = eps, P = P, B = B, p_B = 1, p_C = 1, no_repeats = no_repeats)
test_inputs_ic2, test_labels_ic2 =  generate_input_seqs(mus_label,mus_class,labels_class,S,N, Nmax,eps = eps, P = P, B = B, p_B = 1, p_C = 0, flip_labels = True, no_repeats = no_repeats)
test_inputs_iw, test_labels_iw =  generate_input_seqs(mus_label,mus_class,labels_class,S,N, Nmax,eps = eps, P = P, B = 0, p_B = 0, p_C = 0, no_repeats = no_repeats)

for epoch in range(epochs):

    optim.zero_grad()
    inputs_batch, labels_batch, target_classes = generate_input_seqs(mus_label, mus_class, labels_class, batchsize, N, Nmax, eps=eps, P=P, B=B, p_B=p_B, p_C=p_C, output_target_labels=True, no_repeats=no_repeats)
    
    loss = criterion(model, inputs_batch, labels_batch, -1)
    loss.backward()

    plot_grad_flow(model.named_parameters())
    optim.step()

    print(f"Epoch: {epoch}, Loss: {loss.item()}")
    wandb.log({"epoch": epoch, "train_loss": loss})

    if epoch % 10 == 0:
        acc_test = accuracy(model, test_inputs, test_labels, epoch=-1, vis_mode=-1)
        acc_ic = accuracy(model, test_inputs_ic, test_labels_ic, epoch=epoch, vis_mode=1)
        acc_ic2 = accuracy(model, test_inputs_ic2, test_labels_ic2, epoch=epoch, vis_mode=2, flip_labels = True)
        acc_iw = accuracy(model, test_inputs_iw, test_labels_iw, epoch=epoch, vis_mode=3)
        print(f"Test acc: {acc_test}, IC acc: {acc_ic}, IC acc2: {acc_ic2}, IW acc: {acc_iw}")
        wandb.log({"eval_epoch": epoch, "test_acc": acc_test, "ic1_acc": acc_ic, "ic2_acc": acc_ic2, "iw_acc": acc_iw})

plt.savefig("./grads.png")