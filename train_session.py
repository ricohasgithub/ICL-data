import argparse
import os
import sys
import wandb

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data import get_mus_label_class, generate_input_seqs
from transformer import Transformer


def train_model(epochs, K, L, S, N, Nmax, eps, alpha, B, p_B, p_C, batchsize, no_repeats, output_dir):
    print(f"Saving model to: {output_dir}")

    D = 63

    P = 1.0/(np.arange(1,K+1)**alpha)
    P /= np.sum(P)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.CrossEntropyLoss()

    def criterion(model, inputs, labels):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        return loss

    def accuracy(model, inputs, labels, flip_labels=False):

        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        label_preds = F.softmax(outputs, dim=-1)
        label_preds_inds = torch.argmax(label_preds, dim=1)
        label_inds = torch.argmax(labels, dim=1)

        if flip_labels:
            label_inds = (label_inds + 1) % labels.size(-1)
        
        correct = (label_preds_inds == label_inds).float()
        return correct.mean().item()

    model = Transformer(L).to(device)
    model.train()

    optim = optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-6)
    mus_label, mus_class, labels_class = get_mus_label_class(K,L,D)

    test_inputs, test_labels  = generate_input_seqs(mus_label,mus_class,labels_class,S,N, Nmax,eps = eps, P = P, B = B, p_B = p_B, p_C = p_C, no_repeats = no_repeats)
    test_inputs_ic, test_labels_ic =  generate_input_seqs(mus_label,mus_class,labels_class,S,N, Nmax,eps = eps, P = P, B = B, p_B = 1, p_C = 1, no_repeats = no_repeats)
    test_inputs_ic2, test_labels_ic2 =  generate_input_seqs(mus_label,mus_class,labels_class,S,N, Nmax,eps = eps, P = P, B = B, p_B = 1, p_C = 0, flip_labels = True, no_repeats = no_repeats)
    test_inputs_iw, test_labels_iw =  generate_input_seqs(mus_label,mus_class,labels_class,S,N, Nmax,eps = eps, P = P, B = 0, p_B = 0, p_C = 0, no_repeats = no_repeats)

    for epoch in range(epochs):

        optim.zero_grad()
        inputs_batch, labels_batch, target_classes = generate_input_seqs(mus_label, mus_class, labels_class, batchsize, N, Nmax, eps=eps, P=P, B=B, p_B=p_B, p_C=p_C, output_target_labels=True, no_repeats=no_repeats)
        
        loss = criterion(model, inputs_batch, labels_batch)
        loss.backward()

        optim.step()

        print(f"Epoch: {epoch}, Loss: {loss.item()}")
        wandb.log({"epoch": epoch, "train_loss": loss})

        if epoch % 10 == 0:
            acc_test = accuracy(model, test_inputs, test_labels)
            acc_ic = accuracy(model, test_inputs_ic, test_labels_ic)
            acc_ic2 = accuracy(model, test_inputs_ic2, test_labels_ic2, flip_labels = True)
            acc_iw = accuracy(model, test_inputs_iw, test_labels_iw)
            print(f"Test acc: {acc_test}, IC acc: {acc_ic}, IC acc2: {acc_ic2}, IW acc: {acc_iw}")
            wandb.log({"eval_epoch": epoch, "test_acc": acc_test, "ic1_acc": acc_ic, "ic2_acc": acc_ic2, "iw_acc": acc_iw})



    # Example code to save the model (pseudo-code)
    # model = ...  # Your model training code here
    # model.save(os.path.join(output_dir, f"{model_type}_model.h5"))

    # Add the actual code for training your model and saving the weights here
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ICL Transformer.')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('--K', type=int, required=True, help='Number of classes')
    parser.add_argument('--L', type=int, required=True, help='Number of labels')
    parser.add_argument('--S', type=int, required=True, help='Size of training set')
    parser.add_argument('--N', type=int, required=True, help='Number of sample-label pairs in each input sequence')
    parser.add_argument('--Nmax', type=int, required=True, help='Maximum number of sample-label pairs in each input sequence')
    parser.add_argument('--eps', type=int, required=True, help='Epsilon (within-class variation)')

    parser.add_argument('--alpha', type=int, required=True, help='Alpha (rank-frequency exponent)')
    parser.add_argument('--B', type=int, required=True, help='Burstiness (number of occurrences of each class in each sequence)')

    parser.add_argument('--p_B', type=int, required=True, help='Probability of burstiness sequence in training')
    parser.add_argument('--p_C', type=int, required=True, help='Probability of in-context sequence in training')

    parser.add_argument('--batchsize', type=int, required=True, help='Batch size')
    parser.add_argument('--no_repeats', type=int, required=True, help='No Repeats')

    parser.add_argument('--run_name', type=str, required=True, help='Name of session by independent variable')

    args = parser.parse_args()
    
    wandb.init(
        # Set the wandb project where this run will be logged
        project="icl-data-sessions", name=args.run_name
    )

    train_model(args.epochs, args.K, args.L, args.S, args.N, args.Nmax, args.eps, args.alpha, args.B, args.p_B, args.p_C, args.batchsize, args.no_repeats, f'./sessions/{args.run_name}')
