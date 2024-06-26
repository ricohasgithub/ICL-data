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

wandb.init(
    # Set the wandb project where this run will be logged
    project="icl-data-sessions",
)


def train_model(epochs, batch_size, learning_rate, model_type, output_dir):
    print(f"Training {model_type} model")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Saving model to: {output_dir}")

    # Example code to save the model (pseudo-code)
    # model = ...  # Your model training code here
    # model.save(os.path.join(output_dir, f"{model_type}_model.h5"))

    # Add the actual code for training your model and saving the weights here
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ICL Transformer.')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('--K', type=int, required=True, help='Number of epochs')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs')


    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate')
    parser.add_argument('--model_type', type=str, required=True, help='Type of the model to train')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the model weights')

    args = parser.parse_args()

    train_model(args.epochs, args.batch_size, args.learning_rate, args.model_type, args.output_dir)
