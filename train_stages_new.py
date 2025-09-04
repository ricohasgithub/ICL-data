import argparse
import wandb

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data_new import get_mus_label_class, generate_input_seqs
from transformer import (
    Transformer,
    MLP,
    Readout,
    DisentangledTransformer,
    RestrictedDisentangledTransformer,
)
from util import (
    gen_attention_map_gif,
    create_image_gif_folder_structure,
    vis_attention_weights,
    svd_attention_weights,
    vis_output,
    plot_wo,
)
import uuid


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train transformer with 3-stage process')
    
    # Main hyperparameters (positional for backward compatibility)
    parser.add_argument('p_B', type=float, nargs='?', default=0.5, help='p_B parameter')
    parser.add_argument('p_C', type=float, nargs='?', default=0.5, help='p_C parameter')
    parser.add_argument('circuit_num', type=int, nargs='?', default=0, help='Circuit number')
    parser.add_argument('block0_num', type=int, nargs='?', default=1, help='Block 0 number')
    parser.add_argument('block1_num', type=int, nargs='?', default=0, help='Block 1 number')
    parser.add_argument('T_stage2', type=int, nargs='?', default=100, help='Number of steps for stage 2')
    parser.add_argument('lr_stage1', type=float, nargs='?', default=0.01, help='Learning rate for stage 1')
    parser.add_argument('lr_stage2', type=float, nargs='?', default=0.1, help='Learning rate for stage 2')
    parser.add_argument('lr_stage3', type=float, nargs='?', default=0.05, help='Learning rate for stage 3')
    
    # Optional arguments for other hyperparameters
    parser.add_argument('--D', type=int, default=63, help='Dimension D')
    parser.add_argument('--K', type=int, default=16, help='Number of classes K')
    parser.add_argument('--L', type=int, default=16, help='Number of labels L')
    parser.add_argument('--N', type=int, default=8, help='Number of examples N')
    parser.add_argument('--S', type=int, default=10000, help='Test set size S')
    parser.add_argument('--B', type=int, default=2, help='Burstiness parameter B')
    parser.add_argument('--batchsize', type=int, default=128, help='Batch size')
    parser.add_argument('--lamb', type=float, default=1e-9, help='Weight decay lambda')
    parser.add_argument('--eps', type=float, default=0, help='Epsilon noise parameter')
    parser.add_argument('--alpha', type=float, default=0, help='Alpha parameter for class distribution')
    parser.add_argument('--no_repeats', action='store_true', help='Disable repeats in sequences')
    parser.add_argument('--use_mlp', action='store_true', help='Use MLP instead of Readout')
    parser.add_argument('--no_disentangled', action='store_true', help='Disable disentangled transformer')
    
    return parser.parse_args()

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
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers)
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)


# Parse command line arguments
args = parse_arguments()

# Extract all arguments
p_B = args.p_B
p_C = args.p_C
circuit_num = args.circuit_num
block0_num = args.block0_num
block1_num = args.block1_num
T_stage2 = args.T_stage2
lr_stage1 = args.lr_stage1
lr_stage2 = args.lr_stage2
lr_stage3 = args.lr_stage3

D = args.D
K = args.K
L = args.L
N = args.N
S = args.S
B = args.B
batchsize = args.batchsize
lamb = args.lamb
eps = args.eps
alpha = args.alpha
no_repeats = args.no_repeats
use_mlp = args.use_mlp
use_disentangled = not args.no_disentangled

# Total epochs = 1 (stage 1) + T_stage2 (stage 2) + 1 (stage 3)
epochs = 1 + T_stage2 + 1

Nmax = 9
P = 2 * N + 1

P = 1.0 / (np.arange(1, K + 1) ** alpha)
P /= np.sum(P)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = nn.CrossEntropyLoss()


def criterion(model, inputs, labels, epoch):
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs, epoch=epoch)
    loss = loss_fn(outputs, labels)
    return loss


def accuracy(
    model, inputs, labels, epoch=-1, vis_mode=-1, flip_labels=False, vis_path=None
):

    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs, epoch=epoch, vis_mode=vis_mode, vis_path=vis_path)

    # if vis_mode == 1:
    #     vis_output(vis_path, outputs, labels, inputs, epoch)
    label_preds = F.softmax(outputs, dim=-1)
    label_preds_inds = torch.argmax(label_preds, dim=1)
    label_inds = torch.argmax(labels, dim=1)

    if flip_labels:
        label_inds = (label_inds + 1) % labels.size(-1)

    correct = (label_preds_inds == label_inds).float()
    return correct.mean().item()


if not use_mlp:

    wandb.init(
        # Set the wandb project where this run will be logged
        project="icl-data",
        name=f"3Stage, CN={circuit_num}, B0={block0_num}, B1={block1_num}, T={T_stage2}, lr1={lr_stage1}, lr2={lr_stage2}, lr3={lr_stage3}, p_B={p_B}, p_C={p_C}",
    )

    run_path = f"3Stage|CN={circuit_num}|B0={block0_num}|B1={block1_num}|T={T_stage2}|lr1={lr_stage1}|lr2={lr_stage2}|lr3={lr_stage3}|p_B={p_B}|p_C={p_C}"

    mlp_readout = Readout(L)

    if not use_disentangled:
        model = Transformer(L, mlp=mlp_readout).to(device)
    else:
        # model = DisentangledTransformer(L, mlp=mlp_readout).to(device)
        model = RestrictedDisentangledTransformer(
            L, P=2 * N + 1, mlp=mlp_readout, circuit_num=circuit_num
        ).to(device)
        print(model)
else:

    wandb.init(
        # Set the wandb project where this run will be logged
        project="icl-data",
        name=f"MLP, CN={circuit_num}, B0={block0_num}, B1={block1_num}, K={K}, L={L}, p_B={p_B}, p_C={p_C}, B={B}, eps={eps}, lamb={lamb}",
    )
    run_path = f"MLP|CN={circuit_num}|B0={block0_num}|B1={block1_num}|K={K}|L={L}|p_B={p_B}|p_C={p_C}|B={B}|eps={eps}"

    if not use_disentangled:
        model = Transformer(L).to(device)
    else:
        # model = DisentangledTransformer(L).to(device)
        model = RestrictedDisentangledTransformer(L, circuit_num=circuit_num).to(device)


run_path += f"|{uuid.uuid4()}"

if use_disentangled:
    run_path = "Unrestricted Disentangled|" + run_path
else:
    run_path = "Transformer|" + run_path

create_image_gif_folder_structure(run_path)

model_save_path = "./runs/" + run_path + "/model/"


model.train()

# optim = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)

out_param = [p for name, p in model.named_parameters() if "W_O" in name]
other_params = [p for name, p in model.named_parameters() if "W_O" not in name]


optim = optim.SGD(
    [{"params": other_params}, {"params": out_param, "weight_decay": 0.000}],
    lr=1e-2,
    weight_decay=lamb,
)
mus_label, mus_class, labels_class = get_mus_label_class(K, L, D)

with open(f"./runs/{run_path}/mus_label.npy", "wb") as f:
    np.save(f, mus_label)
with open(f"./runs/{run_path}/mus_class.npy", "wb") as f:
    np.save(f, mus_class)
with open(f"./runs/{run_path}/labels_class.npy", "wb") as f:
    np.save(f, labels_class)

test_inputs, test_labels = generate_input_seqs(
    mus_label,
    mus_class,
    labels_class,
    S,
    N,
    N,
    eps=eps,
    P=P,
    B=B,
    p_B=p_B,
    p_C=p_C,
    no_repeats=no_repeats,
)
test_inputs_ic, test_labels_ic = generate_input_seqs(
    mus_label,
    mus_class,
    labels_class,
    S,
    N,
    N,
    eps=eps,
    P=P,
    B=B,
    p_B=1,
    p_C=1,
    no_repeats=no_repeats,
    gen_vis=True,
    save_path=run_path,
)
test_inputs_ic2, test_labels_ic2 = generate_input_seqs(
    mus_label,
    mus_class,
    labels_class,
    S,
    N,
    N,
    eps=eps,
    P=P,
    B=B,
    p_B=1,
    p_C=0,
    flip_labels=True,
    no_repeats=no_repeats,
)
test_inputs_iw, test_labels_iw = generate_input_seqs(
    mus_label,
    mus_class,
    labels_class,
    S,
    N,
    N,
    eps=eps,
    P=P,
    B=0,
    p_B=0,
    p_C=0,
    no_repeats=no_repeats,
)

print("Running experiment " + run_path)

# Stage 1: Train W_O for 1 step
print("Stage 1: Training W_O for 1 step")
optim_stage1 = optim.SGD(
    [{"params": model.W_O.parameters()}],
    lr=lr_stage1,
    weight_decay=0,
)

for epoch in range(1):
    
    optim_stage1.zero_grad()
    inputs_batch, labels_batch, target_classes = generate_input_seqs(
        mus_label,
        mus_class,
        labels_class,
        batchsize,
        N,
        N,
        eps=eps,
        P=P,
        B=B,
        p_B=p_B,
        p_C=p_C,
        output_target_labels=True,
        no_repeats=no_repeats,
    )
    
    loss = criterion(model, inputs_batch, labels_batch, -1)
    loss.backward()
    
    # Only train W_O in stage 1 - zero out all other gradients
    if use_disentangled:
        model.transformer_block_0.causal_block.W_QK.weight.grad[:, :] = 0
        model.transformer_block_1.causal_block.W_QK.weight.grad[:, :] = 0
        if model.transformer_block_0.causal_block.W_V.weight.grad != None:
            model.transformer_block_0.causal_block.W_V.weight.grad[:, :] = 0
            model.transformer_block_1.causal_block.W_V.weight.grad[:, :] = 0
        
        # Apply circuit mask to W_O
        if circuit_num >= 0:
            out_mask = torch.zeros_like(model.W_O.weight)
            out_mask[
                :L,
                circuit_num
                * (model.P + model.D) : (circuit_num + 1)
                * (model.P + model.D),
            ] = 1
            model.W_O.weight.grad[:, :] = model.W_O.weight.grad * out_mask
    
    optim_stage1.step()
    
    print(f"Stage 1 - Epoch: {epoch}, Loss: {loss.item()}")
    wandb.log({"stage": 1, "epoch": epoch, "train_loss": loss})
    
    # Evaluate after stage 1
    acc_test = accuracy(
        model, test_inputs, test_labels, epoch=-1, vis_mode=-1, vis_path=run_path
    )
    acc_ic = accuracy(
        model,
        test_inputs_ic,
        test_labels_ic,
        epoch=epoch,
        vis_mode=1,
        vis_path=run_path,
    )
    acc_ic2 = accuracy(
        model,
        test_inputs_ic2,
        test_labels_ic2,
        epoch=epoch,
        vis_mode=2,
        flip_labels=True,
        vis_path=run_path,
    )
    acc_iw = accuracy(
        model,
        test_inputs_iw,
        test_labels_iw,
        epoch=epoch,
        vis_mode=3,
        vis_path=run_path,
    )
    print(
        f"Stage 1 - Test acc: {round(acc_test, 4)}, IC acc: {round(acc_ic, 4)}, IC acc2: {round(acc_ic2, 4)}, IW acc: {round(acc_iw, 4)}"
    )
    wandb.log(
        {
            "stage": 1,
            "eval_epoch": epoch,
            "test_acc": acc_test,
            "ic1_acc": acc_ic,
            "ic2_acc": acc_ic2,
            "iw_acc": acc_iw,
        }
    )

# Stage 2: Train Layer 0 QK for T_stage2 steps
print(f"Stage 2: Training Layer 0 QK for {T_stage2} steps")
optim_stage2 = optim.SGD(
    [{"params": model.transformer_block_0.causal_block.W_QK.parameters()}],
    lr=lr_stage2,
    weight_decay=lamb,
)

for epoch in range(1, 1 + T_stage2):
    
    optim_stage2.zero_grad()
    inputs_batch, labels_batch, target_classes = generate_input_seqs(
        mus_label,
        mus_class,
        labels_class,
        batchsize,
        N,
        N,
        eps=eps,
        P=P,
        B=B,
        p_B=p_B,
        p_C=p_C,
        output_target_labels=True,
        no_repeats=no_repeats,
    )

    loss = criterion(model, inputs_batch, labels_batch, -1)
    loss.backward()

    # Only train Layer 0 QK in stage 2 - zero out all other gradients
    if use_disentangled:
        # Apply block mask to Layer 0 QK based on block0_num
        if block0_num in [1, 2, 3]:
            model.transformer_block_0.causal_block.W_QK.weight.grad[
                : model.P, : model.P
            ] = 0
        if block0_num in [0, 2, 3]:
            model.transformer_block_0.causal_block.W_QK.weight.grad[
                model.P :, : model.P
            ] = 0
        if block0_num in [0, 1, 3]:
            model.transformer_block_0.causal_block.W_QK.weight.grad[
                : model.P, model.P :
            ] = 0
        if block0_num in [0, 1, 2]:
            model.transformer_block_0.causal_block.W_QK.weight.grad[
                model.P :, model.P :
            ] = 0
        
        # Zero out Layer 1 QK gradients
        model.transformer_block_1.causal_block.W_QK.weight.grad[:, :] = 0
        
        # Zero out W_V gradients
        if model.transformer_block_0.causal_block.W_V.weight.grad != None:
            model.transformer_block_0.causal_block.W_V.weight.grad[:, :] = 0
            model.transformer_block_1.causal_block.W_V.weight.grad[:, :] = 0
        
        # Zero out W_O gradients
        model.W_O.weight.grad[:, :] = 0
    
    optim_stage2.step()

    print(f"Stage 2 - Epoch: {epoch}, Loss: {loss.item()}")
    wandb.log({"stage": 2, "epoch": epoch, "train_loss": loss})

    if epoch % 10 == 0:
        acc_test = accuracy(
            model, test_inputs, test_labels, epoch=-1, vis_mode=-1, vis_path=run_path
        )
        acc_ic = accuracy(
            model,
            test_inputs_ic,
            test_labels_ic,
            epoch=epoch,
            vis_mode=1,
            vis_path=run_path,
        )
        acc_ic2 = accuracy(
            model,
            test_inputs_ic2,
            test_labels_ic2,
            epoch=epoch,
            vis_mode=2,
            flip_labels=True,
            vis_path=run_path,
        )
        acc_iw = accuracy(
            model,
            test_inputs_iw,
            test_labels_iw,
            epoch=epoch,
            vis_mode=3,
            vis_path=run_path,
        )
        print(
            f"Stage 2 - Test acc: {round(acc_test, 4)}, IC acc: {round(acc_ic, 4)}, IC acc2: {round(acc_ic2, 4)}, IW acc: {round(acc_iw, 4)}"
        )
        wandb.log(
            {
                "stage": 2,
                "eval_epoch": epoch,
                "test_acc": acc_test,
                "ic1_acc": acc_ic,
                "ic2_acc": acc_ic2,
                "iw_acc": acc_iw,
            }
        )

# Stage 3: Train Layer 1 QK for 1 step
print("Stage 3: Training Layer 1 QK for 1 step")
optim_stage3 = optim.SGD(
    [{"params": model.transformer_block_1.causal_block.W_QK.parameters()}],
    lr=lr_stage3,
    weight_decay=lamb,
)

for epoch in range(1 + T_stage2, 2 + T_stage2):
    
    optim_stage3.zero_grad()
    inputs_batch, labels_batch, target_classes = generate_input_seqs(
        mus_label,
        mus_class,
        labels_class,
        batchsize,
        N,
        N,
        eps=eps,
        P=P,
        B=B,
        p_B=p_B,
        p_C=p_C,
        output_target_labels=True,
        no_repeats=no_repeats,
    )
    
    loss = criterion(model, inputs_batch, labels_batch, -1)
    loss.backward()
    
    # Only train Layer 1 QK in stage 3 - zero out all other gradients
    if use_disentangled:
        # Zero out Layer 0 QK gradients
        model.transformer_block_0.causal_block.W_QK.weight.grad[:, :] = 0
        
        # Apply block mask to Layer 1 QK based on block1_num
        mask1 = torch.zeros_like(model.transformer_block_1.causal_block.W_QK.weight)
        
        if block1_num == 0:
            # C^2_{D, D}
            mask1[model.P : model.P + model.D, 2 * model.P + model.D :] = 1
        if block1_num == 1:
            # F^2_{P, P}
            mask1[
                model.P + model.D : 2 * model.P + model.D,
                model.P + model.D : 2 * model.P + model.D,
            ] = 1
        if block1_num == 2:
            # E^2_{P, P}
            mask1[model.P + model.D : 2 * model.P + model.D, 0 : model.P] = 1
        if block1_num == 3:
            # F^2_{D, D}
            mask1[2 * model.P + model.D :, 2 * model.P + model.D :] = 1
        if block1_num == 4:
            mask1[0 : model.P, 0 : model.P] = 1
            mask1[model.P + model.D : 2 * model.P + model.D, 0 : model.P] = 1
            mask1[
                model.P + model.D : 2 * model.P + model.D,
                model.P + model.D : 2 * model.P + model.D,
            ] = 1
        
        if block1_num in [0, 1, 2, 3, 4]:
            model.transformer_block_1.causal_block.W_QK.weight.grad = (
                model.transformer_block_1.causal_block.W_QK.weight.grad * mask1
            )
        
        # Zero out W_V gradients
        if model.transformer_block_0.causal_block.W_V.weight.grad != None:
            model.transformer_block_0.causal_block.W_V.weight.grad[:, :] = 0
            model.transformer_block_1.causal_block.W_V.weight.grad[:, :] = 0
        
        # Zero out W_O gradients
        model.W_O.weight.grad[:, :] = 0
    
    optim_stage3.step()
    
    print(f"Stage 3 - Epoch: {epoch}, Loss: {loss.item()}")
    wandb.log({"stage": 3, "epoch": epoch, "train_loss": loss})
    
    # Final evaluation after stage 3
    acc_test = accuracy(
        model, test_inputs, test_labels, epoch=-1, vis_mode=-1, vis_path=run_path
    )
    acc_ic = accuracy(
        model,
        test_inputs_ic,
        test_labels_ic,
        epoch=epoch,
        vis_mode=1,
        vis_path=run_path,
    )
    acc_ic2 = accuracy(
        model,
        test_inputs_ic2,
        test_labels_ic2,
        epoch=epoch,
        vis_mode=2,
        flip_labels=True,
        vis_path=run_path,
    )
    acc_iw = accuracy(
        model,
        test_inputs_iw,
        test_labels_iw,
        epoch=epoch,
        vis_mode=3,
        vis_path=run_path,
    )
    print(
        f"Stage 3 Final - Test acc: {round(acc_test, 4)}, IC acc: {round(acc_ic, 4)}, IC acc2: {round(acc_ic2, 4)}, IW acc: {round(acc_iw, 4)}"
    )
    wandb.log(
        {
            "stage": 3,
            "eval_epoch": epoch,
            "test_acc": acc_test,
            "ic1_acc": acc_ic,
            "ic2_acc": acc_ic2,
            "iw_acc": acc_iw,
        }
    )

torch.save(model.state_dict(), model_save_path + f"model_final")

# plt.savefig("./grads.png")

for vis_mode in range(1, 4):
    for vis_mode in range(1, 4):
        for layer in range(2):
            gen_attention_map_gif(run_path, vis_mode=vis_mode, layer=layer)

# plt.savefig("./grads.png")
