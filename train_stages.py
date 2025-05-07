import sys
import wandb

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data import get_mus_label_class, generate_input_seqs
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
)
import uuid


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


epochs = 1000


K = 512
L = 32
S = 10000
N = 8
Nmax = 9
eps = 0

D = 63
P = 17

alpha = 0

P = 1.0 / (np.arange(1, K + 1) ** alpha)
P /= np.sum(P)

B = 1
p_C = 1
p_B = 0

epochs_icl = 2000
epochs_icl_1 = 1000

epochs_iwl = 2000

batchsize = 128
no_repeats = False

use_mlp = False
use_disentangled = True

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

    label_preds = F.softmax(outputs, dim=-1)
    label_preds_inds = torch.argmax(label_preds, dim=1)
    label_inds = torch.argmax(labels, dim=1)

    if flip_labels:
        label_inds = (label_inds + 1) % labels.size(-1)

    correct = (label_preds_inds == label_inds).float()
    return correct.mean().item()


wandb.init(
    # Set the wandb project where this run will be logged
    project="icl-data",
    name=f"Readout, K={K}, L={L}, p_B={p_B}, p_C={p_C}, epochs_icl={epochs_icl}, epochs_iwl={epochs_iwl}, B={B}, eps={eps}",
)

run_path = f"Readout|K={K}|L={L}|p_B={p_B}|p_C={p_C}|epochs_icl={epochs_icl}|epochs_iwl={epochs_iwl}|B={B}|eps={eps}"

mlp_readout = Readout(L)


model = RestrictedDisentangledTransformer(L, mlp=mlp_readout).to(device)

run_path += f"|{uuid.uuid4()}"

if use_disentangled:
    run_path = "Disentangled|" + run_path
else:
    run_path = "Transformer|" + run_path

create_image_gif_folder_structure(run_path)

model_save_path = "./runs/" + run_path + "/model/"

model.train()

# optim = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)

out_param = [p for name, p in model.named_parameters() if "W_O" in name]
other_params = [p for name, p in model.named_parameters() if "W_O" not in name]


optim = optim.SGD(
    [
        {"params": other_params},
        {
            "params": out_param,
            "weight_decay": 0.0005,
            "lr": 0.5,
        },
    ],
    lr=1e-1,
    weight_decay=1e-6,
)
# optim = optim.SGD(model.parameters(), lr=1e-1)
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


print("ICL Phase: Stage 1")

print("ICL Phase: Stage 2")
for epoch in range(epochs_icl_1):
    if epoch % 50 == 0:
        torch.save(model.state_dict(), model_save_path + f"model_{epoch}")
        model_params = model.state_dict()

        if use_disentangled:
            QK0 = model_params["transformer_block_0.causal_block.W_QK.weight"]
            QK1 = model_params["transformer_block_1.causal_block.W_QK.weight"]

            W_O = model_params["W_O.weight"]

            # vis_attention_weights(
            #     QK0.cpu().detach().numpy(),
            #     QK1.cpu().detach().numpy(),
            #     W_O.cpu().detach().numpy(),
            #     # save_dir="./disentangled_model_plots/" + run_path + "/",
            #     save_dir="./circuit_plots/" + run_path + "/",
            #     model_name=f"model_{epoch}",
            #     hyper_params={"p_B": p_B, "p_C": p_C},
            # )

            svd_attention_weights(
                QK1.cpu()
                .detach()
                .numpy()[model.P : model.P + model.D, 2 * model.P + model.D :],
                layer=2,
                save_file_name=f"model_{epoch}",
                # save_dir="./disentangled_model_plots/" + run_path + "/",
                save_dir="./circuit_plots/" + run_path + "/",
                model_name=f"model_{epoch}",
                hyper_params={"p_B": p_B, "p_C": p_C},
            )

    optim.zero_grad()
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

    if use_disentangled:

        model.transformer_block_0.causal_block.W_QK.weight.grad[
            model.P :, : model.P
        ] = 0
        model.transformer_block_0.causal_block.W_QK.weight.grad[
            : model.P, model.P :
        ] = 0

        model.transformer_block_0.causal_block.W_QK.weight.grad[
            model.P :, model.P :
        ] = 0

        mask1 = torch.zeros_like(model.transformer_block_1.causal_block.W_QK.weight)
        mask1[model.P : model.P + model.D, 2 * model.P + model.D :] = 1
        model.transformer_block_1.causal_block.W_QK.weight.grad = (
            model.transformer_block_1.causal_block.W_QK.weight.grad * mask1
        )

        # model.transformer_block_0.causal_block.W_V.weight.grad[:, :] = 0
        # model.transformer_block_1.causal_block.W_V.weight.grad[:, :] = 0

        out_mask = torch.zeros_like(model.W_O.weight)
        # out_mask[:L, model.P : model.P + model.D] = 1
        # out_mask[:L, 3 * model.P + 2 * model.D : 3 * model.P + 3 * model.D] = 1
        model.W_O.weight.grad[:, :] = model.W_O.weight.grad * out_mask

    optim.step()

    print(f"Epoch: {epoch}, Loss: {loss.item()}")
    wandb.log({"epoch": epoch, "train_loss": loss})

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
            f"Test acc: {round(acc_test, 4)}, IC acc: {round(acc_ic, 4)}, IC acc2: {round(acc_ic2, 4)}, IW acc: {round(acc_iw, 4)}"
        )
        wandb.log(
            {
                "eval_epoch": epoch,
                "test_acc": acc_test,
                "ic1_acc": acc_ic,
                "ic2_acc": acc_ic2,
                "iw_acc": acc_iw,
            }
        )

for epoch in range(epochs_icl_1, epochs_icl):
    if epoch % 50 == 0:
        torch.save(model.state_dict(), model_save_path + f"model_{epoch}")
        model_params = model.state_dict()

        if use_disentangled:
            QK0 = model_params["transformer_block_0.causal_block.W_QK.weight"]
            QK1 = model_params["transformer_block_1.causal_block.W_QK.weight"]

            W_O = model_params["W_O.weight"]

            # vis_attention_weights(
            #     QK0.cpu().detach().numpy(),
            #     QK1.cpu().detach().numpy(),
            #     W_O.cpu().detach().numpy(),
            #     # save_dir="./disentangled_model_plots/" + run_path + "/",
            #     save_dir="./circuit_plots/" + run_path + "/",
            #     model_name=f"model_{epoch}",
            #     hyper_params={"p_B": p_B, "p_C": p_C},
            # )

            svd_attention_weights(
                QK1.cpu()
                .detach()
                .numpy()[model.P : model.P + model.D, 2 * model.P + model.D :],
                layer=2,
                save_file_name=f"model_{epoch}",
                # save_dir="./disentangled_model_plots/" + run_path + "/",
                save_dir="./circuit_plots/" + run_path + "/",
                model_name=f"model_{epoch}",
                hyper_params={"p_B": p_B, "p_C": p_C},
            )

    optim.zero_grad()
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

    if use_disentangled:

        model.transformer_block_0.causal_block.W_QK.weight.grad[:, :] = 0
        model.transformer_block_1.causal_block.W_QK.weight.grad[:, :] = 0

        model.transformer_block_0.causal_block.W_V.weight.grad[:, :] = 0
        model.transformer_block_1.causal_block.W_V.weight.grad[:, :] = 0

        out_mask = torch.zeros_like(model.W_O.weight)
        # out_mask[:L, model.P : model.P + model.D] = 1
        out_mask[:L, 3 * model.P + 2 * model.D : 3 * model.P + 3 * model.D] = 1
        model.W_O.weight.grad[:, :] = model.W_O.weight.grad * out_mask

    optim.step()

    print(f"Epoch: {epoch}, Loss: {loss.item()}")
    wandb.log({"epoch": epoch, "train_loss": loss})

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
            f"Test acc: {round(acc_test, 4)}, IC acc: {round(acc_ic, 4)}, IC acc2: {round(acc_ic2, 4)}, IW acc: {round(acc_iw, 4)}"
        )
        wandb.log(
            {
                "eval_epoch": epoch,
                "test_acc": acc_test,
                "ic1_acc": acc_ic,
                "ic2_acc": acc_ic2,
                "iw_acc": acc_iw,
            }
        )

print("IWL Phase")

for epoch in range(epochs_icl, epochs_icl + epochs_iwl):

    if epoch % 50 == 0:
        torch.save(model.state_dict(), model_save_path + f"model_{epoch}")
        model_params = model.state_dict()

        if use_disentangled:
            QK0 = model_params["transformer_block_0.causal_block.W_QK.weight"]
            QK1 = model_params["transformer_block_1.causal_block.W_QK.weight"]

            W_O = model_params["W_O.weight"]

            vis_attention_weights(
                QK0.cpu().detach().numpy(),
                QK1.cpu().detach().numpy(),
                W_O.cpu().detach().numpy(),
                # save_dir="./disentangled_model_plots/" + run_path + "/",
                save_dir="./circuit_plots/" + run_path + "/",
                model_name=f"model_{epoch}",
                hyper_params={"p_B": p_B, "p_C": p_C},
                P=P,
                D=32,
            )

            svd_attention_weights(
                QK1.cpu()
                .detach()
                .numpy()[model.P : model.P + model.D, 2 * model.P + model.D :],
                layer=2,
                save_file_name=f"model_{epoch}",
                # save_dir="./disentangled_model_plots/" + run_path + "/",
                save_dir="./circuit_plots/" + run_path + "/",
                model_name=f"model_{epoch}",
                hyper_params={"p_B": p_B, "p_C": p_C},
            )

    optim.zero_grad()
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

    if use_disentangled:

        model.transformer_block_0.causal_block.W_QK.weight.grad[:, :] = 0
        model.transformer_block_1.causal_block.W_QK.weight.grad[:, :] = 0

        model.transformer_block_0.causal_block.W_V.weight.grad[:, :] = 0
        model.transformer_block_1.causal_block.W_V.weight.grad[:, :] = 0

        out_mask = torch.zeros_like(model.W_O.weight)
        # out_mask[:L, model.P : model.P + model.D] = 1
        out_mask[:L, 3 * model.P + 2 * model.D : 3 * model.P + 3 * model.D] = 1
        model.W_O.weight.grad[:, :] = model.W_O.weight.grad * out_mask

    # plot_grad_flow(model.named_parameters())
    optim.step()

    print(f"Epoch: {epoch}, Loss: {loss.item()}")
    wandb.log({"epoch": epoch, "train_loss": loss})

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
            f"Test acc: {round(acc_test, 4)}, IC acc: {round(acc_ic, 4)}, IC acc2: {round(acc_ic2, 4)}, IW acc: {round(acc_iw, 4)}"
        )
        wandb.log(
            {
                "eval_epoch": epoch,
                "test_acc": acc_test,
                "ic1_acc": acc_ic,
                "ic2_acc": acc_ic2,
                "iw_acc": acc_iw,
            }
        )


torch.save(model.state_dict(), model_save_path + f"model_{epoch}")

# plt.savefig("./grads.png")

for vis_mode in range(1, 4):
    for vis_mode in range(1, 4):
        for layer in range(2):
            gen_attention_map_gif(run_path, vis_mode=vis_mode, layer=layer)

# plt.savefig("./grads.png")
