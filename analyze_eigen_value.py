import torch
import seaborn
import matplotlib.pyplot as plt
import os
from transformer import (
    DisentangledTransformer,
    Readout,
    RestrictedDisentangledTransformer,
)
from util import vis_attention_weights
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


use_mlp = False
use_disentangled = True
K = 512
L = 32
p_B = 0.375
p_C = 0.375
B = 1
eps = 0
uuid = "6403da1a-5b9d-4a41-b2dd-7b11cc0c9090"

if use_mlp:
    last_layer = "MLP"
else:
    last_layer = "Readout"


model_type = "Disentangled"


run_name = (
    f"{model_type}|{last_layer}|K={K}|L={L}|p_B={p_B}|p_C={p_C}|B={B}|eps={eps}|{uuid}"
)

run_name_125 = "Disentangled|Readout|K=512|L=32|p_B=0.125|p_C=0.125|B=1|eps=0|b9be0dd3-ad6c-4e03-a359-f463b034e6fd"
run_name_375 = "Disentangled|Readout|K=512|L=32|p_B=0.375|p_C=0.375|B=1|eps=0|0da5c838-a8eb-4984-b80d-3f183ae958ec"
run_name_75 = "Disentangled|Readout|K=512|L=32|p_B=0.75|p_C=0.75|B=1|eps=0|557fe377-a1ea-4b4e-9a56-eef4e25a44ea"

path_to_run = f"./runs/{run_name}/model/"

path_to_run_125 = f"./runs/{run_name_125}/model/"
path_to_run_375 = f"./runs/{run_name_375}/model/"
path_to_run_75 = f"./runs/{run_name_75}/model/"

# model_files = os.listdir(path_to_run)
# model_files.sort(key=lambda file: int(file.split("_")[-1]))

model_files_125 = os.listdir(path_to_run_125)
model_files_125.sort(key=lambda file: int(file.split("_")[-1]))
model_files_375 = os.listdir(path_to_run_375)
model_files_375.sort(key=lambda file: int(file.split("_")[-1]))
model_files_75 = os.listdir(path_to_run_75)
model_files_75.sort(key=lambda file: int(file.split("_")[-1]))


fig, ax = plt.subplots(figsize=(8, 6))
ax.set_ylabel("Imaginary")
ax.set_xlabel("Real")
ax.set_title("Eigenvalue Plot")
ax.axhline(y=0, color="k", linestyle="--")
ax.axvline(x=0, color="k", linestyle="--")
ax.grid(True)

readout_125 = Readout(L)
readout_375 = Readout(L)
readout_75 = Readout(L)

model_125 = RestrictedDisentangledTransformer(L, mlp=readout_125)
model_375 = RestrictedDisentangledTransformer(L, mlp=readout_375)
model_75 = RestrictedDisentangledTransformer(L, mlp=readout_75)


def get_eigen(model):

    model_params = model.state_dict()

    QK1 = model_params["transformer_block_1.causal_block.W_QK.weight"].detach().numpy()

    matrixB = QK1[model.P : model.P + model.D, 2 * model.P + model.D :]

    evalue, evect = np.linalg.eig(matrixB)

    # extract real part
    real = [ele.real for ele in evalue]
    # extract imaginary part
    imag = [ele.imag for ele in evalue]

    return real, imag


def update(frame):
    # model_file = model_files[frame]

    try:
        model_file_125 = model_files_125[frame]
    except:
        model_file_125 = model_files_125[-1]

    try:
        model_file_375 = model_files_375[frame]
    except:
        model_file_375 = model_files_375[-1]

    try:
        model_file_75 = model_files_75[frame]
    except:
        model_file_75 = model_files_75[-1]

    epoch = model_file_125.split("_")[-1]

    model_125.load_state_dict(torch.load(path_to_run_125 + model_file_125))
    model_375.load_state_dict(torch.load(path_to_run_375 + model_file_375))
    model_75.load_state_dict(torch.load(path_to_run_75 + model_file_75))

    model_125.eval()
    model_375.eval()
    model_75.eval()

    real_125, imag_125 = get_eigen(model_125)
    real_375, imag_375 = get_eigen(model_375)
    real_75, imag_75 = get_eigen(model_75)

    ax.clear()

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])

    ax.set_title(f"Epoch {epoch}")

    ax.scatter(real_125, imag_125, label="p_C=0.125", s=10)
    ax.scatter(real_375, imag_375, label="p_C=0.375", s=10)
    ax.scatter(real_75, imag_75, label="p_C=0.75", s=10)

    ax.legend()


# ani = FuncAnimation(fig, update, frames=len(model_files_125))

# writer = PillowWriter(fps=5, metadata=dict(artist="Me"), bitrate=1800)
# ani.save("eigen_values.gif", writer=writer)
# print("Saved animation as gif")

# plt.show()

plt.close()


latest_125 = model_files_125[-1]
latest_375 = model_files_375[-1]
latest_75 = model_files_75[-1]
epoch = latest_125.split("_")[-1]

model_125.load_state_dict(torch.load(path_to_run_125 + latest_125))
model_375.load_state_dict(torch.load(path_to_run_375 + latest_375))
model_75.load_state_dict(torch.load(path_to_run_75 + latest_75))

model_125.eval()
model_375.eval()
model_75.eval()

real_125, imag_125 = get_eigen(model_125)
real_375, imag_375 = get_eigen(model_375)
real_75, imag_75 = get_eigen(model_75)


plt.figure()

plt.scatter(real_125, imag_125, label="p_C=0.125", s=10)
plt.scatter(real_375, imag_375, label="p_C=0.375", s=10)
plt.scatter(real_75, imag_75, label="p_C=0.75", s=10)

plt.legend()
plt.show()
