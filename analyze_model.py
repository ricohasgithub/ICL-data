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

use_mlp = False
use_disentangled = True
K = 512
L = 32
p_B = 0.375
p_C = 0.375
B = 1
eps = 0
uuid = "4bb91b89-d345-4eac-8cfc-f55b9ce9d39a"

if use_mlp:
    last_layer = "MLP"
else:
    last_layer = "Readout"

if use_disentangled:
    model_type = "Disentangled"
else:
    model_type = "Transformer"

# run_name = (
#     f"{model_type}|{last_layer}|K={K}|L={L}|p_B={p_B}|p_C={p_C}|B={B}|eps={eps}|{uuid}"
# )

run_name = "Disentangled|Readout|K=512|L=32|epochs_icl=500|epochs_iwl=500|B=1|eps=0|b6a19e6d-0475-4cba-8b87-15b05c48dad6"

with open(f"./runs/{run_name}/mus_label.npy", "rb") as f:
    mus_label = np.load(f)
with open(f"./runs/{run_name}/mus_class.npy", "rb") as f:
    mus_class = np.load(f)
with open(f"./runs/{run_name}/labels_class.npy", "rb") as f:
    labels_class = np.load(f)

mus = np.concatenate([mus_label, mus_class], axis=0)
combined_mus = 0.5 * mus_label[:, None, :] + 0.5 * mus_class[None, :, :]
combined_mus = combined_mus.reshape(-1, mus.shape[1])

path_to_run = f"./runs/{run_name}/model/"

model_files = os.listdir(path_to_run)
model_files.sort(key=lambda file: int(file.split("_")[-1]), reverse=True)

# latest_model = model_files[0]
latest_model = "model_150"

print("Loading " + path_to_run + latest_model + "...")

if use_mlp:
    model = RestrictedDisentangledTransformer(L)
else:
    readout = Readout(L)
    model = RestrictedDisentangledTransformer(L, mlp=readout)

model.load_state_dict(torch.load(path_to_run + latest_model))

model.eval()

model_params = model.state_dict()

if use_disentangled:
    QK0 = model_params["transformer_block_0.causal_block.W_QK.weight"]
    QK1 = model_params["transformer_block_1.causal_block.W_QK.weight"]

    V0 = model_params["transformer_block_0.causal_block.W_V.weight"]
    V1 = model_params["transformer_block_1.causal_block.W_V.weight"]

else:
    Q0 = model_params["transformer_block_0.causal_block.W_Q.weight"]
    K0 = model_params["transformer_block_0.causal_block.W_K.weight"]
    QK0 = torch.matmul(Q0, K0.t())

    Q1 = model_params["transformer_block_1.causal_block.W_Q.weight"]
    K1 = model_params["transformer_block_1.causal_block.W_K.weight"]
    QK1 = torch.matmul(Q1, K1.t())

    V0 = model_params["transformer_block_0.causal_block.W_V.weight"]
    V1 = model_params["transformer_block_1.causal_block.W_V.weight"]

W_O = model_params["W_O.weight"]

token_to_token_submatrix = QK1[model.P : model.P + model.D, model.P : model.P + model.D]
combined_token_to_token_submatrix = QK1[
    model.P : model.P + model.D, 2 * model.P + model.D :
]

token_to_token = np.matmul(
    mus, np.matmul(token_to_token_submatrix, mus_class.transpose())
)
combined_token_to_token = np.matmul(
    combined_mus, np.matmul(combined_token_to_token_submatrix, mus_class.transpose())
)

combined_token_to_token = combined_token_to_token.transpose(0, 1)


U, S, Vh = np.linalg.svd(combined_token_to_token_submatrix, full_matrices=False)

U_label = np.max(np.matmul(mus_label, U), axis=0)
Vh_label = np.max(np.matmul(Vh, mus_label.transpose()), axis=1)

iwl_out = W_O[:L, model.P : model.P + model.D]
icl_out = W_O[:L, 3 * model.P + 2 * model.D : 3 * model.P + 3 * model.D]

icl_res = np.matmul(icl_out, mus_label.transpose())

class_centroids = []

for label in range(L):
    class_centers = mus_class[labels_class == label]

    class_centroid = np.mean(class_centers, axis=0).reshape(-1, 1)

    class_centroids.append(class_centroid)
class_centroids = np.stack(class_centroids, axis=0)[:, :, 0]

iwl_res = np.matmul(iwl_out, class_centroids.transpose())

plt.figure()

plt.subplot(1, 2, 1)
plt.title("IWL Plot: Z_1 * Class Centroids")


seaborn.heatmap(iwl_res)

plt.subplot(1, 2, 2)
plt.title("ICL Plot: Z_2 * Label Embeddings")

seaborn.heatmap(icl_res)

plt.show()
