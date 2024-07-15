import torch
import seaborn
import matplotlib.pyplot as plt
import os
from transformer import DisentangledTransformer, Readout
from util import vis_attention_weights

use_mlp = False
K = 512
L = 32
p_B = 0.375
p_C = 0.375
B = 1
eps = 0
uuid = "7d4a307b-c773-46fd-ae74-59d97ccde72c"

if use_mlp:
    last_layer = "MLP"
else:
    last_layer = "Readout"

run_name = f"{last_layer}|K={K}|L={L}|p_B={p_B}|p_C={p_C}|B={B}|eps={eps}|{uuid}"

path_to_run = f"./runs/{run_name}/model/"

model_files = os.listdir(path_to_run)
model_files.sort(key=lambda file: int(file.split("_")[-1]), reverse=True)

latest_model = model_files[0]

print("Loading " + path_to_run + latest_model + "...")

if use_mlp:
    model = DisentangledTransformer(L)
else:
    readout = Readout(L)
    model = DisentangledTransformer(L, mlp=readout)

model.load_state_dict(torch.load(path_to_run + latest_model))

model.eval()

model_params = model.state_dict()

QK0 = model_params["transformer_block_0.causal_block.W_QK.weight"]
QK1 = model_params["transformer_block_1.causal_block.W_QK.weight"]

vis_attention_weights(
    QK0.detach().numpy(),
    QK1.detach().numpy(),
    save_dir="./disentangled_model_plots/" + run_name + "/",
    model_name=latest_model,
)
