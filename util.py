import matplotlib.pyplot as plt
import seaborn as sns
import imageio
import os
import numpy as np


def seq_vis(seq_vectors, seq_labels):

    plt.figure()

    sns.heatmap(seq_vectors, xticklabels=seq_labels)

    plt.show()


def attention_map_vis(
    attention_matrix, path, classes=None, layer=0, vis_mode=-1, epoch=-1
):
    attention_matrix = attention_matrix.cpu().detach().numpy()
    attention_matrix = attention_matrix[0][0]

    plt.figure()
    plt.title(f"{epoch} Attention Map")

    if classes == None:
        sns.heatmap(attention_matrix)
    else:
        sns.heatmap(attention_matrix, xticklabels=classes, yticklabels=classes)

    if vis_mode == 0:
        plt.savefig(
            f"./runs/{path}/train/layer{layer}/attn_map_{epoch}.png",
            dpi=300,
            bbox_inches="tight",
        )
    elif vis_mode == 1:
        plt.savefig(
            f"./runs/{path}/icl1/layer{layer}/attn_map_{epoch}.png",
            dpi=300,
            bbox_inches="tight",
        )
    elif vis_mode == 2:
        plt.savefig(
            f"./runs/{path}/icl2/layer{layer}/attn_map_{epoch}.png",
            dpi=300,
            bbox_inches="tight",
        )
    elif vis_mode == 3:
        plt.savefig(
            f"./runs/{path}/iwl/layer{layer}/attn_map_{epoch}.png",
            dpi=300,
            bbox_inches="tight",
        )

    plt.close()


def plot_wo(wo, path, epoch=-1):
    try:
        os.makedirs(f"{path}w_o/")
    except:
        print(path + " already exists.")
    sns.heatmap(wo)
    plt.savefig(path + f"w_o/w_o{epoch}_weights_vis.png")
    plt.close()


def gen_attention_map_gif(path, vis_mode=-1, layer=-1):
    folder = None
    if vis_mode == 0:
        folder = "train"
    elif vis_mode == 1:
        folder = "icl1"
    elif vis_mode == 2:
        folder = "icl2"
    elif vis_mode == 3:
        folder = "iwl"

    images = []
    files = os.listdir(f"./runs/{path}/{folder}/layer{layer}/")
    files = [file for file in files if file.endswith(".png")]
    files.sort(key=lambda file: int(file[9 : file.index(".")]))

    for filename in files:
        if filename.endswith(".png"):
            # print(filename)
            images.append(
                imageio.imread(f"./runs/{path}/{folder}/layer{layer}/{filename}")
            )

    imageio.mimsave(f"./runs/{path}/{folder}/layer{layer}.gif", images, duration=0.25)


def create_image_gif_folder_structure(run_name):

    try:
        os.makedirs(f"./runs/{run_name}/icl1/layer0/")
        os.makedirs(f"./runs/{run_name}/icl1/layer1/")
        os.makedirs(f"./runs/{run_name}/icl2/layer0/")
        os.makedirs(f"./runs/{run_name}/icl2/layer1/")
        os.makedirs(f"./runs/{run_name}/iwl/layer0/")
        os.makedirs(f"./runs/{run_name}/iwl/layer1/")
        os.makedirs(f"./runs/{run_name}/model/")
        os.makedirs(f"./runs/{run_name}/output/")
    except:
        print(f"{run_name} already exists.")


# def vis_attention_weights(
#     layer0_weights,
#     layer1_weights,
#     out_weights,
#     P,
#     D,
#     save_dir="",
#     model_name="",
#     hyper_params=None,
# ):
#     p_B = -1
#     p_C = -1
#     if hyper_params:
#         p_B = hyper_params.get("p_B", p_B)
#         p_C = hyper_params.get("p_C", p_C)

#     ###############################
#     # Visualize Layer 0 Weights
#     # layer0_weights is a (P+D) x (P+D) matrix
#     ###############################
#     fig, axs = plt.subplots(2, 2, figsize=(10, 8))
#     fig.suptitle(
#         f"model={model_name} p_B={p_B} p_C={p_C} Layer 0 - Block Weights (P & D)"
#     )

#     # Block: Top-left (P -> P)
#     axs[0, 0].set_title("P → P")
#     sns.heatmap(layer0_weights[:P, :P], ax=axs[0, 0], cbar=False)

#     # Block: Top-right (P -> D)
#     axs[0, 1].set_title("P → D")
#     sns.heatmap(layer0_weights[:P, P : P + D], ax=axs[0, 1], cbar=False)

#     # Block: Bottom-left (D -> P)
#     axs[1, 0].set_title("D → P")
#     sns.heatmap(layer0_weights[P : P + D, :P], ax=axs[1, 0], cbar=False)

#     # Block: Bottom-right (D -> D)
#     axs[1, 1].set_title("D → D")
#     sns.heatmap(layer0_weights[P : P + D, P : P + D], ax=axs[1, 1], cbar=False)

#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     if save_dir:
#         os.makedirs(os.path.join(save_dir, "layer0_blocks"), exist_ok=True)
#         fig.savefig(
#             os.path.join(
#                 save_dir, "layer0_blocks", f"{model_name}_layer0_weights_blocks.png"
#             )
#         )
#     else:
#         plt.show()
#     plt.close(fig)

#     ###############################
#     # Visualize Layer 1 Weights
#     # layer1_weights is a 2(P+D) x 2(P+D) matrix.
#     # We split the rows (and similarly the columns) into four slices:
#     #   - 0 to P           : first half, P
#     #   - P to P+D         : first half, D
#     #   - P+D to 2P+D      : second half, P
#     #   - 2P+D to 2(P+D)   : second half, D
#     ###############################
#     fig, axs = plt.subplots(4, 4, figsize=(16, 16))
#     fig.suptitle(
#         f"model={model_name} p_B={p_B} p_C={p_C} Layer 1 - Block Weights (P & D)"
#     )

#     # Define slicing indices for rows and columns
#     row_indices = [0, P, P + D, 2 * P + D, 2 * (P + D)]
#     col_indices = row_indices  # same split for columns

#     # Loop over the 4x4 grid

#     for i in range(4):
#         for j in range(4):
#             r_start, r_end = row_indices[i], row_indices[i + 1]
#             c_start, c_end = col_indices[j], col_indices[j + 1]
#             # Determine labels: even index -> "P", odd index -> "D"
#             row_label = "P" if i % 2 == 0 else "D"
#             col_label = "P" if j % 2 == 0 else "D"
#             axs[i, j].set_title(f"{row_label} → {col_label}", fontsize=10)
#             sns.heatmap(
#                 layer1_weights[r_start:r_end, c_start:c_end],
#                 ax=axs[i, j],
#                 cbar=False,
#                 xticklabels=False,
#                 yticklabels=False,
#             )
#     # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     if save_dir:
#         os.makedirs(os.path.join(save_dir, "layer1_blocks"), exist_ok=True)
#         fig.savefig(
#             os.path.join(
#                 save_dir, "layer1_blocks", f"{model_name}_layer1_weights_blocks.png"
#             )
#         )
#     else:
#         plt.show()
#     plt.close(fig)

#     ###############################
#     # Visualize Output Weights (W_O)
#     # Assume out_weights is also a (P+D) x (P+D) matrix.
#     ###############################
#     plt.figure()

#     plt.title(f"model={model_name} p_B={p_B} p_C={p_C} W_O")
#     sns.heatmap(out_weights)

#     if save_dir == "":
#         plt.show()
#     else:

#         try:
#             os.makedirs(f"{save_dir}out/")
#         except:
#             print(save_dir + " already exists.")

#         plt.savefig(save_dir + f"out/{model_name}_out_weights_vis.png")

#     plt.close()


def vis_attention_weights(
    layer0_weights,
    layer1_weights,
    out_weights,
    P,
    D,
    save_dir="",
    model_name="",
    hyper_params=None,
):
    plt.figure()

    p_B = -1
    p_C = -1
    if hyper_params:
        p_B = hyper_params["p_B"]
        p_C = hyper_params["p_C"]

    plt.title(f"model={model_name} p_B={p_B} p_C={p_C} Layer 1")

    sns.heatmap(layer0_weights)

    if save_dir == "":
        plt.show()
    else:

        try:
            os.makedirs(f"{save_dir}layer1/")
        except:
            print(save_dir + " already exists.")

        plt.savefig(save_dir + f"layer1/{model_name}_layer1_weights_vis.png")

    plt.close()

    plt.figure()

    plt.title(f"model={model_name} p_B={p_B} p_C={p_C} Layer 2")
    sns.heatmap(layer1_weights)

    if save_dir == "":
        plt.show()
    else:

        try:
            os.makedirs(f"{save_dir}layer2/")
        except:
            print(save_dir + " already exists.")

        plt.savefig(save_dir + f"layer2/{model_name}_layer2_weights_vis.png")

    plt.close()

    plt.figure()

    plt.title(f"model={model_name} p_B={p_B} p_C={p_C} W_O")
    sns.heatmap(out_weights)

    if save_dir == "":
        plt.show()
    else:

        try:
            os.makedirs(f"{save_dir}out/")
        except:
            print(save_dir + " already exists.")

        plt.savefig(save_dir + f"out/{model_name}_out_weights_vis.png")

    plt.close()


def svd_attention_weights(
    weights, layer, save_file_name="", save_dir="", model_name="", hyper_params=None
):

    p_B = -1
    p_C = -1
    if hyper_params:
        p_B = hyper_params["p_B"]
        p_C = hyper_params["p_C"]

    U, S, Vh = np.linalg.svd(weights, full_matrices=False)

    plt.figure()

    plt.title(f"p_B={p_B} p_C={p_C} Layer 2")
    sns.heatmap(np.diag(S))

    if save_dir == "":
        plt.show()
    else:

        try:
            os.makedirs(f"{save_dir}layer{layer}_svd/")
        except:
            print(save_dir + " already exists.")

        plt.savefig(save_dir + f"layer{layer}_svd/{save_file_name}_S.png")

    plt.close()

    plt.figure()

    plt.title(f"p_B={p_B} p_C={p_C} Layer 2")
    sns.heatmap(U)

    if save_dir == "":
        plt.show()
    else:

        try:
            os.makedirs(f"{save_dir}layer{layer}_svd/")
        except:
            print(save_dir + " already exists.")

        plt.savefig(save_dir + f"layer{layer}_svd/{save_file_name}_U.png")

    plt.close()

    plt.figure()

    plt.title(f"p_B={p_B} p_C={p_C} Layer 2")
    sns.heatmap(Vh)

    if save_dir == "":
        plt.show()
    else:

        try:
            os.makedirs(f"{save_dir}layer{layer}_svd/")
        except:
            print(save_dir + " already exists.")

        plt.savefig(save_dir + f"layer{layer}_svd/{save_file_name}_Vh.png")

    plt.close()


def gen_qk_weights_gif(run_name, layer):

    images = []

    # path = f"./disentangled_model_plots/{run_name}/"
    path = f"./circuit_plots/{run_name}/"

    if layer == "out":
        # image_path = f"./disentangled_model_plots/{run_name}/out/"
        image_path = f"./circuit_plots/{run_name}/out/"
    else:
        # image_path = f"./disentangled_model_plots/{run_name}/layer{layer}/"
        image_path = f"./circuit_plots/{run_name}/layer{layer}/"

    files = os.listdir(image_path)
    files = [file for file in files if file.endswith(".png")]
    files.sort(key=lambda file: int(file.split("_")[1]))

    for filename in files:
        if filename.endswith(".png"):
            # print(filename)
            images.append(imageio.imread(f"{image_path}{filename}"))

    imageio.mimsave(
        f"{path}layer{layer}_weights.gif",
        images,
        fps=10,
    )


def vis_output(run_name, logits, labels, inputs, epoch):

    logits = logits.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    inputs = inputs.cpu().detach().numpy()

    first_input = inputs[0]
    first_batch_logits = logits[0]
    first_batch_label = np.argmax(labels[0])

    labels_mask = (np.arange(len(input_labels)) % 2) == 1
    input_labels = first_input[labels_mask]

    plt.figure()

    plt.bar(
        [_ for _ in range(len(first_batch_logits))],
        first_batch_logits,
        color=[
            "red" if i == first_batch_label else "blue"
            for i in range(len(first_batch_logits))
        ],
    )

    plt.savefig(f"./runs/{run_name}/output/{epoch}.png")

    plt.close()
