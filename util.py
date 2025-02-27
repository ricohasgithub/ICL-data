import matplotlib.pyplot as plt
import seaborn
import imageio
import os
import numpy as np


def seq_vis(seq_vectors, seq_labels):

    plt.figure()

    seaborn.heatmap(seq_vectors, xticklabels=seq_labels)

    plt.show()


def attention_map_vis(
    attention_matrix, path, classes=None, layer=0, vis_mode=-1, epoch=-1
):
    attention_matrix = attention_matrix.cpu().detach().numpy()
    attention_matrix = attention_matrix[0][0]

    plt.figure()
    plt.title(f"{epoch} Attention Map")

    if classes == None:
        seaborn.heatmap(attention_matrix)
    else:
        seaborn.heatmap(attention_matrix, xticklabels=classes, yticklabels=classes)

    if vis_mode == 0:
        plt.savefig(f"./runs/{path}/train/layer{layer}/attn_map_{epoch}.png", dpi=300, bbox_inches="tight")
    elif vis_mode == 1:
        plt.savefig(f"./runs/{path}/icl1/layer{layer}/attn_map_{epoch}.png", dpi=300, bbox_inches="tight")
    elif vis_mode == 2:
        plt.savefig(f"./runs/{path}/icl2/layer{layer}/attn_map_{epoch}.png", dpi=300, bbox_inches="tight")
    elif vis_mode == 3:
        plt.savefig(f"./runs/{path}/iwl/layer{layer}/attn_map_{epoch}.png", dpi=300, bbox_inches="tight")

    plt.close()

def plot_wo(wo, path, epoch=-1):
    try:
        os.makedirs(f"{path}w_o/")
    except:
        print(path + " already exists.")
    seaborn.heatmap(wo)
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

def vis_attention_weights(
    layer0_weights,
    layer1_weights,
    out_weights,
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

    seaborn.heatmap(layer0_weights)

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
    seaborn.heatmap(layer1_weights)

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
    seaborn.heatmap(out_weights)

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
    seaborn.heatmap(np.diag(S))

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
    seaborn.heatmap(U)

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
    seaborn.heatmap(Vh)

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

    path = f"./disentangled_model_plots/{run_name}/"

    if layer == "out":
        image_path = f"./disentangled_model_plots/{run_name}/out/"
    else:
        image_path = f"./disentangled_model_plots/{run_name}/layer{layer}/"

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
