import matplotlib.pyplot as plt
import seaborn
import imageio
import os


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
        plt.savefig(f"./runs/{path}/train/layer{layer}/attn_map_{epoch}.png")
    elif vis_mode == 1:
        plt.savefig(f"./runs/{path}/icl1/layer{layer}/attn_map_{epoch}.png")
    elif vis_mode == 2:
        plt.savefig(f"./runs/{path}/icl2/layer{layer}/attn_map_{epoch}.png")
    elif vis_mode == 3:
        plt.savefig(f"./runs/{path}/iwl/layer{layer}/attn_map_{epoch}.png")

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
    except:
        print(f"{run_name} already exists.")


def vis_attention_weights(layer0_weights, layer1_weights, save_dir="", model_name=""):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)

    plt.title("Layer 1 Weights")

    seaborn.heatmap(layer0_weights)

    plt.subplot(1, 2, 2)

    plt.title("Layer 2 Weights")
    seaborn.heatmap(layer1_weights)

    if save_dir == "":
        plt.show()
    else:

        try:
            os.makedirs(f"{save_dir}")
        except:
            print(save_dir + " already exists.")

        plt.savefig(save_dir + f"{model_name}_weights_vis.png")

    plt.close()
