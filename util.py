import matplotlib.pyplot as plt
import seaborn
import imageio
import os


def seq_vis(seq_vectors, seq_labels):

    plt.figure()

    seaborn.heatmap(seq_vectors, xticklabels=seq_labels)

    plt.show()


def attention_map_vis(attention_matrix, classes=None, layer=0, vis_mode=-1, epoch=-1):
    attention_matrix = attention_matrix.cpu().detach().numpy()
    attention_matrix = attention_matrix[0][0]

    plt.figure()
    plt.title("Attention Map")

    if classes == None:
        seaborn.heatmap(attention_matrix)
    else:
        seaborn.heatmap(attention_matrix, xticklabels=classes, yticklabels=classes)

    if vis_mode == 0:
        plt.savefig(f"./train_attention_vis/layer{layer}/attn_map_{epoch}.png")
    elif vis_mode == 1:
        plt.savefig(f"./ic1_attention_vis/layer{layer}/attn_map_{epoch}.png")
    elif vis_mode == 2:
        plt.savefig(f"./ic2_attention_vis/layer{layer}/attn_map_{epoch}.png")
    elif vis_mode == 3:
        plt.savefig(f"./iw_attention_vis/layer{layer}/attn_map_{epoch}.png")


def gen_attention_map_gif(vis_mode=-1, layer=-1):
    folder = None
    if vis_mode == 0:
        folder = "train_attention_vis"
    elif vis_mode == 1:
        folder = "ic1_attention_vis"
    elif vis_mode == 2:
        folder = "ic2_attention_vis"
    elif vis_mode == 3:
        folder = "iw_attention_vis"

    images = []
    files = os.listdir(f"./{folder}/layer{layer}/")
    files.sort()

    for filename in files:
        if filename.endswith(".png"):
            images.append(imageio.imread(f"./{folder}/layer{layer}/{filename}"))

    imageio.mimsave(f"./{folder}/layer{layer}.gif", images, duration=0.5)
