import matplotlib.pyplot as plt
import seaborn


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
