import matplotlib.pyplot as plt
import seaborn


def seq_vis(seq_vectors, seq_labels):

    plt.figure()

    seaborn.heatmap(seq_vectors, xticklabels=seq_labels)

    plt.show()

def attention_map_vis(attention_matrix, classes=None, layer=0, vis_mode=-1):
    attention_matrix = attention_matrix.cpu().detach().numpy()
    attention_matrix = attention_matrix[0][0]

    plt.figure()
    plt.title("Attention Map")

    if classes == None:
        seaborn.heatmap(attention_matrix)
    else:
        seaborn.heatmap(attention_matrix, xticklabels=classes, yticklabels=classes)

    if vis_mode == 0:
        plt.savefig(f"./attn_map_{layer}_train.png")
    elif vis_mode == 1:
        plt.savefig(f"./attn_map_{layer}_IC1.png")
    elif vis_mode == 2:
        plt.savefig(f"./attn_map_{layer}_IC2.png")
    elif vis_mode == 3:
        plt.savefig(f"./attn_map_{layer}_IW.png")