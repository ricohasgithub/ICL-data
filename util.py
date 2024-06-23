import matplotlib.pyplot as plt
import seaborn


def seq_vis(seq_vectors, seq_labels):

    plt.figure()

    seaborn.heatmap(seq_vectors, xticklabels=seq_labels)

    plt.show()
