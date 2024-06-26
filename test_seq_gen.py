from data_old import SeqGenerator
from util import seq_vis
import numpy as np

seq_gen = SeqGenerator()


for (
    context,
    context_labels,
    label_inds,
    query_vector,
    query_label,
    query_label_ind,
) in seq_gen.generate_iwl_seq():
    print(context)
    print(context_labels)
    print(label_inds)
    seq_vis(
        np.concatenate([context, query_vector], axis=1),
        np.concatenate([label_inds, query_label_ind]),
    )
