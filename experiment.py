import sys

from train import Trainer
from data import SeqGenerator, GaussianVectorGenerator

from transformer import Transformer, SequenceEmbedder


import torch


def experiment_base(dataset, p_bursty, zipf_exp):

    if dataset == "synthetic":
        input_embedding = InputEmbedder(linear_input_dim=64, example_encoding="linear")
        seq_generator_factory = SeqGenerator(
            dataset_for_sampling=GaussianVectorDatasetForSampling(),
            n_rare_classes=1603,  # 1623 - 20
            n_common_classes=10,
            n_holdout_classes=10,
            zipf_exponent=zipf_exp,
            use_zipf_for_common_rare=False,
            noise_scale=0.0,
            preserve_ordering_every_n=None,
        )
    elif dataset == "omniglot":
        input_embedding = InputEmbedder(
            linear_input_dim=11025, example_encoding="resnet"
        )
        seq_generator_factory = SeqGenerator(
            dataset_for_sampling=OmniglotDatasetForSampling(
                omniglot_split="all",  # 1623 total classes
                exemplars="all",  # 'single' / 'separated' / 'all'
                augment_images=False,
            ),
            n_rare_classes=1603,  # 1623 - 20
            n_common_classes=10,
            n_holdout_classes=10,
            zipf_exponent=1,
            use_zipf_for_common_rare=False,
            noise_scale=0.0,
            preserve_ordering_every_n=None,
        )

    seq_generator_factory = SeqGenerator(
        data_generator=GaussianVectorGenerator(num_classes=32, input_dim=63),
        n_rare_classes=1603,  # 1623 - 20
        n_common_classes=10,
        n_holdout_classes=10,
        zipf_exponent=1,
        use_zipf_for_common_rare=False,
        noise_scale=0.0,
        preserve_ordering_every_n=None,
    )
    input_embedding = SequenceEmbedder(65, 63, 8)
    model = Transformer(input_embedder=input_embedding)

    """ (cfg.seq_len, cfg.bursty_shots, cfg.ways, cfg.p_bursty,
                        cfg.p_bursty_common, cfg.p_bursty_zipfian,
                        cfg.non_bursty_type, cfg.labeling_common,
                        cfg.labeling_rare, cfg.randomly_generate_rare,
                        cfg.grouped"""

    data_generator = lambda: seq_generator_factory.get_bursty_seq(
        seq_len=9,
        shots=3,
        ways=2,
        p_bursty=p_bursty,
        p_bursty_common=0,
        p_bursty_zipfian=1,
        non_bursty_type="zipfian",
        labeling_common="ordered",
        labeling_rare="ordered",
        randomly_generate_rare=False,
        grouped=False,
    )

    trainer = Trainer(
        model,
        data_generator,
        seq_generator_factory,
        p_bursty=p_bursty,
        dataset_name=dataset,
        zipf_exponent=zipf_exp,
    )
    torch.autograd.set_detect_anomaly(True)
    trainer.train()


if __name__ == "__main__":

    # Read command line inputs to decide which experiment to run
    experiment_id = sys.argv[1]
    dataset = sys.argv[2]
    p_bursty = float(sys.argv[3])
    zipf_exp = float(sys.argv[4])

    if experiment_id == "base":
        experiment_base(dataset, p_bursty, zipf_exp)
    elif experiment_id == "mixed":
        pass
