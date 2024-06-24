import random
import logging
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import Omniglot
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset

IMAGE_SIZE = 105
N_CHARACTER_CLASSES = 1623
N_EXEMPLARS_PER_CLASS = 20


class GaussianVectorGenerator:
    def __init__(self, num_classes, num_labels, input_dim, noise_param=0.75):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_labels = num_labels
        self.class_center_mean = np.zeros(shape=(input_dim))
        self.class_center_var = (1 / input_dim) * np.identity(input_dim)

        # Generate ranodm c
        self.class_means = np.stack(
            [
                np.random.multivariate_normal(
                    self.class_center_mean, self.class_center_var
                )
                for _ in range(num_classes)
            ],
            axis=0,
        )

        self.class_labels = np.stack(
            [
                np.random.multivariate_normal(
                    self.class_center_mean, self.class_center_var
                )
                for _ in range(num_labels)
            ],
            axis=0,
        )

        self.noise_param = noise_param  # epsilon for in

        self.classes_for = [[] for _ in range(num_labels)]

        self.label_to_class = [[] for _ in range(num_labels)]

        for class_num in range(num_classes):
            self.label_to_class[class_num % num_labels].append(class_num)

    def generateVectors(self, class_num, num=1, new_classes=False, new_labels=False):

        class_centers = self.class_means
        class_labels = self.class_labels
        if new_classes:
            class_centers = np.stack(
                [
                    np.random.multivariate_normal(
                        self.class_center_mean, self.class_center_var
                    )
                    for _ in range(self.num_classes)
                ],
                axis=0,
            )

        if new_labels:
            class_labels = np.stack(
                [
                    np.random.multivariate_normal(
                        self.class_center_mean, self.class_center_var
                    )
                    for _ in range(self.num_labels)
                ],
                axis=0,
            )

        class_center = class_centers[class_num].reshape(self.input_dim, 1)
        noise_vectors = self.noise_param * np.random.multivariate_normal(
            self.class_center_mean, self.class_center_var, size=num
        )

        noise_vectors = np.transpose(noise_vectors, axes=[1, 0])

        class_vector = (class_center + noise_vectors) * (
            1 / np.sqrt(1 + self.noise_param**2)
        )

        class_label = class_labels[class_num % self.num_labels]

        return class_vector, class_label

    def get_class_centers(self, num=1):
        return np.random.multivariate_normal(
            self.class_center_mean, self.class_center_var, size=num
        )


class SeqGenerator:
    """Generates sequences of 'common', 'rare', or Zipf-distributed classes."""

    def __init__(
        self,
        num_classes=1024,  # K
        num_labels=32,  # L
        input_dim=63,  # D
        seq_len=8,  # N
        noise_param=0.75,  # e
        zipf_exp=0,
    ):
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.num_labels = num_labels
        self.data_generator = GaussianVectorGenerator(
            num_classes, num_labels, input_dim, noise_param
        )

        zipfian_dist = np.power(np.arange(num_labels), -zipf_exp)
        zipfian_dist = zipfian_dist / np.sum(zipfian_dist)

        self.random_label_prob = zipfian_dist

    def generate_train_seq(self, p_bursty=1, burst_num=1):
        while True:
            if np.random.rand() < p_bursty:
                yield self.get_bursty_seq(burst_num)
            else:
                yield self.get_random_seq()

    def generate_icl_seq(self, burst_num=1, mode="icl1"):
        while True:
            yield self.get_bursty_seq(burst_num, mode)

    def generate_iwl_seq(self):
        while True:
            yield self.get_random_seq(iwl=True)

    def get_bursty_seq(self, burst_num=1, mode="train"):  # B  # "train", "icl1", "icl2"

        assert (self.seq_len % burst_num) == 0

        num_label_in_seq = self.seq_len // burst_num

        label_indices = np.repeat(np.arange(num_label_in_seq), burst_num)

        np.random.shuffle(label_indices)

        context_label_inds = np.random.choice(
            len(self.data_generator.class_labels), size=num_label_in_seq, replace=False
        )

        context = np.zeros((self.data_generator.input_dim, self.seq_len))
        context_labels = np.zeros((self.data_generator.input_dim, self.seq_len))

        context_classes = np.zeros(self.seq_len)

        for label_ind in range(num_label_in_seq):

            label_num = context_label_inds[label_ind]

            classes_for_label = np.random.choice(
                self.data_generator.label_to_class[label_num], burst_num
            )

            label_vectors = np.zeros((self.data_generator.input_dim, burst_num))

            class_label = None
            for class_num in np.unique(classes_for_label):

                if mode == "icl1":
                    class_vectors, cur_class_label = (
                        self.data_generator.generateVectors(
                            class_num,
                            num=np.sum(classes_for_label == class_num),
                            new_classes=True,
                        )
                    )
                elif mode == "icl2":
                    class_vectors, cur_class_label = (
                        self.data_generator.generateVectors(
                            class_num,
                            num=np.sum(classes_for_label == class_num),
                            new_classes=True,
                            new_labels=True,
                        )
                    )
                else:
                    class_vectors, cur_class_label = (
                        self.data_generator.generateVectors(
                            class_num, num=np.sum(classes_for_label == class_num)
                        )
                    )

                class_label = cur_class_label
                label_vectors[:, classes_for_label == class_num] = class_vectors

            context_labels[:, label_indices == label_ind] = class_label.reshape(
                self.input_dim, 1
            )

            context_classes[label_indices == label_ind] = classes_for_label
            context[:, label_indices == label_ind] = label_vectors

        query_class = np.random.choice(np.unique(context_classes))
        query_class = int(query_class)

        if mode == "icl1":
            query_vector, _ = self.data_generator.generateVectors(
                query_class, new_classes=True
            )

            query_label = np.unique(
                context_labels[:, context_classes == query_class], axis=1
            )
        elif mode == "icl2":
            query_vector, _ = self.data_generator.generateVectors(
                query_class, new_classes=True, new_labels=True
            )
            query_label = np.unique(
                context_labels[:, context_classes == query_class], axis=1
            )
        else:
            query_vector, query_label = self.data_generator.generateVectors(query_class)

        query_label_ind = context_label_inds[
            np.unique(label_indices[context_classes == query_class])
        ]
        return (
            context,
            context_labels,
            context_label_inds[label_indices],
            query_vector,
            query_label.reshape(self.input_dim, 1),
            query_label_ind,
        )

    def get_random_seq(self, iwl=False):
        context_label_inds = np.random.choice(
            np.arange(self.num_labels), self.seq_len, p=self.random_label_prob
        )
        context_classes = np.zeros(self.seq_len)
        for label_num in np.unique(context_label_inds):
            classes_for_label = np.random.choice(
                self.data_generator.label_to_class[label_num],
                np.sum(context_label_inds == label_num),
            )

            context_classes[context_label_inds == label_num] = classes_for_label

        context = np.zeros((self.data_generator.input_dim, self.seq_len))
        context_labels = np.zeros((self.data_generator.input_dim, self.seq_len))

        for class_num in np.unique(context_classes):
            class_vectors, class_label = self.data_generator.generateVectors(
                int(class_num), num=np.sum(context_classes == class_num)
            )

            context[:, context_classes == class_num] = class_vectors
            context_labels[:, context_classes == class_num] = class_label.reshape(
                self.input_dim, 1
            )

        if iwl:
            query_label_ind = np.random.choice(
                np.arange(self.num_labels), p=self.random_label_prob
            )
            query_class = np.random.choice(
                self.data_generator.label_to_class[query_label_ind]
            )
            query_label_ind = [query_label_ind]
        else:
            query_class = np.random.choice(np.unique(context_classes))
            query_label_ind = np.unique(
                context_label_inds[context_classes == query_class]
            )

        query_class = int(query_class)
        query_vector, query_label = self.data_generator.generateVectors(query_class)

        return (
            context,
            context_labels,
            context_label_inds,
            query_vector,
            query_label.reshape(self.input_dim, 1),
            query_label_ind,
        )


def _convert_dict(
    example, use_constant_labels=False, interleave_targets=True, downsample=False
):
    # (dims: B:batch, SS:original seqlen, H:height, W:width, C:channels)
    is_image = len(example["example"].shape) == 5
    is_vector = len(example["example"].shape) == 4

    # Cast the examples into the correct shape and tf datatype.
    if is_image or is_vector:
        examples = example["example"].type(torch.float)  # (B,SS,H,W,C)
        # if downsample:
        #     examples = tf.map_fn(
        #         lambda batch: tf.image.resize(batch, [28, 28]), examples
        #     )
    else:
        examples = example["example"].type(torch.int32)  # (B, SS)

    # Cast the labels into the correct tf datatype.
    if use_constant_labels:
        labels = torch.ones_like(example["label"], dtype=torch.int32)
    else:
        labels = example["label"].type(torch.int32)  # (B,SS)
    seq_len = labels.shape[-1]

    # Create the target sequence.
    if interleave_targets:
        # Alternating labels with zeros, e.g. [label, 0, label, 0, ...].
        zeros = torch.zeros_like(labels)
        target = torch.stack((labels[..., None], zeros[..., None]), axis=-1)
        target = torch.reshape(target, [-1, seq_len * 2])[:, :-1]  # (B,SS*2-1)
    else:
        # Just use the original sequence of labels, e.g. [label, label, ...]
        target = labels  # (B,SS)

    ret_dict = {"examples": examples, "labels": labels, "target": target}
    return ret_dict
