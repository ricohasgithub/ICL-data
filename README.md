# In-Context Learning Experiments
This repository contains experiments to evaluate when transformers exhibit in-context learning (ICL) capabilities, and how expressive ICL is. Repo contains a pytorch implementation of the experiments described in "The Mechanistic Basis of Data Dependence and Abrupt Learning in an In-Context Classification Task" ([Reddy, 2024](https://arxiv.org/pdf/2312.03002)).

## About ICL

Given a function class $\mathcal{H}$ with distribution $\mathcal{D}_ \mathcal{H}$ and data space $\mathcal{X}$ with distribution $\mathcal{D}_ \mathcal{X}$, in-context learning describes the process of learning to predict $h(x_{query})$ given the prompt

$$ S=(x_1, h(x_1), x_2, h(x_2), ..., x_n, h(x_n), x_{query}) $$

where $h \sim \mathcal{D}_ \mathcal{H}$, and $x_1, ..., x_n, x_{query} \sim \mathcal{D}_ \mathcal{X}$.

A model $f$ in-context learns the function class $\mathcal{H}$ up to $\epsilon$ if the population loss

$$ \mathbb{E}_ {x_i, x_{query}, h}\left[(f(S)-h(x_{query}))^2\right] \leq \epsilon $$

for large enough $n$ (from [Garg et al., 2022](https://arxiv.org/pdf/2208.01066)).
