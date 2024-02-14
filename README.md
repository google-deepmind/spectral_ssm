# Spectral State Space Models

This repository contains code for training and evaluating spectral state space
models and accompanies the paper [Spectral State Space
Models](https://arxiv.org/abs/2312.06837).

The paper studies sequence modeling for prediction tasks with long range
dependencies. We propose a new formulation for state space models (SSMs) based
on learning linear dynamical systems with the spectral filtering algorithm
(Hazan et al. (2017)). This gives rise to a novel sequence prediction
architecture we call a spectral state space model.

Spectral state space models have two primary advantages. First, they have
provable robustness properties as their performance depends on neither the
spectrum of the underlying dynamics nor the dimensionality of the problem.
Second, these models are constructed with fixed convolutional filters that do
not require learning while still outperforming SSMs in both theory and practice.
The resulting models are evaluated on synthetic dynamical systems and long-range
prediction tasks of various modalities. These evaluations support the
theoretical benefits of spectral filtering for tasks requiring very long range
memory.

## Installation

Clone and navigate to the `spectral_ssm` directory containing `setup.py`. Run:

```bash
pip install -e .
```

## Usage

The `example.py` file contains the full training pipeline. `model.py` contains
code for the model itself, including the Spectral Temporal Unit (STU) block.

```bash
python3 example.py
```

## Citing this work

```latex
@misc{agarwal2024spectral,
      title={Spectral State Space Models},
      author={Naman Agarwal and Daniel Suo and Xinyi Chen and Elad Hazan},
      year={2024},
      eprint={2312.06837},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## License and disclaimer

Copyright 2024 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
