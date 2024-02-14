# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Spectral State Space Models.

See more details in the
[`README.md`](https://github.com/google-deepmind/spectral_ssm).
"""

import setuptools

setuptools.setup(
    name='spectral_ssm',
    version='0.0.1',
    description='Spectral State Space Models',
    author='Spectral SSM Team',
    author_email='dsuo@google.com',
    url='http://github.com/google-deepmind/spectral_ssm',
    license='Apache 2.0',
    packages=setuptools.find_packages(),
    install_requires=[
        'absl-py',
        'chex',
        'haiku',
        'jax',
        'numpy',
        'optax',
        'tensorflow==2.5.0',
        'tensorflow-datasets',
        'tqdm',
    ],
    extras_require={},
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='jax machine learning spectral state space models',
)
