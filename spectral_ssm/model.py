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

"""Spectral temporal unit (STU) block."""

import functools

import haiku as hk
import jax
import jax.numpy as jnp

from spectral_ssm import stu_utils


@functools.partial(jax.vmap, in_axes=(None, 0, None))
def apply_stu(
    params: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    inputs: jnp.ndarray,
    eigh: tuple[jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
  """Apply STU.

  Args:
    params: A tuple of parameters of shapes [d_out, d_out], [d_in, d_out, k_u],
      [d_in * k, d_out] and [d_in * k, d_out]
    inputs: Input matrix of shape [l, d_in].
    eigh: A tuple of eigenvalues [k] and circulant eigenvecs [k, l, l].

  Returns:
    A sequence of y_ts of shape [l, d_out].
  """
  m_y, m_u, m_phi = params

  x_tilde = stu_utils.compute_x_tilde(inputs, eigh)

  # Compute deltas from the spectral filters, which are of shape [l, d_out].
  delta_phi = x_tilde @ m_phi

  # Compute deltas from AR on x part
  delta_ar_u = stu_utils.compute_ar_x_preds(m_u, inputs)

  # Compute y_ts, which are of shape [l, d_out].
  return stu_utils.compute_y_t(m_y, delta_phi + delta_ar_u)


class STU(hk.Module):
  """Simple STU Layer."""

  def __init__(
      self,
      d_out: int = 256,
      input_len: int = 1024,
      num_eigh: int = 24,
      auto_reg_k_u: int = 3,
      auto_reg_k_y: int = 2,
      learnable_m_y: bool = True,
  ) -> None:
    """Initialize STU layer.

    Args:
      d_out: Output dimension.
      input_len: Input sequence length.
      num_eigh: Tuple of eigen values and vecs sized (k,) and (l, k)
      auto_reg_k_u: Auto-regressive depth on the input sequence,
      auto_reg_k_y: Auto-regressive depth on the output sequence,
      learnable_m_y: m_y matrix learnable,
    """
    super().__init__()
    self.d_out = d_out
    self.eigh = stu_utils.get_top_hankel_eigh(input_len, num_eigh)
    self.l, self.k = input_len, num_eigh
    self.auto_reg_k_u = auto_reg_k_u
    self.auto_reg_k_y = auto_reg_k_y
    self.learnable_m_y = learnable_m_y
    self.m_x_var = 1.0 / (float(self.d_out) ** 0.5)

    self.init_m_y = jnp.zeros([self.d_out, self.auto_reg_k_y, self.d_out])

    # NOTE: Assume d_in = d_out.
    self.init_m_u = stu_utils.get_random_real_matrix(
        (self.d_out, self.d_out, self.auto_reg_k_u), self.m_x_var
    )
    self.init_m_phi = jnp.zeros([self.d_out * self.k, self.d_out])

  def __call__(
      self,
      inputs: jnp.ndarray,
  ) -> jnp.ndarray:
    """Forward pass.

    Args:
      inputs: Assumed to be of shape (B, L, H) where B is batch size, L is
        sequence length, H is number of features (channels) in the input.

    Returns:
      `jnp.ndarray` of preactivations.
    """
    d_in = inputs.shape[-1]
    input_dtype = inputs.dtype

    if self.learnable_m_y:
      m_y = hk.get_parameter(
          'm_y',
          shape=[self.d_out, self.auto_reg_k_y, self.d_out],
          dtype=input_dtype,
          init=lambda s, d: self.init_m_y,
      )
    else:
      m_y = self.init_m_y

    m_u = hk.get_parameter(
        'm_u',
        shape=[d_in, self.d_out, self.auto_reg_k_u],
        dtype=input_dtype,
        init=lambda s, d: self.init_m_u,
    )
    m_phi = hk.get_parameter(
        'm_phi',
        shape=[d_in * self.k, self.d_out],
        dtype=input_dtype,
        init=lambda s, d: self.init_m_phi,
    )

    params = (m_y, m_u, m_phi)

    return apply_stu(params, inputs, self.eigh)


class Architecture(hk.Module):
  """General model architecture."""

  def __init__(
      self,
      name=None,
      d_model: int = 256,
      d_target: int = 10,
      num_layers: int = 6,
      dropout: float = 0.1,
      input_len: int = 1024,
      num_eigh: int = 24,
      auto_reg_k_u: int = 3,
      auto_reg_k_y: int = 2,
      learnable_m_y: bool = True,
  ) -> None:
    """Initialize general model architecture.

    Args:
      name: Name of the module.
      d_model: Dimension of the embedding.
      d_target: Dimension of the target.
      num_layers: Number of layers.
      dropout: Dropout rate.
      input_len: Input sequence length.
      num_eigh: Number of eigen values and vecs.
      auto_reg_k_u: Auto-regressive depth on the input sequence.
      auto_reg_k_y: Auto-regressive depth on the output sequence.
      learnable_m_y: m_y matrix learnable.
    """
    super().__init__(name=name)
    self.d_model = d_model
    self.d_target = d_target
    self.num_layers = num_layers
    self.dropout = dropout
    self.input_len = input_len
    self.num_eigh = num_eigh
    self.auto_reg_k_u = auto_reg_k_u
    self.auto_reg_k_y = auto_reg_k_y
    self.learnable_m_y = learnable_m_y

  def __call__(
      self,
      inputs: jax.Array,
      is_training: bool = True,
  ) -> jax.Array:
    """Forward pass of classification pipeline.

    Args:
      inputs: (batch, length, source dim).
      is_training: True for training mode.

    Returns:
      Outputs of general architecture
    """
    # Embedding layer.
    x = hk.Linear(self.d_model)(inputs)

    for _ in range(self.num_layers):
      # Saving input to layer for residual.
      z = x

      # Construct pre-layer batch norm.
      x = hk.BatchNorm(
          create_offset=True,
          create_scale=True,
          decay_rate=0.9,
          cross_replica_axis='i',
      )(x, is_training=is_training)

      x = STU(
          d_out=self.d_model,
          input_len=self.input_len,
          num_eigh=self.num_eigh,
          auto_reg_k_u=self.auto_reg_k_u,
          auto_reg_k_y=self.auto_reg_k_y,
          learnable_m_y=self.learnable_m_y,
      )(x)

      # GeLU + dropout.
      x = jax.nn.gelu(x)
      if is_training:
        x = hk.dropout(hk.next_rng_key(), self.dropout, x)
      x = hk.Linear(2 * self.d_model)(x)
      x = jax.nn.glu(x)

      # Dropout.
      if is_training:
        x = hk.dropout(hk.next_rng_key(), self.dropout, x)

      # Residual connection.
      x = x + z

    # Projection
    x = jnp.mean(x, axis=1, keepdims=True)
    x = hk.Linear(self.d_target, w_init=None)(x)

    return x
