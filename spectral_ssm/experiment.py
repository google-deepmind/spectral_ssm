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

"""Utilities for running an experiment."""

import functools
from typing import Any

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf

from spectral_ssm import utils


class Experiment:
  """Class to initialize and maintain experiment state."""

  def __init__(
      self,
      forward: hk.TransformedWithState,
      optimizer: optax.GradientTransformation,
      rng: jax.Array = jax.random.PRNGKey(0),
  ) -> None:
    """Initializes an experiment.

    Args:
      forward: A haiku transformed forward function.
      optimizer: An optax optimizer.
      rng: A jax random key.
    """
    self.forward = forward
    self.optimizer = optimizer
    self.rng = utils.broadcast_to_local_devices(rng)

    self.params = None
    self.network_state = None
    self.opt_state = None

  def step(self, inputs: chex.ArrayTree) -> dict[str, jax.Array]:
    """Takes a single step of the experiment.

    Args:
      inputs: A batch of inputs.

    Returns:
      metrics: A dictionary of metrics.
    """
    if self.params is None:
      self.init_fn(inputs)

    self.params, self.network_state, self.opt_state, self.rng, metrics = (
        self.update_fn(
            self.params,
            self.network_state,
            self.opt_state,
            self.rng,
            inputs,
        )
    )

    return metrics

  def init_fn(self, inputs: chex.ArrayTree) -> None:
    """Initializes the experiment.

    Args:
      inputs: A batch of inputs.
    """
    p_init = jax.pmap(
        functools.partial(self.forward.init, is_training=True), axis_name='i'
    )
    self.params, self.network_state = p_init(
        self.rng,
        inputs['src'],
    )
    self.opt_state = jax.pmap(self.optimizer.init, axis_name='i')(self.params)

  def task_fn(
      self,
      outputs: jax.Array,
      targets: jax.Array,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Computes the task loss, accuracy, and metrics.

    Args:
      outputs: A batch of outputs.
      targets: A batch of targets.

    Returns:
      The number of examples in the batch, and the loss and accuracy over the
      batch.
    """
    num_classes = outputs.shape[-1]

    # (batch, output_len, num_classes)
    one_hot_targets = jax.nn.one_hot(targets, num_classes)
    loss = jnp.sum(optax.softmax_cross_entropy(outputs, one_hot_targets))

    # (batch, output_len, num_classes)
    probs = jax.nn.softmax(outputs, axis=-1)

    # (batch, output_len)
    preds = jnp.argmax(probs, axis=-1).astype(jnp.int32)
    targets = targets.astype(jnp.int32)
    correct = jnp.sum((preds == targets).astype(jnp.float32))
    count = jnp.sum(jnp.ones_like(targets).astype(jnp.float32))

    return count, loss, correct

  def loss_fn(
      self,
      params: chex.ArrayTree,
      state: chex.ArrayTree,
      rng: jax.Array,
      data: chex.ArrayTree,
      is_training: bool = True,
  ) -> tuple[jax.Array, tuple[dict[str, jax.Array], Any]]:
    """Computes the loss and metrics for a batch of data.

    Args:
      params: The model parameters.
      state: The model state.
      rng: A jax random key.
      data: A batch of data.
      is_training: Whether we are training or not.

    Returns:
      The loss, metrics, and model state for the batch of data.
    """
    outputs, state = self.forward.apply(
        params,
        state,
        rng=rng,
        inputs=data['src'],
        is_training=is_training,
    )
    count, loss, correct = self.task_fn(outputs, data['tgt'])

    metrics = dict(count=count, loss=loss, correct=correct)
    return loss, (metrics, state)

  @functools.partial(jax.pmap, static_broadcasted_argnums=(0,), axis_name='i')
  def update_fn(
      self,
      params: chex.ArrayTree,
      network_state: chex.ArrayTree,
      opt_state: chex.ArrayTree,
      rng: jax.Array,
      inputs: chex.ArrayTree,
  ) -> tuple[
      chex.ArrayTree,
      chex.ArrayTree,
      chex.ArrayTree,
      jax.Array,
      dict[str, jax.Array],
  ]:
    """Applies an update to parameters and returns new state.

    Args:
      params: The model parameters.
      network_state: The model state.
      opt_state: The optimizer state.
      rng: A jax random key.
      inputs: A batch of inputs.

    Returns:
      params: The updated model parameters.
      network_state: The updated model state.
      opt_state: The updated optimizer state.
      rng: The updated random key.
      metrics: A dictionary of metrics.
    """
    rng, subrng = jax.random.split(rng)
    grad_loss_fn = jax.grad(self.loss_fn, has_aux=True)
    grads, (metrics, network_state) = grad_loss_fn(
        params, network_state, subrng, inputs
    )
    grads = jax.lax.pmean(grads, axis_name='i')

    # Compute and apply updates via our optimizer.
    updates, opt_state = self.optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    # Scalars to log (average across all hosts/devices).
    metrics = jax.lax.pmean(metrics, axis_name='i')

    return params, network_state, opt_state, rng, metrics

  def eval_epoch(self, dataset: tf.data.Dataset) -> dict[str, jax.Array]:
    """Evaluates an epoch.

    Args:
      dataset: A tf.data.Dataset to use for eval.

    Returns:
      epoch_metrics: A dictionary of metrics.
    """
    epoch_metrics = {
        'count': jnp.array(0.0),
        'loss': jnp.array(0.0),
        'correct': jnp.array(0.0),
    }
    for inputs in dataset:
      _, (metrics, _) = jax.pmap(
          functools.partial(self.loss_fn, is_training=False),
          axis_name='i',
      )(self.params, self.network_state, self.rng, inputs)
      for k, v in metrics.items():
        epoch_metrics[k] += v

    return epoch_metrics
