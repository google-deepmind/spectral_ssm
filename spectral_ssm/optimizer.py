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

"""AdamW with linear warmup and cosine decay."""

import jax
import jax.numpy as jnp
import optax
from spectral_ssm import utils


class WarmupCosineDecay:
  """Cosine decay with linear warmup."""

  def __init__(
      self,
      start_val: float,
      min_lr: float,
      lr: float,
      num_steps: int,
      warmup_steps: int,
  ) -> None:
    """Initialize a cosine decay schedule with warmup.

    Args:
      start_val: The value to start at.
      min_lr: The minimum value to decay to.
      lr: The peak value to reach.
      num_steps: The total number of steps to decay over.
      warmup_steps: The number of steps to warmup for.
    """
    self.start_val = start_val
    self.min_lr = min_lr
    self.lr = lr
    self.num_steps = num_steps
    self.warmup_steps = warmup_steps

  def __call__(self, itr) -> jax.Array:
    """Get learning rate for a given step.

    Args:
      itr: The current step.

    Returns:
      The learning rate for the given step.
    """
    warmup_val = (self.lr - self.start_val) * (
        itr / self.warmup_steps
    ) + self.start_val

    cos_itr = (itr - self.warmup_steps) / (self.num_steps - self.warmup_steps)
    cos = 1 + jnp.cos(jnp.pi * cos_itr)
    cos_val = 0.5 * (self.lr - self.min_lr) * cos + self.min_lr

    # Select warmup_val if itr < warmup, else cosine val
    values = jnp.array([warmup_val, cos_val])
    index = jnp.sum(jnp.array(self.warmup_steps) < itr)

    return jnp.take(values, index)


def get_optimizer(
    num_steps: int = 180_000,
    warmup_steps: int = 18_000,
    learning_rate: float = 5e-4,
    weight_decay: float = 0.1,
    m_y_learning_rate: float = 5e-5,
    m_y_weight_decay: float = 0.0,
) -> optax.GradientTransformation:
  """Build AdamW optimizer with linear warmup and cosine decay.

  Args:
    num_steps: The total number of steps to decay over.
    warmup_steps: The number of steps to warmup for.
    learning_rate: The peak learning rate.
    weight_decay: The weight decay to use.
    m_y_learning_rate: The peak learning rate for m_y parameters.
    m_y_weight_decay: The weight decay to use for m_y parameters.

  Returns:
    An optax.GradientTransformation.
  """
  optimizers = {
      'default': optax.adamw(
          learning_rate=WarmupCosineDecay(
              lr=learning_rate,
              min_lr=1e-7,
              start_val=1e-7,
              num_steps=num_steps,
              warmup_steps=warmup_steps,
          ),
          b1=0.9,
          b2=0.999,
          eps=1e-8,
          eps_root=0.0,
          weight_decay=weight_decay,
      ),
      'm_y': optax.adamw(
          learning_rate=WarmupCosineDecay(
              lr=m_y_learning_rate,
              min_lr=1e-7,
              start_val=1e-7,
              num_steps=num_steps,
              warmup_steps=warmup_steps,
          ),
          b1=0.9,
          b2=0.999,
          eps=1e-8,
          eps_root=0.0,
          weight_decay=m_y_weight_decay,
      ),
  }

  label_fn = utils.map_nested_fn(
      lambda k, _: 'm_y' if k.startswith('m_y') else 'default'
  )

  return optax.multi_transform(optimizers, label_fn)
