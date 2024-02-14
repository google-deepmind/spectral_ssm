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

"""General-purpose utilities."""

from typing import Callable

import chex
import jax


def broadcast_to_local_devices(pytree: chex.ArrayTree) -> chex.ArrayTree:
  """Broadcasts a Pytree to all local devices.

  Args:
    pytree: The Pytree to broadcast.

  Returns:
    A Pytree with the same structure as `pytree`, but with values broadcasted
    to all local devices.
  """
  devices = jax.local_devices()
  return jax.tree_util.tree_map(
      lambda v: jax.device_put_sharded(len(devices) * [v], devices), pytree
  )


def map_nested_fn(
    fn: Callable[[str, jax.Array], jax.Array],
) -> Callable[[chex.ArrayTree], chex.ArrayTree]:
  """Recursively apply `fn` to the key-value pairs of a nested dict.

  Example from optax.multi_transform for defining custom schedulers.

  Args:
    fn: local function applied to leaves mapping (k, v) to string key

  Returns:
    function mapping parameter names to key
  """

  def map_fn(nested_dict):
    return {
        k: map_fn(v) if isinstance(v, dict) else fn(k, v)
        for k, v in nested_dict.items()
    }

  return map_fn
