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

"""Data pipeline for CIFAR10."""

import jax
import tensorflow as tf
import tensorflow_datasets as tfds


def preprocess(
    example: dict[str, tf.data.Dataset],
) -> dict[str, tf.Tensor]:
  """Preprocess each example in the dataset.

  Args:
    example: A dict with data for a single example.

  Returns:
    A preprocessed example.
  """
  x = example['image']
  # Floats in [0, 1] instead of ints in [0, 255]
  x = tf.cast(x, dtype=tf.float32) / 255.0
  x = tf.reshape(x, (-1, x.shape[-1]))
  data = {
      'src': x,
      'tgt': tf.reshape(example['label'], (1,)),
  }
  # Normalize by dataset-specific statistics
  means = [0.49139968, 0.48215841, 0.44653091]
  stds = [0.24703223, 0.24348513, 0.26158784]
  data['src'] = (data['src'] - means) / stds

  return data


def get_dataset(
    split: str,
    batch_size: int,
    shuffle_buffer_size: int = 10_000,
) -> tf.data.Dataset:
  """Retrieves a dataset.

  Args:
    split: The dataset split to use.
    batch_size: The batch size to use.
    shuffle_buffer_size: The size of the shuffle buffer to use.

  Returns:
    An iterator over the dataset.
  """

  if split == 'train':
    tfds_split = 'train[:90%]'
  elif split == 'test':
    tfds_split = 'test[90%:]'
  else:
    raise ValueError(f'Unknown split: {split}')

  dataset = tfds.load('cifar10', split=tfds_split)
  dataset = dataset.shard(jax.process_count(), jax.process_index())
  dataset = dataset.map(
      preprocess,
      num_parallel_calls=tf.data.AUTOTUNE,
  )
  if split == 'train':
    dataset = dataset.shuffle(
        buffer_size=shuffle_buffer_size,
        reshuffle_each_iteration=True,
    )
    dataset = dataset.repeat()
  else:
    dataset = dataset.repeat(1)

  # Prepare batches for potentially multi-host, multi-device training.
  local_device_count = jax.local_device_count()
  per_host_batch_size = batch_size // jax.process_count()
  per_device_batch_size = per_host_batch_size // local_device_count
  dataset = dataset.batch(per_device_batch_size, drop_remainder=True)
  dataset = dataset.batch(local_device_count, drop_remainder=True)
  dataset = dataset.prefetch(tf.data.AUTOTUNE)

  # Each key in the batch dict should be of shape (num_hosts,
  # num_devices_per_host,) + value_shape.
  return iter(tfds.as_numpy(dataset))
