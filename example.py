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

"""Example training loop."""

from collections.abc import Sequence

from absl import app
import haiku as hk
import tqdm

from spectral_ssm import cifar10
from spectral_ssm import experiment
from spectral_ssm import model
from spectral_ssm import optimizer


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  train_batch_size = 49
  eval_batch_size = 48

  num_steps = 180_000
  eval_period = 1000

  def forward_fn(*args, **kwargs):
    return model.Architecture(
        name=None,
        d_model=256,
        d_target=10,
        num_layers=6,
        dropout=0.1,
        input_len=1024,
        num_eigh=24,
        auto_reg_k_u=3,
        auto_reg_k_y=2,
        learnable_m_y=True,
    )(*args, **kwargs)

  forward = hk.transform_with_state(forward_fn)

  opt = optimizer.get_optimizer(
      num_steps=180_000,
      warmup_steps=18_000,
      learning_rate=5e-4,
      weight_decay=1e-1,
      m_y_learning_rate=5e-5,
      m_y_weight_decay=0,
  )
  exp = experiment.Experiment(forward=forward, optimizer=opt)

  train_ds = cifar10.get_dataset('train', batch_size=train_batch_size)
  pbar = tqdm.tqdm(range(num_steps))
  for global_step in pbar:
    inputs = next(train_ds)
    metrics = exp.step(inputs)
    pbar.set_description(
        f'Step {global_step} - train/acc:'
        f' {metrics["correct"][0] / metrics["count"][0]:.2} train/loss:'
        f' {metrics["loss"][0] / metrics["count"][0]:.2}'
    )

    if global_step > 0 and global_step % eval_period == 0:
      epoch_metrics = exp.eval_epoch(
          cifar10.get_dataset('test', batch_size=eval_batch_size)
      )
      print(f'Eval {global_step}:')
      print(
          f' \t{epoch_metrics["correct"][0] / epoch_metrics["count"][0]:.2} train/loss:'
          f' \t{epoch_metrics["loss"][0] / epoch_metrics["count"][0]:.2}'
      )


if __name__ == '__main__':
  app.run(main)
