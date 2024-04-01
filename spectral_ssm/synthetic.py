"""Synthetic long-context datasets."""

import numpy as np
import tensorflow as tf


def generate_copy(
    num_examples: int = 5,
    num_categories: int = 10,
    copy_len: int = 10,
    blank_len: int = 5,
    selective: bool = False,
    seed: int = 0,
) -> tf.data.Dataset:
  r"""Generate a copy task.

  This copy task is taken from (Arjovsky, Shah, and Bengio, 2016). From their
  paper:

    Following a similar setup to (Hochreiter & Schmidhuber, 1997), we outline
    the copy memory task. Consider 10 categories, a_0 to a_9. The input takes
    the form of a T + 20 length vector of categories, where we test over a range
    of values of T. The first 10 entries are sampled uniformly, independently
    and with replacement from a_0 to a_7, and represent the sequence which will
    need to be remembered. The next T − 1 entries are set to a_8, which can be
    thought of as the ’blank’ category. The next single entry is a_9, which
    represents a delimiter, which should indicate to the algorithm that it is
    now required to reproduce the initial 10 categories in the output. The
    remaining 10 entries are set to a_8. The required output sequence consists
    of T + 10 repeated entries of a_8, followed by the first 10 categories of
    the input sequence in exactly the same order. The goal is to minimize the
    average cross entropy of category predictions at each time step of the
    sequence. The task amounts to having to remember a categorical sequence of
    length 10, for T time steps.

    A simple baseline can be established by considering an optimal strategy when
    no memory is available, which we deem the memoryless strategy. The
    memoryless strategy would be to predict a_8 for T + 10 entries and then
    predict each of the final 10 categories from the set {a_i}_{i=0}^7
    independently and uniformly at random. The categorical cross entropy of this
    strategy is \frac{10 log(8)}{T + 20}.

    If selective is True, then shuffle the blank spaces between the array to
    copy.

  Args:
    num_examples: Number of examples to generate.
    num_categories: Number of token types. One is used as a blank token, one is
      used as a delimiter, and the remaining are used to choose from for
      copying.
    copy_len: Number of tokens to copy.
    blank_len: Number of blank tokens inbetween copy and paste.
    selective: Whether to return a selective copy task or not.
    seed: Seed for random number generator.

  Returns:
    A Tensorflow dataset.
  """
  # Assign characters.
  copy_chars = np.arange(num_categories - 2)
  blank_char = num_categories - 2
  delim_char = num_categories - 1

  # Construct input sequences.
  rng = np.random.default_rng(seed=seed)
  to_copy = rng.choice(copy_chars, (num_examples, copy_len), replace=True)
  blank = np.full((num_examples, blank_len - 1), blank_char)
  delim = np.full((num_examples, 1), delim_char)
  to_fill = np.full((num_examples, copy_len), blank_char)

  if selective:

    def insert(row):
      return np.insert(
          row, rng.choice(copy_len, blank_len - 1, replace=True), blank_char
      )

    inputs = np.apply_along_axis(insert, axis=1, arr=to_copy)
  else:
    inputs = np.hstack((to_copy, blank))
  inputs = np.hstack((inputs, delim, to_fill))

  # Construct output sequences.
  blank = np.full((num_examples, blank_len + copy_len), blank_char)
  outputs = np.hstack((blank, to_copy))

  # Construct dataset.
  ds = {
      'src': inputs,
      'tgt': outputs,
  }

  return tf.data.Dataset.from_tensor_slices(ds)


def generate_adding(
    num_examples: int = 5,
    sequence_len: int = 10,
    seed: int = 0,
) -> tf.data.Dataset:
  """Generate an adding task.

  This adding task is taken from (Arjovsky, Shah, and Bengio, 2016). From their
  paper:

    We closely follow the adding problem defined in (Hochreiter & Schmidhuber,
    1997) to explain the task at hand. Each input consists of two sequences of
    length T. The first sequence, which we denote x, consists of numbers sampled
    uniformly at random U[0, 1]. The second sequence is an indicator sequence
    consisting of exactly two entries of 1 and remaining entries 0. The first 1
    entry is located uniformly at random in the first half of the sequence,
    whilst the second 1 entry is located uniformly at random in the second half.
    The output is the sum of the two entries of the first sequence,
    corresponding to where the 1 entries are located in the second sequence. A
    naive strategy of predicting 1 as the output regardless of the input
    sequence gives an expected mean squared error of 0.167, the variance of the
    sum of two independent uniform distributions. This is our baseline to beat.

  Args:
    num_examples: Number of examples to generate.
    sequence_len: Length of each sequence.
    seed: Seed for random number generator.

  Returns:
    A Tensorflow dataset.
  """
  # Construct the first sequence.
  rng = np.random.default_rng(seed=seed)
  seq_1 = rng.uniform(low=0, high=1.0, size=(num_examples, sequence_len))

  # Construct the second sequence.
  seq_2 = np.full((num_examples, sequence_len), 0)
  idx_1 = rng.choice(np.arange(0, sequence_len // 2), num_examples)
  idx_2 = rng.choice(np.arange(sequence_len // 2, sequence_len), num_examples)
  seq_2[np.arange(num_examples), idx_1] = 1
  seq_2[np.arange(num_examples), idx_2] = 1

  # Compute the outputs.
  outputs = np.sum(seq_1 * seq_2, axis=1)

  # Concatenate the inputs.
  inputs = np.hstack((seq_1, seq_2))

  # Construct dataset.
  ds = {
      'src': inputs,
      'tgt': outputs,
  }

  return tf.data.Dataset.from_tensor_slices(ds)


def generate_induction_heads(
    num_examples: int = 5,
    sequence_len: int = 30,
    vocab_size: int = 20,
    seed: int = 0,
) -> tf.data.Dataset:
  """Generate an induction heads task.

  This induction heads task is taken from (Dao, Fu, Saab, 2023). From their
  paper:

    The Induction Head task tests how well a model can recall content after a
    special token. At the end of the sequence, the model must recall the token
    that appeared immediately after the special token earlier in the sequence.

  Args:
    num_examples: Number of examples to generate.
    sequence_len: Length of each sequence.
    vocab_size: Size of the vocabulary, including the special token.
    seed: Seed for random number generator.

  Returns:
    A Tensorflow dataset.
  """
  # Set the special token.
  special = vocab_size - 1
  rng = np.random.default_rng(seed=seed)

  inputs = rng.choice(vocab_size - 1, (num_examples, sequence_len),
                      replace=True)

  # Place special token somewhere before the last token.
  idx = rng.choice(sequence_len - 2, num_examples, replace=True)
  inputs[np.arange(num_examples), idx] = special

  # Place special token at the end of the sequence.
  inputs[np.arange(num_examples), -1] = special

  outputs = inputs[np.arange(num_examples), idx + 1].squeeze()

  ds = {
      'src': inputs,
      'tgt': outputs,
  }

  return tf.data.Dataset.from_tensor_slices(ds)


def generate_associative_recall(
    num_examples: int = 5,
    sequence_len: int = 30,
    vocab_size: int = 10,
    seed: int = 0,
) -> tf.data.Dataset:
  """Generate an associative recall task.

  This associative recall task is taken from (Dao, Fu, Saab, 2023). From their
  paper:

    Associative Recall is similar to the induction head task, but requires the
    model to remember multiple key-value pairs. At the end of the sequence, the
    model must recall a specific value belonging to a specific key.

  Questions:
    - Should each example have a new association?
    - Should we use the same tokens each time?

  Args:
    num_examples: Number of examples to generate.
    sequence_len: Length of each sequence.
    vocab_size: Size of the vocabulary.
    seed: Seed for random number generator.

  Returns:
    A Tensorflow dataset.
  """
  rng = np.random.default_rng(seed=seed)
  idx = rng.choice(vocab_size, (num_examples, sequence_len // 2), replace=True)

  def get_assoc(start: int, end: int):
    # Range of values
    x = np.arange(start, end)

    # Make num_examples copies.
    x = np.tile(x, (num_examples, 1))

    # Shuffle each row independently.
    x = rng.permuted(x, axis=1)

    # Grab the corresponding indices
    return np.take(x, idx)

  keys = get_assoc(0, vocab_size)
  vals = get_assoc(vocab_size, 2 * vocab_size)

  # Interleave keys and values by row.
  inputs = np.zeros((num_examples, sequence_len), dtype=keys.dtype)
  inputs[:, 0::2] = keys
  inputs[:, 1::2] = vals

  # Get key we want to find associated value for.
  idx = rng.choice(vocab_size, num_examples, replace=True)
  keys = np.expand_dims(keys[np.arange(num_examples), idx], axis=1)
  inputs = np.hstack((inputs, keys))
  outputs = vals[np.arange(num_examples), idx].squeeze()

  ds = {
      'src': inputs,
      'tgt': outputs,
  }

  return tf.data.Dataset.from_tensor_slices(ds)
