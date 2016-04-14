# Copyright 2015 Google Inc. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Creates a replayable queue."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib


import six
import tensorflow as tf

from prettytensor import pretty_tensor_class as prettytensor


def _make_tuple(x):
  """TF has an obnoxious habit of being lenient with single vs tuple."""
  if isinstance(x, prettytensor.PrettyTensor):
    if x.is_sequence():
      return tuple(x.sequence)
    else:
      return (x.tensor,)
  elif isinstance(x, tuple):
    return x
  elif (isinstance(x, collections.Sequence) and
        not isinstance(x, six.string_types)):
    return tuple(x)
  else:
    return (x,)


class ReplayableQueue(object):
  """A switchable queue between the original data and a replayed subset.

  This queue combines 2 concepts:

  1. A replay queue that re-enqueues the data every time it is dequeued so that
      multiple passes over the same data can be made (make sure to iterated
      `replay_steps` times).
  2. The ability to switch data sources between the original source and the
      replayable queue. Embedding the switch makes it easy to construct a single
      graph on top of both input sources.

  Note: The queue requires manual filling by calling `refill`!

  The typical use case for replaying data is when you want to run an experiment
  on a subset of the training data and need to reuse the data multiple times.

  Here is an example that uses the replay queue to monitor loss on a dynamically
  selected validation set:

  ```
  replay = pt.train.ReplayableQueue(lambda: MY_Q.dequeue_many(BATCH_SIZE),
                                REPLAY_SIZE)
  # Build a graph with replay.output
  my_train_op, my_loss = build_graph(replay.output)

  with tf.Session() as sess:
    # Capture some data
    replay.refill(sess)

    for epoch in xrange(EPOCHS):
      # Train for a while
      for _ in xrange(1000):
        sess.run(my_train_op)
      loss = 0
      with replay.replay_scope():
        for _ in xrange(replay.replay_steps):
          loss += sess.run(my_loss)
        loss /= replay.replay_steps
      print('Loss at epoch %d: %g' % (epoch, loss))
  ```
  """

  def __init__(self, input_fn, replay_size, batch_size=None):
    """Creates a ReplayableQueue that takes data from `input_fn`.

    See also: `pt.train.ReplayableQueue.build_from_queue`.

    Note: the shapes of the inputs must be fully defined.

    Note: `input_fn` is a function instead of an input. This is because
      otherwise if the input came from a queue, dependencies wouldn't be set up
      properly and the data would always be dequeued. If you are providing data
      from a queue, then pass in `lambda: q.dequeue_many(batch_size)`.

    Args:
      input_fn: A function of no arguments that returns the input as a tuple of
        `Tensors`.
      replay_size: The size of the replay queue.
      batch_size: If provided, use this as the batch size otherwise infer it.

    Raises:
      ValueError: if `replay_size` is not divisible by `batch_size` or if the
        shapes on the input are wrong.
    """
    inputs = _make_tuple(input_fn())

    for x in inputs:
      x.get_shape().assert_is_fully_defined()
      if batch_size is not None:
        x.get_shape()[0].assert_is_compatible_with(batch_size)
      else:
        batch_size = x.get_shape()[0].value

      dtypes = [x.dtype for x in inputs]
      shapes = [x.get_shape()[1:] if x.get_shape() else () for x in inputs]

    if replay_size % batch_size != 0:
      raise ValueError('replay_size size (%d) must be a multiple of batch size '
                       '(%d)' % (replay_size, batch_size))

    # Setup the flag that controls replay.
    self._replay_var = tf.get_variable(
        'replay',
        dtype=tf.bool,
        shape=[],
        initializer=tf.constant_initializer(False),
        trainable=False)
    self._set_replay_ph = tf.placeholder(dtype=tf.bool)
    self._set_replay = self._replay_var.assign(self._set_replay_ph)

    self._replay_queue = tf.FIFOQueue(replay_size, dtypes, shapes)

    # _fill_queue adds data to the queue and then returns whether it is full.
    with tf.control_dependencies([self._replay_queue.enqueue_many(inputs)]):
      self._fill_queue = tf.less(self._replay_queue.size(), replay_size)

    # Dequeue all the things!
    self._clear_queue = self._replay_queue.dequeue_many(
        self._replay_queue.size())

    def _pull_from_replay():
      data_tuple = _make_tuple(self._replay_queue.dequeue_many(batch_size))
      with tf.control_dependencies([self._replay_queue.enqueue_many(data_tuple)
                                   ]):
        return (tf.identity(data_tuple[0]),) + data_tuple[1:]

    def _pull_from_original():
      return _make_tuple(input_fn())

    self._output = prettytensor.wrap(
        tf.cond(self._replay_var, _pull_from_replay, _pull_from_original))

  @classmethod
  def build_from_queue(cls, input_queue, replay_size, batch_size):
    """Builds a `ReplayableQueue` that draws from a regular `input_queue`.

    Args:
      input_queue: The queue to draw from.
      replay_size: The size of the replay buffer.
      batch_size: The size of each batch.

    Returns:
      A ReplayableQueue.
    """
    return cls(
        lambda: input_queue.dequeue_many(batch_size),
        replay_size,
        batch_size=batch_size)

  @property
  def output(self):
    """Returns the output Tensor for this queue.

    The output is selected between the original data and the replay data
    depending on the replay value.

    Returns:
      The output tensors as a tuple.
    """
    return self._output

  @property
  def replay_steps(self):
    """Returns the number of steps to replay."""
    return self._replay_size // self._batch_size

  @property
  def replay_size(self):
    """Returns the total number of examples this queue holds."""
    return self._replay_size

  @contextlib.contextmanager
  def replay_scope(self, sess):
    """Enters a replay scope that unsets it at the end."""
    current_replay = self.replay(sess)
    try:
      self.set_replay(sess, True)
      yield
    finally:
      self.set_replay(sess, current_replay)

  def replay(self, sess):
    """Gets the current value of replay from the graph.

    Note: this runs the graph, but it is just a var read so it is fairly cheap.

    Args:
      sess: The session in which to run.
    Returns:
      The value of the replay variable.
    """
    return sess.run(self._replay_var)

  def set_replay(self, sess, replay):
    """Changes the current replay setting on the graph."""
    sess.run(self._set_replay, {self._set_replay_ph: replay})

  def refill(self, sess):
    """Clears the current queue and then refills it with new data."""
    sess.run(self._clear_queue)
    # Run until full.
    while sess.run(self._fill_queue):
      pass
