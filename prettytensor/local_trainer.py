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
"""Simple training and evaluation that runs locally.

See trainer.py in the parent directory for a full featured trainer that runs
well with multiple replicas.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import functools
import itertools
import operator
import os.path
import sys
import time

import six
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin
import tensorflow as tf

from prettytensor import bookkeeper


SESSION_MANAGER_FACTORY = tf.train.SessionManager


def create_follower_runner():
  """Creates a runner that will wait for another runne/Session to run."""
  return Runner(follower=True)


def create_checkpointing_runner(save_path, logdir):
  """Creates a runner that will initialize the model.

  Args:
    save_path: The path to checkpoint or None for no checkpoints.
    logdir: Directory for writing summaries.  If None, then defaults to the
      directory of save_path.
  Returns:
    A runner configured to initialize the model and write data to path.
  """
  return Runner(save_path, logdir, follower=False)


class Runner(object):
  """The runner provides convenience methods to train and evaluate models."""

  def __init__(self,
               save_path=None,
               logdir=None,
               restore=True,
               coord=None,
               initial_checkpoint=None,
               follower=False):
    """Create a Runner object that checkpoints to the given path.

    Args:
      save_path: The path to checkpoint or None for no checkpoints.
      logdir: Directory for writing summaries.  If None, then defaults to the
        directory of save_path.
      restore: If False, disable restoring the model (force a fresh run).
      coord: The coordinator to use for threads.
      initial_checkpoint: If not None, restore from the given checkpoint instead
        of starting fresh.
      follower: True to make this wait for another session.
    """
    self._restore = restore
    self._save_path = save_path
    self._var_count = 0
    self._initial_checkpoint = initial_checkpoint

    # Used primarily for testing.
    self._last_init = None
    self._last_restore = None
    if self._save_path:
      self._logdir = logdir if logdir else os.path.dirname(self._save_path)
      self._summary_writer = tf.summary.FileWriter(self._logdir)
    else:
      self._logdir = logdir
      self._summary_writer = None
    self._create_initializers()
    self._test_vars = None
    if coord:
      self._coord = coord
    else:
      self._coord = tf.train.Coordinator()
    self._qr2threads = {}
    self._sess = None
    self._follower = follower

  @contextlib.contextmanager
  def session(self, master='', config=None):
    """Takes care of starting any local servers and stopping queues on exit.

    In general, the Runner is designed to work with any user provided session,
    but this provides a convenience for properly stopping the queues.

    Args:
      master: The master session to use.
      config: A tf.ConfigProto or None.

    Yields:
      A session.
    """
    session_manager = SESSION_MANAGER_FACTORY()
    # Initialization is handled manually at a later point and session_manager
    # is just used for distributed compatibility.
    with session_manager.prepare_session(master, None, config=config,
                                         init_fn=lambda _: None) as sess:
      try:
        yield sess
      finally:
        self.stop_queues()

  def _create_initializers(self):
    if self._var_count != len(tf.global_variables()):
      save_dir = os.path.dirname(self._save_path) if self._save_path else None
      if save_dir and not tf.gfile.IsDirectory(save_dir):
        tf.gfile.MakeDirs(save_dir)
      self._saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
      self._init = tf.global_variables_initializer()
      self._local_init = tf.local_variables_initializer()
      self._check_inited = tf.assert_variables_initialized()
      self._var_count = len(tf.global_variables())
      if self._summary_writer:
        self._summaries = tf.summary.merge_all()
        self._summary_writer.add_graph(tf.get_default_graph())

  def _init_model(self, sess, allow_initialize):
    if allow_initialize:
      self._create_initializers()
    try:
      self._check_inited.op.run()
      self._last_init = False
    except tf.errors.FailedPreconditionError:
      if allow_initialize:
        self._local_init.run()
        if self._summary_writer:
          tf.train.write_graph(tf.get_default_graph().as_graph_def(),
                               self._logdir, 'graph.pbtxt')
        self._init.run()
        self._last_init = True
      else:
        self._last_init = False
      if self._restore and self.load_from_checkpoint(sess):
        self._local_init.run()
        self._last_restore = self._saver.last_checkpoints[-1]
      else:
        if self._initial_checkpoint:
          self._local_init.run()
          self._saver.restore(sess, self._initial_checkpoint)
          self._last_restore = self._initial_checkpoint
        else:
          self._last_restore = None

  def prepare_model(self, sess, allow_initialize=True):
    """Initialize the model and if necessary launch the queue runners."""
    if self._follower:
      self.wait_for_initialization()
    else:
      self._init_model(sess, allow_initialize)

    if sess is not self._sess:
      if self.threads:
        raise ValueError('You must call stop_queues() before '
                         'starting a new session with QueueRunners.')
      self._sess = sess
    self._start_threads(sess)

  @property
  def saver(self):
    self._create_initializers()
    return self._saver

  def load_from_checkpoint(self, sess, latest_filename=None):
    """Loads the model from the most recent checkpoint.

    This gets the most current list of checkpoints each time it is called.

    Args:
      sess: The current session.
      latest_filename: The filename for the latest set of checkpoints, defaults
        to 'checkpoints'.
    Returns:
      The loaded checkpoint or None if it failed to load.
    """
    # Set list of not-yet-deleted checkpoints.
    self._create_initializers()
    if self._save_path:
      ckpt = tf.train.get_checkpoint_state(
          os.path.dirname(self._save_path), latest_filename)
      if ckpt and ckpt.all_model_checkpoint_paths:
        # Copy it because last_checkpoints is immutable.
        # Manually configure a new Saver so that we get the latest snapshots.
        self._saver = tf.train.Saver(saver_def=self._saver.as_saver_def())
        self._saver.set_last_checkpoints(list(ckpt.all_model_checkpoint_paths))
    if self._saver.last_checkpoints:
      self._saver.restore(sess, self._saver.last_checkpoints[-1])
      return self._saver.last_checkpoints[-1]
    else:
      return None

  def _log_and_save(self, sess, results):
    step = results[0]
    to_print = [x for x in results[1:] if x is not None]
    print('[%d] %s' % (step, to_print))
    sys.stdout.flush()
    if self._save_path:
      self._saver.save(sess, self._save_path, step)

  def _start_threads(self, sess):
    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
      if qr not in self._qr2threads:
        self._qr2threads[qr] = tuple(qr.create_threads(sess,
                                                       coord=self._coord,
                                                       daemon=True,
                                                       start=True))

  @property
  def threads(self):
    return functools.reduce(operator.add, six.itervalues(self._qr2threads), ())

  def stop_queues(self):
    # QQQ: Should this automatically be called when the session is closed?

    self._coord.request_stop()
    try:
      if self.threads:
        # All the code in coordinator looks good, but it sometimes deadlocks.
        time.sleep(1.0)
        self._coord.join(self.threads)
    finally:
      self._sess = None
      self._qr2threads = {}
      self._coord.clear_stop()

  def run_model(self,
                op_list,
                num_steps,
                feed_vars=(),
                feed_data=None,
                print_every=100,
                allow_initialize=True):
    """Runs `op_list` for `num_steps`.

    Args:
      op_list: A list of ops to run.
      num_steps: Number of steps to run this for.  If feeds are used, this is a
        maximum. `None` can be used to signal "forever".
      feed_vars: The variables to feed.
      feed_data: An iterator that feeds data tuples.
      print_every: Print a log line and checkpoing every so many steps.
      allow_initialize: If True, the model will be initialized if any variable
        is uninitialized, if False the model will not be initialized.
    Returns:
      The final run result as a list.
    Raises:
      ValueError: If feed_data doesn't match feed_vars.
    """
    feed_data = feed_data or itertools.repeat(())

    ops = [bookkeeper.global_step()]
    ops.extend(op_list)

    sess = tf.get_default_session()
    self.prepare_model(sess, allow_initialize=allow_initialize)
    results = []

    try:
      if num_steps is None:
        counter = itertools.count(0)
      elif num_steps >= 0:
        counter = xrange(num_steps)
      else:
        raise ValueError('num_steps cannot be negative: %s' % num_steps)
      for i, data in zip(counter, feed_data):
        log_this_time = print_every and i % print_every == 0
        if len(data) != len(feed_vars):
          raise ValueError(
              'feed_data and feed_vars must be the same length: %d vs %d' % (
                  len(data), len(feed_vars)))
        if self._coord.should_stop():
          print('Coordinator stopped')
          sys.stdout.flush()
          self.stop_queues()
          break
        if len(feed_vars) != len(data):
          raise ValueError('Feed vars must be the same length as data.')

        if log_this_time and self._summary_writer:
          results = sess.run(ops + [self._summaries],
                             dict(zip(feed_vars, data)))
          self._summary_writer.add_summary(results[-1], results[0])
          results = results[:-1]
        else:
          results = sess.run(ops, dict(zip(feed_vars, data)))
        if log_this_time:
          self._log_and_save(sess, results)

      # Print the last line if it wasn't just printed
      if print_every and not log_this_time:
        self._log_and_save(sess, results)
    except tf.errors.OutOfRangeError as ex:
      print('Done training -- epoch limit reached %s' % ex.message)
      sys.stdout.flush()
      self.stop_queues()
    except BaseException as ex:
      print('Exception -- stopping threads: %s' % ex, file=sys.stderr)
      sys.stdout.flush()
      self.stop_queues()
      raise
    return results

  def train_model(self,
                  train_op,
                  cost_to_log,
                  num_steps,
                  feed_vars=(),
                  feed_data=None,
                  print_every=100):
    """Trains the given model.

    Args:
      train_op: The training operation.
      cost_to_log: A cost to log.
      num_steps: Number of batches to run.
      feed_vars: A list or tuple of the variables that will be fed.
      feed_data: A generator that produces tuples of the same length as
        feed_vars.
      print_every: Print and save every so many steps.
    Returns:
      `cost_to_log` from the final step.
    """
    costs = [train_op]
    if (isinstance(cost_to_log, collections.Sequence)
        and not isinstance(cost_to_log, six.string_types)):
      costs.extend(cost_to_log)
    else:
      costs.append(cost_to_log)
    return self.run_model(costs,
                          num_steps,
                          feed_vars=feed_vars,
                          feed_data=feed_data,
                          print_every=print_every)[2:]

  def _run_init_test_vars_op(self):
    test_vars = tf.get_collection(bookkeeper.GraphKeys.TEST_VARIABLES)
    if test_vars:
      if test_vars != self._test_vars:
        self._test_vars = list(test_vars)
        self._test_var_init_op = tf.variables_initializer(test_vars)
      return self._test_var_init_op.run()

  def evaluate_model(self,
                     accuracy,
                     num_steps,
                     feed_vars=(),
                     feed_data=None,
                     summary_tag=None,
                     print_every=0):
    """Evaluates the given model.

    Args:
      accuracy: The metric that is being evaluated or a tuple of metrics.
      num_steps: The number of steps to run in the evaluator.
      feed_vars: A list or tuple of the variables that will be fed.
      feed_data: A generator that produces tuples of the same length as
        feed_vars.
      summary_tag: If provided, the final result of running the model will be
        published to this tag.
      print_every: Print a summary every so many steps, use 0 to disable.
    Returns:
      The accuracy.
    Raises:
      ValueError: If the wrong number of summary tags are provided or previously
        running QueueRunners haven't been stopped.
    """
    if not hasattr(self, '_saver'):
      raise ValueError('Before evaluating, you must initialize the model with '
                       'load_from_checkpoint, prepare or saver.')
    self._run_init_test_vars_op()
    if (not isinstance(accuracy, collections.Sequence) or
        isinstance(accuracy, six.string_types)):
      accuracy = (accuracy,)
      if summary_tag:
        summary_tag = (summary_tag,)
    if summary_tag and len(summary_tag) != len(accuracy):
      raise ValueError(
          'If summaries are requested, there must be a tag per accuracy node.')

    result = self.run_model(accuracy,
                            num_steps,
                            feed_vars=feed_vars,
                            feed_data=feed_data,
                            print_every=print_every,
                            allow_initialize=False)
    assert len(result) == len(accuracy) + 1, (
        'results is wrong length, was %s but should be 1 longer than %s' %
        (result, accuracy))
    if summary_tag:
      self.add_summaries(result[0], *zip(summary_tag, result[1:]))
    return result[1:]

  def add_summaries(self, step, *tags_and_values):
    """Adds summaries to the writer and prints a log statement."""
    values = []
    to_print = []
    for tag, value in tags_and_values:
      values.append(tf.Summary.Value(tag=tag, simple_value=float(value)))
      to_print.append('%s=%g' % (tag, value))
    if self._summary_writer:
      summary = tf.Summary(value=values)
      event = tf.Event(wall_time=time.time(),
                       summary=summary,
                       step=int(step))
      self._summary_writer.add_event(event)
    print('[%d] %s' % (step, ', '.join(to_print)))

  def wait_for_initialization(self, wait_time_seconds=10):
    while True:
      try:
        self._check_inited.op.run()
        return
      except tf.errors.FailedPreconditionError:
        print('Model not yet available, sleeping for %d seconds.' %
              wait_time_seconds)
        sys.stdout.flush()
        time.sleep(wait_time_seconds)

  def load_new_checkpoint_when_available(
      self, sess, current_checkpoint, sleep_seconds=10):
    """Waits for a new checkpoint to be available and then loads it.

    Args:
      sess: The current session.
      current_checkpoint: The current checkpoint or None to just load the next
        one.
      sleep_seconds: How long to sleep between checks.

    Returns:
      The next checkpoint to use.
    """
    # Load the checkpoint.
    while True:
      next_checkpoint = self.load_from_checkpoint(sess)
      if not next_checkpoint or next_checkpoint == current_checkpoint:
        print('Model not yet available, sleeping for %d seconds: '
              'path %s; found: %s' %
              (sleep_seconds,
               os.path.dirname(self._save_path), current_checkpoint))
        sys.stdout.flush()
        time.sleep(sleep_seconds)
      else:
        return next_checkpoint

  def evaluate_repeatedly(self,
                          accuracy,
                          num_steps,
                          feed_vars=(),
                          feed_data=None,
                          summary_tag=None,
                          evaluation_times=-1):
    """Runs the evaluation in a loop for `evaluation_times`.

    On each iteration, `evaluate_model` is called with the supplied arguments.
    This manages the queue threads itself.

    Args:
      accuracy: The metric that is being evaluated.
      num_steps: The number of steps to run in the evaluator.
      feed_vars: A list or tuple of the variables that will be fed.
      feed_data: A generator that produces tuples of the same length as
        feed_vars.
      summary_tag: If provided, the final result of each evaluation will be
        published to this tag.
      evaluation_times: Run this loop for this many times or forever if it is
        `-1`.

    Returns:
      The final evaluation result from `evaluate_model` if `evaluation_times`
      ever ends.
    """
    current_checkpoint = None
    try:
      for i in itertools.count(0):
        # New session each time to reset queues - Yay.
        with self.session() as sess:
          current_checkpoint = self.load_new_checkpoint_when_available(
              sess, current_checkpoint)
          # Create relevant ops before starting queue runners.
          self._run_init_test_vars_op()

          accuracy_result = self.evaluate_model(accuracy,
                                                num_steps,
                                                summary_tag=summary_tag,
                                                print_every=0,
                                                feed_vars=feed_vars,
                                                feed_data=feed_data)
          if not summary_tag:
            print('[%d] %s' % (sess.run(bookkeeper.global_step()),
                               accuracy_result))
          if (i + 1) == evaluation_times:
            return accuracy_result
    finally:
      print('Shutting down')
      sys.stdout.flush()
      self.stop_queues()
