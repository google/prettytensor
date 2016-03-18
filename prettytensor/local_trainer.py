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

import itertools
import os.path
import sys
import time

import six
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin
import tensorflow as tf

from prettytensor import bookkeeper


class Runner(object):
  """The runner provides convenience methods to train and evaluate models."""

  def __init__(self, save_path=None, logdir=None, restore=True):
    """Create a Runner object that checkpoints to the given path.

    Args:
      save_path: The path to checkpoint or None for no checkpoints.
      logdir: Directory for writing summaries.  If None, then defaults to the
        directory of save_path.
      restore: If False, disable restoring the model (force a fresh run).
    """
    self._restore = restore
    self._save_path = save_path
    self._var_count = -1

    # Used primarily for testing.
    self._last_init = None
    self._last_restore = None
    if self._save_path:
      self._summary_writer = tf.train.SummaryWriter(
          logdir if logdir else os.path.dirname(self._save_path))
    else:
      self._summary_writer = None
    self._create_initializers()
    self._test_vars = None

  def _create_initializers(self):
    if self._var_count != len(tf.all_variables()):
      save_dir = os.path.dirname(self._save_path) if self._save_path else None
      if save_dir and not tf.gfile.IsDirectory(save_dir):
        tf.gfile.MakeDirs(save_dir)
      self._saver = tf.train.Saver(tf.all_variables(), max_to_keep=5)
      self._init = tf.initialize_all_variables()
      self._check_inited = tf.assert_variables_initialized()
      self._var_count = len(tf.all_variables())
      if self._summary_writer:
        self._summaries = tf.merge_all_summaries()
        self._summary_writer.add_graph(tf.get_default_graph().as_graph_def())

  def _init_model(self, sess, allow_initialize):
    if allow_initialize:
      self._create_initializers()
    try:
      self._check_inited.op.run()
      self._last_init = False
    except tf.errors.FailedPreconditionError:
      if allow_initialize:
        self._init.run()
        self._last_init = True
      else:
        self._last_init = False
      if self._restore and self.load_from_checkpoint(sess):
        self._last_restore = self._saver.last_checkpoints[-1]
      else:
        self._last_restore = None

  def load_from_checkpoint(self, sess, latest_filename=None):
    """Loads the model from the most recent checkpoint.

    This gets the most current list of checkpoints each time it is called.

    Args:
      sess: The current session.
      latest_filename: The filename for the latest set of checkpoints, defaults
        to 'checkpoints'.
    Returns:
      True if the model was restored from a checkpoint and False otherwise.
    """
    # Set list of not-yet-deleted checkpoints.
    if self._save_path:
      ckpt = tf.train.get_checkpoint_state(
          os.path.dirname(self._save_path), latest_filename)
      if ckpt and ckpt.all_model_checkpoint_paths:
        # Copy it because last_checkpoints is immutable.
        # Manually configure a new Saver so that we get the latest snapshots.
        self._saver = tf.train.Saver(saver_def=self._saver.as_saver_def())
        self._saver.set_last_checkpoints(list(ckpt.all_model_checkpoint_paths))
    self._create_initializers()
    if self._saver.last_checkpoints:
      self._saver.restore(sess, self._saver.last_checkpoints[-1])
      return self._saver.last_checkpoints[-1]
    else:
      return False

  def _log_and_save(self, sess, results):
    step = results[0]
    to_print = [x for x in results[1:] if x is not None]
    print('[%d] %s' % (step, to_print))
    sys.stdout.flush()
    if self._save_path:
      self._saver.save(sess, self._save_path, step)

  def run_model(self,
                op_list,
                num_steps,
                feed_vars=(),
                feed_data=None,
                print_every=100,
                allow_initialize=True,
                external_coordinator=None):
    """Runs `op_list` for `num_steps`.

    Args:
      op_list: A list of ops to run.
      num_steps: Number of steps to run this for.  If feeds are used, this is a
        maximum.
      feed_vars: The variables to feed.
      feed_data: An iterator that feeds data tuples.
      print_every: Print a log line and checkpoing every so many steps.
      allow_initialize: If True, the model will be initialized if any variable
        is uninitialized, if False the model will not be initialized.
      external_coordinator: If you are managing queuing threads outside of the
        evaluation, then pass the external_coordinator to ensure that they are
        properly coordinated.
    Returns:
      The final run result as a list.
    Raises:
      ValueError: If feed_data doesn't match feed_vars.
    """
    feed_data = feed_data or itertools.repeat(())

    ops = [bookkeeper.global_step()]
    ops.extend(op_list)

    sess = tf.get_default_session()
    self._init_model(sess, allow_initialize)

    if not external_coordinator:
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    else:
      coord = external_coordinator

    try:
      for i, data in zip(xrange(num_steps), feed_data):
        log_this_time = print_every and i % print_every == 0
        if len(data) != len(feed_vars):
          raise ValueError(
              'feed_data and feed_vars must be the same length: %d vs %d' % (
                  len(data), len(feed_vars)))
        if coord.should_stop():
          print('Coordinator stopped')
          sys.stdout.flush()
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
    except tf.errors.OutOfRangeError:
      print('Done training -- epoch limit reached')
    finally:
      # When done, ask the threads to stop.
      if not external_coordinator:
        coord.request_stop()
        if threads:
          # All the code in coordinator looks good, but it sometimes deadlocks.
          time.sleep(1.0)
          coord.join(threads)

    return results

  def train_model(self,
                  train_op,
                  cost_to_log,
                  num_steps,
                  feed_vars=(),
                  feed_data=None,
                  print_every=100,
                  external_coordinator=None):
    """Trains the given model.

    Args:
      train_op: The training operation.
      cost_to_log: A cost to log.
      num_steps: Number of batches to run.
      feed_vars: A list or tuple of the variables that will be fed.
      feed_data: A generator that produces tuples of the same length as
        feed_vars.
      print_every: Print and save every so many steps.
      external_coordinator: If you are managing queuing threads outside of the
        evaluation, then pass the external_coordinator to ensure that they are
        properly coordinated.
    Returns:
      `cost_to_log` from the final step.
    """
    costs = [train_op]
    if (not isinstance(cost_to_log, six.string_types) and
        hasattr(cost_to_log, '__iter__')):
      costs.extend(cost_to_log)
    else:
      costs.append(cost_to_log)
    return self.run_model(costs,
                          num_steps,
                          feed_vars=feed_vars,
                          feed_data=feed_data,
                          print_every=print_every,
                          external_coordinator=external_coordinator)[2:]

  def _run_init_test_vars_op(self):
    test_vars = tf.get_collection(bookkeeper.GraphKeys.TEST_VARIABLES)
    if test_vars:
      if test_vars != self._test_vars:
        self._test_vars = list(test_vars)
        self._test_var_init_op = tf.initialize_variables(test_vars)
      return self._test_var_init_op.run()

  def evaluate_model(self,
                     accuracy,
                     num_steps,
                     feed_vars=(),
                     feed_data=None,
                     summary_tag=None,
                     print_every=0,
                     external_coordinator=None):
    """Evaluates the given model.

    Args:
      accuracy: The metric that is being evaluated.
      num_steps: The number of steps to run in the evaluator.
      feed_vars: A list or tuple of the variables that will be fed.
      feed_data: A generator that produces tuples of the same length as
        feed_vars.
      summary_tag: If provided, the final result of running the model will be
        published to this tag.
      print_every: Print a summary every so many steps, use 0 to disable.
      external_coordinator: If you are managing queuing threads outside of the
        evaluation, then pass the external_coordinator to ensure that they are
        properly coordinated.
    Returns:
      The accuracy.
    """
    self._run_init_test_vars_op()
    result = self.run_model([accuracy],
                            num_steps,
                            feed_vars=feed_vars,
                            feed_data=feed_data,
                            print_every=print_every,
                            allow_initialize=False,
                            external_coordinator=external_coordinator)
    if summary_tag and self._summary_writer:
      summary = tf.Summary(
          value=[tf.Summary.Value(tag=summary_tag,
                                  simple_value=float(result[1]))])
      event = tf.Event(wall_time=time.time(),
                       summary=summary,
                       step=int(result[0]))
      self._summary_writer.add_event(event)
    return result[1]

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
    """
    i = 0
    sess = tf.get_default_session()

    current_checkpoint = self.load_from_checkpoint(sess)
    while not current_checkpoint:
      print('Model not yet available, sleeping for 10 seconds %s.' %
            os.path.dirname(self._save_path))
      sys.stdout.flush()
      time.sleep(10)
      current_checkpoint = self.load_from_checkpoint(sess)

    # Create relevant ops before starting queue runners.
    self._run_init_test_vars_op()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
      while i != evaluation_times:
        i += 1
        accuracy_result = self.evaluate_model(accuracy,
                                              num_steps,
                                              summary_tag=summary_tag,
                                              print_every=0,
                                              feed_vars=feed_vars,
                                              feed_data=feed_data,
                                              external_coordinator=coord)
        print('[%d] %s %g' % (sess.run(bookkeeper.global_step()),
                              summary_tag,
                              accuracy_result))
        while True:
          next_checkpoint = self.load_from_checkpoint(sess)
          if next_checkpoint == current_checkpoint:
            time.sleep(10)
          else:
            break

        current_checkpoint = next_checkpoint
    finally:
      print('Shutting down')
      sys.stdout.flush()
      coord.request_stop()
      time.sleep(1.0)
      coord.join(threads)
