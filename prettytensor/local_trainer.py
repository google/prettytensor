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

import itertools
import os.path
import time

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

  def _create_initializers(self):
    if self._var_count != len(tf.all_variables()):
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
      ckpt = tf.train.get_checkpoint_state(os.path.dirname(self._save_path),
                                           latest_filename)
      if ckpt and ckpt.all_model_checkpoint_paths:
        # Copy it because last_checkpoints is immutable.
        self._saver = tf.train.Saver(saver_def=self._saver.as_saver_def())
        self._saver.set_last_checkpoints(list(
            ckpt.all_model_checkpoint_paths))
    self._create_initializers()
    if self._saver.last_checkpoints:
      self._saver.restore(sess, self._saver.last_checkpoints[-1])
      return True
    else:
      return False

  def _log_and_save(self, sess, results):
    step = results[0]
    to_print = [x for x in results[1:] if x]
    print '[%d] %s' % (step, to_print)
    if self._save_path:
      self._saver.save(sess, self._save_path, step)

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
        maximum.
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
    self._init_model(sess, allow_initialize)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
      for i, data in zip(xrange(num_steps), feed_data):
        log_this_time = print_every and i % print_every == 0
        if len(data) != len(feed_vars):
          raise ValueError(
              'feed_data and feed_vars must be the same length: %d vs %d' % (
                  len(data), len(feed_vars)))
        if coord.should_stop():
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
      print 'Done training -- epoch limit reached'
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()
    coord.join(threads)

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
    if (not isinstance(cost_to_log, basestring) and
        hasattr(cost_to_log, '__iter__')):
      costs.extend(cost_to_log)
    else:
      costs.append(cost_to_log)
    return self.run_model(costs,
                          num_steps,
                          feed_vars=feed_vars,
                          feed_data=feed_data,
                          print_every=print_every)[2:]

  def evaluate_model(self, accuracy, num_steps, feed_vars=(), feed_data=None,
                     summary_tag=None, print_every=0):
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
    Returns:
      The accuracy.
    """
    test_vars = tf.get_collection(bookkeeper.GraphKeys.TEST_VARIABLES)
    if test_vars:
      tf.initialize_variables(test_vars).run()
    result = self.run_model([accuracy],
                            num_steps,
                            feed_vars=feed_vars,
                            feed_data=feed_data,
                            print_every=print_every,
                            allow_initialize=False)
    if summary_tag and self._summary_writer:
      summary = tf.Summary(
          value=[tf.Summary.Value(tag=summary_tag,
                                  simple_value=float(result[1]))])
      event = tf.Event(wall_time=time.time(),
                       summary=summary,
                       step=int(result[0]))
      self._summary_writer.add_event(event)
    return result[1]
