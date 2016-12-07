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
"""Tests for local_trainer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os
import shutil
import tempfile
import threading
import unittest



import numpy
import tensorflow as tf

import prettytensor as pt
from prettytensor import input_helpers
from prettytensor import local_trainer


class LocalTrainerTest(unittest.TestCase):

  def random_numpy(self, shape, dtype, partition_info=None):
    _ = partition_info
    if tf.float32.is_compatible_with(dtype):
      size = 1
      for n in shape:
        size *= n
      return self.prng.normal(size=size).astype(numpy.float32).reshape(shape)
    else:
      raise ValueError('This method only supports float32: %s' % dtype)

  def setUp(self):
    tf.reset_default_graph()
    self.prng = numpy.random.RandomState(42)

    self.input = tf.placeholder(tf.float32, [4, 2])
    self.target = tf.placeholder(tf.float32)
    xor_inputs = numpy.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    xor_outputs = numpy.array([[0., 1.], [1., 0.], [0., 1.], [1., 0.]])

    self.xor_data = itertools.cycle(
        input_helpers.feed_numpy(4, xor_inputs, xor_outputs))

    self.softmax_result = (
        pt.wrap(self.input).fully_connected(2,
                                            activation_fn=tf.sigmoid,
                                            weights=self.random_numpy)
        .fully_connected(2,
                         activation_fn=None,
                         weights=self.random_numpy).softmax(self.target))
    self.tmp_file = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.tmp_file)

  def test_run(self):
    runner = local_trainer.Runner()
    with tf.Session():
      optimizer = tf.train.GradientDescentOptimizer(0.5)
      train_op = pt.apply_optimizer(optimizer,
                                    losses=[self.softmax_result.loss])

      runner.train_model(train_op,
                         self.softmax_result.loss,
                         10,
                         (self.input, self.target),
                         self.xor_data,
                         print_every=2)

  def test_checkpoint(self):
    f = os.path.join(self.tmp_file, 'checkpoint')
    runner = local_trainer.Runner(save_path=f)
    with tf.Session():
      optimizer = tf.train.GradientDescentOptimizer(0.1)
      train_op = pt.apply_optimizer(optimizer,
                                    losses=[self.softmax_result.loss])

      runner.train_model(train_op,
                         self.softmax_result.loss,
                         10,
                         (self.input, self.target),
                         self.xor_data,
                         print_every=2)
    assert runner._saver.last_checkpoints, 'Expected checkpoints.'
    for x in runner._saver.last_checkpoints:
      self.assertTrue(tf.train.checkpoint_exists(x),
                      'Promised file not saved: %s' % x)
      self.assertTrue(x.startswith(f), 'Name not as expected: %s' % x)

  def test_eval(self):
    f = os.path.join(self.tmp_file, 'checkpoint')
    runner = local_trainer.Runner(save_path=f)
    with tf.Session():
      classification_acuracy = self.softmax_result.softmax.evaluate_classifier(
          self.target, phase=pt.Phase.test)

      optimizer = tf.train.GradientDescentOptimizer(0.2)
      train_op = pt.apply_optimizer(optimizer,
                                    losses=[self.softmax_result.loss])

      runner.train_model(train_op,
                         self.softmax_result.loss,
                         100,
                         (self.input, self.target),
                         self.xor_data,
                         print_every=50)
      self.assertTrue(runner._last_init)
      save_paths = list(runner._saver.last_checkpoints)

      # The accuracy should be 50% right now since model is consistently
      # generated.
      accuracy = runner.evaluate_model(classification_acuracy,
                                       1,
                                       (self.input, self.target),
                                       self.xor_data)
      self.assertEquals(runner._saver.last_checkpoints, save_paths,
                        'No additional paths should have been saved.')
      self.assertFalse(runner._last_init)
      self.assertEqual(accuracy, [0.5])

      # Train the model to 100% accuracy.
      runner.train_model(train_op,
                         self.softmax_result.loss,
                         2000,
                         (self.input, self.target),
                         self.xor_data,
                         print_every=1000)
      accuracy = runner.evaluate_model(classification_acuracy, 1,
                                       (self.input, self.target), self.xor_data)
      self.assertFalse(runner._last_init)

      # Make sure that the previous computation didn't impact this eval.
      self.assertEqual(accuracy, [1.0])

  def restore_helper(self, runner):
    with tf.Session():
      classification_acuracy = self.softmax_result.softmax.evaluate_classifier(
          self.target, phase=pt.Phase.test)

      optimizer = tf.train.GradientDescentOptimizer(0.5)
      train_op = pt.apply_optimizer(optimizer,
                                    losses=[self.softmax_result.loss])

      runner.train_model(train_op,
                         self.softmax_result.loss,
                         10,
                         (self.input, self.target),
                         self.xor_data,
                         print_every=2)
      self.assertTrue(runner._last_init)
      self.assertFalse(runner._last_restore)
    with tf.Session():
      save_paths = list(runner._saver.last_checkpoints)
      runner.evaluate_model(classification_acuracy, 1,
                            (self.input, self.target), self.xor_data)
      self.assertEquals(runner._saver.last_checkpoints, save_paths,
                        'No additional paths should have been saved.')
      self.assertFalse(runner._last_init)

  def test_manual_save_restore(self):
    runner = local_trainer.Runner()
    f = os.path.join(self.tmp_file, 'manual.chkpt')

    v = tf.Variable(tf.random_normal(shape=[100], dtype=tf.float32))

    # Save it.
    with runner.session() as sess:
      runner.prepare_model(sess)  # Create variables
      value = v.eval()  # Grab the variable
      runner.saver.save(sess, f)

    with runner.session() as sess:
      # Restore the model
      runner.saver.restore(sess, f)
      new_value = v.eval()
    numpy.testing.assert_array_equal(value, new_value)

  def test_restore(self):
    f = os.path.join(self.tmp_file, 'checkpoint')
    runner = local_trainer.Runner(save_path=f)
    self.restore_helper(runner)
    self.assertTrue(runner._last_restore)

  def test_not_restored(self):
    f = os.path.join(self.tmp_file, 'checkpoint')
    runner = local_trainer.Runner(save_path=f, restore=False)
    with self.assertRaises(tf.errors.FailedPreconditionError):
      self.restore_helper(runner)

  def test_evaluate_without_initialize_error(self):
    with tf.Graph().as_default():
      runner = local_trainer.Runner()
      tf.Variable(1)  # Put a variable in the graph.

      with runner.session(), self.assertRaises(ValueError):
        runner.evaluate_model(
            self.softmax_result, 1, (self.input, self.target), self.xor_data)

  def test_evaluate_repeatedly_one_time(self):
    f = os.path.join(self.tmp_file, 'checkpoint')
    runner = local_trainer.Runner(save_path=f)
    self.restore_helper(runner)
    local_variable = tf.Variable(22, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    accuracy = local_variable.assign_add(1)

    answer = runner.evaluate_repeatedly(accuracy, 20, evaluation_times=1)
    self.assertEqual([42], answer)

  def test_queues(self):
    qr = FakeQueueRunner()
    tf.train.add_queue_runner(qr)
    runner = local_trainer.Runner()
    with tf.Session():
      optimizer = tf.train.GradientDescentOptimizer(0.5)
      train_op = pt.apply_optimizer(optimizer,
                                    losses=[self.softmax_result.loss])

      runner.train_model(train_op,
                         self.softmax_result.loss,
                         100,
                         (self.input, self.target),
                         self.xor_data,
                         print_every=2)
    with tf.Session():
      with self.assertRaisesRegexp(ValueError, r'.*\bstop_queues\b.*'):
        runner.train_model(train_op,
                           self.softmax_result.loss,
                           100,
                           (self.input, self.target),
                           self.xor_data,
                           print_every=2)

    runner.stop_queues()
    qr.assert_worked(self)

  def test_queue_error(self):
    qr = FakeQueueRunner(RuntimeError('expected'))
    tf.train.add_queue_runner(qr)
    runner = local_trainer.Runner()
    with tf.Session():
      optimizer = tf.train.GradientDescentOptimizer(0.5)
      train_op = pt.apply_optimizer(optimizer,
                                    losses=[self.softmax_result.loss])

      with self.assertRaisesRegexp(RuntimeError, 'expected'):
        runner.train_model(train_op,
                           self.softmax_result.loss,
                           100,
                           (self.input, self.target),
                           self.xor_data,
                           print_every=2)
    qr.assert_worked(self)


class FakeQueueRunner(object):
  called = 0
  stopped = False

  def __init__(self, error=None):
    self.error = error

  def create_threads(self, sess, coord=None, daemon=False, start=False):  # pylint: disable=unused-argument
    self.called += 1
    threads = [threading.Thread(target=self.set_stopped, args=(coord,))]
    if self.error:
      threads.append(threading.Thread(target=self.die,
                                      args=(coord, self.error)))
    if start:
      for t in threads:
        t.start()
    return threads

  def die(self, coord, error):
    try:
      raise error
    except RuntimeError as e:
      coord.request_stop(e)

  def set_stopped(self, coord):
    coord.wait_for_stop()
    self.stopped = True

  def assert_worked(self, test):
    test.assertEqual(1, self.called)
    test.assertTrue(self.stopped)

if __name__ == '__main__':
  tf.test.main()
