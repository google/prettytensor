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
import itertools
import os
import shutil
import tempfile
import unittest



import numpy
import tensorflow as tf

import prettytensor as pt
from prettytensor import input_helpers
from prettytensor import local_trainer


class LocalTrainerTest(unittest.TestCase):

  def random_numpy(self, shape, dtype):
    if tf.float32.is_compatible_with(dtype):
      size = 1
      for n in shape:
        size *= n
      return self.prng.normal(size=size).astype(numpy.float32).reshape(shape)
    else:
      raise ValueError('This method only supports float32: %s' % dtype)

  def setUp(self):
    self.prng = numpy.random.RandomState(42)
    tf.ops.reset_default_graph()
    self.input = tf.placeholder(tf.float32, [4, 2])
    self.target = tf.placeholder(tf.float32)
    xor_inputs = numpy.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    xor_outputs = numpy.array([[0., 1.], [1., 0.], [0., 1.], [1., 0.]])

    self.xor_data = itertools.cycle(
        input_helpers.feed_numpy(4, xor_inputs, xor_outputs))

    self.softmax_result = (
        pt.wrap(self.input).fully_connected(2,
                                            activation_fn=tf.sigmoid,
                                            init=self.random_numpy)
        .fully_connected(2,
                         activation_fn=None,
                         init=self.random_numpy).softmax(self.target))
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
      self.assertTrue(os.path.isfile(x), 'Promised file not saved: %s' % x)
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
      self.assertEqual(accuracy, 0.5)

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
      self.assertEqual(accuracy, 1.0)

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

if __name__ == '__main__':
  tf.test.main()
