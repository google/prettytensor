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
"""Test class for the recurrent networks module."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest



import numpy
from numpy import testing
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin
import tensorflow as tf

import prettytensor
from prettytensor import pretty_tensor_testing
from prettytensor import recurrent_networks


TOLERANCE = 0.00001


class SequenceInputMock(object):

  def __init__(self, bookkeeper, input_list, label_list, node_depth):
    self.inputs = self.for_constants(input_list)
    self.targets = self.for_constants(label_list)
    self.node_depth = node_depth
    self.batch_size = input_list[0].shape[0]
    self.requested_tensors = {}
    self.bookkeeper = bookkeeper
    self.num_timesteps = len(input_list)

  def for_constants(self, ls):
    return [tf.constant(x, dtype=tf.float32) for x in ls]

  def state(self, state_name):
    if state_name not in self.requested_tensors:
      count = tf.get_variable('count_%s' % state_name,
                              [],
                              tf.int32,
                              tf.zeros_initializer,
                              trainable=False)
      value = tf.get_variable(state_name, [self.batch_size, self.node_depth],
                              tf.float32, tf.zeros_initializer)
      self.requested_tensors[state_name] = (count, value)

    return self.requested_tensors[state_name][1]

  def save_state(self, state_name, unused_value, name='SaveState'):
    return tf.assign_add(self.requested_tensors[state_name][0], 1, name=name)


class RecurrentNetworksTest(pretty_tensor_testing.PtTestCase):

  def setUp(self):
    super(self.__class__, self).setUp()
    self.input_data = numpy.array(
        [
            [[1.], [2.], [3.], [4.]], [[5.], [6.], [7.], [8.]],
            [[5.], [6.], [7.], [8.]], [[9.], [10.], [11.], [12.]]
        ],
        dtype=numpy.float)
    self.sequence = SequenceInputMock(self.bookkeeper, self.input_data,
                                      [[0.], [0.], [0.], [1.]], 13)

    self.input, self.output = recurrent_networks.create_sequence_pretty_tensor(
        self.sequence)

  def testSquashAndCleave(self):
    squashed = self.input.squash_sequence()
    result = self.RunTensor(squashed)

    testing.assert_allclose(
        self.input_data.reshape(16, 1),
        result,
        rtol=TOLERANCE)

    result = self.RunTensor(squashed.cleave_sequence())

    for i in xrange(len(self.input_data)):
      testing.assert_allclose(
          self.input_data[i], result[i],
          rtol=TOLERANCE)

  def testSquashAndCleaveLength1(self):
    input_data = numpy.array(
        [[[1.], [2.], [3.], [4.]]], dtype=numpy.float)
    sequence = SequenceInputMock(self.bookkeeper, input_data, [[0.]], 13)

    inp, _ = recurrent_networks.create_sequence_pretty_tensor(sequence)
    squashed = inp.squash_sequence()
    result = self.RunTensor(squashed)

    testing.assert_allclose(
        input_data.reshape(4, 1), result,
        rtol=TOLERANCE)

    result = self.RunTensor(squashed.cleave_sequence())

    testing.assert_allclose(input_data[0], result[0], rtol=TOLERANCE)
    self.assertEquals(1, len(result))

  def testSequenceLstm(self):
    lstm = self.input.sequence_lstm(13)
    result = self.RunTensor(lstm)

    self.assertEquals([4, 13], lstm.shape)
    self.assertEquals(4, len(result))
    for i in xrange(4):
      self.assertSequenceEqual(lstm.shape, result[i].shape)

  def testSequenceGru(self):
    gru = self.input.sequence_gru(13)
    result = self.RunTensor(gru)

    self.assertEquals([4, 13], gru.shape)
    self.assertEquals(4, len(result))
    for i in xrange(4):
      self.assertSequenceEqual(gru.shape, result[i].shape)

  def performTestArbitraryBatchSizeRnn(self, cell_type):
    # Tests whether LSTM / GRU / Bookkeeper function when batch_size is not
    # specified at graph creation time (i.e., None).
    self.assertTrue(cell_type == 'lstm' or cell_type == 'gru')
    super(self.__class__, self).SetBookkeeper(
        prettytensor.bookkeeper_for_new_graph())

    # Build a graph. Specify None for the batch_size dimension.
    placeholder = tf.placeholder(tf.float32, [None, 1])
    input_pt = prettytensor.wrap_sequence([placeholder])
    if cell_type == 'lstm':
      output, _ = (input_pt
                   .sequence_lstm(4)
                   .squash_sequence()
                   .softmax_classifier(2))
    elif cell_type == 'gru':
      output, _ = (input_pt
                   .sequence_gru(4)
                   .squash_sequence()
                   .softmax_classifier(2))

    self.sess.run(tf.global_variables_initializer())

    # Use RecurrentRunner for state saving and managing feeds.
    recurrent_runner = recurrent_networks.RecurrentRunner(batch_size=1)

    # Run with a batch size of 1 for 10 steps, save output for reference.
    out_orig = []
    for t in xrange(10):
      outs = recurrent_runner.run(
          [output.name],
          {placeholder.name: numpy.array([[1.2]])},
          sess=self.sess)
      out = outs[0]
      self.assertEqual(1, len(out))
      self.assertEqual(2, len(out[0]))
      out_orig.append(out[0])

    # Test the reset functionality - after a reset, the results must be
    # identical to what we just got above.
    recurrent_runner.reset()
    for t in xrange(10):
      outs = recurrent_runner.run(
          [output.name],
          {placeholder.name: numpy.array([[1.2]])},
          sess=self.sess)
      out = outs[0]
      self.assertEqual(1, len(out))
      self.assertEqual(2, len(out[0]))
      testing.assert_allclose(out[0], out_orig[t])

    # Test whether the recurrent runner detects changes to the default graph.
    # It should raise an Assertion because RecurrentRunner's state saver
    # information (collected during __init__) is not valid anymore.
    with tf.Graph().as_default():
      placeholder2 = tf.placeholder(tf.float32, [None, 1])
      input_pt2 = prettytensor.wrap_sequence([placeholder2])
      if cell_type == 'lstm':
        output2, _ = (input_pt2
                      .sequence_lstm(4)
                      .squash_sequence()
                      .softmax_classifier(2))
      elif cell_type == 'gru':
        output2, _ = (input_pt2
                      .sequence_gru(4)
                      .squash_sequence()
                      .softmax_classifier(2))
      self.assertRaises(ValueError,
                        recurrent_runner.run,
                        [output2.name], None, self.sess)

    # Run with a batch size of 3; first and third input are identical and must
    # yield identical output, and the same output as in the single batch run
    # above (up to floating point rounding errors).
    recurrent_runner = recurrent_networks.RecurrentRunner(batch_size=3)
    for t in xrange(10):
      outs = recurrent_runner.run(
          [output.name],
          {placeholder.name: numpy.array([[1.2], [3.4], [1.2]])},
          sess=self.sess)
      out = outs[0]
      self.assertEqual(3, len(out))
      self.assertEqual(2, len(out[0]))
      testing.assert_allclose(out[0], out[2], rtol=TOLERANCE)
      testing.assert_allclose(out[0], out_orig[t], rtol=TOLERANCE)
      # Sanity check to protect against trivial outputs that might hide errors.
      # Need to avoid checking after t = 2 since untrained GRUs have a
      # tendency to converge to large state values, leading to outputs like
      # 1.0, 0.0.
      if cell_type == 'gru' and t > 2:
        continue
      self.assertFalse((out[0] == out[1]).all())

  def testArbitraryBatchSizeLstm(self):
    self.performTestArbitraryBatchSizeRnn('lstm')

  def testArbitraryBatchSizeGru(self):
    self.performTestArbitraryBatchSizeRnn('gru')

  def testSequence(self):
    result = self.RunTensor(self.input[-1])
    testing.assert_allclose(
        self.input_data[-1], result,
        rtol=TOLERANCE)

  def testEmbeddingNameWorkaround(self):
    """Just make sure this runs since it is ensuring that a workaround works."""
    input_data = self.Wrap(self.input_data.astype(numpy.int32).reshape([16, 1]))
    result = input_data.embedding_lookup(
        13, [1], name='params')
    self.RunTensor(result)

  def testEmbeddingLookupRequiresRank2(self):
    """Just make sure this runs since it is ensuring that a workaround works."""
    input_data = self.Wrap(self.input_data.astype(numpy.int32))
    with self.assertRaises(ValueError):
      input_data.embedding_lookup(13, [1], name='params')

  def testLstmStateTuples(self):
    self.states = recurrent_networks.lstm_state_tuples(13, 'blah')
    self.RunTensor(
        self.input.sequence_lstm(13, name='blah')[-1])

    for state in self.states:
      self.assertTrue(
          state[0] in self.sequence.requested_tensors, '%s missing: %s' %
          (state[0], list(six.iterkeys(self.sequence.requested_tensors))))
    self.assertEqual(
        len(self.states), len(self.sequence.requested_tensors),
        'Wrong number of Tensor states.')

  def testGruStateTuples(self):
    self.states = recurrent_networks.gru_state_tuples(13, 'blah')
    self.RunTensor(
        self.input.sequence_gru(13, name='blah')[-1])

    for state in self.states:
      self.assertTrue(
          state[0] in self.sequence.requested_tensors, '%s missing: %s' %
          (state[0], list(six.iterkeys(self.sequence.requested_tensors))))
    self.assertEqual(
        len(self.states), len(self.sequence.requested_tensors),
        'Wrong number of Tensor states.')

  def testLength(self):
    tf.set_random_seed(4321)
    with tf.variable_scope('test') as vs:
      base_lstm = self.input.sequence_lstm(13)
    lengths = tf.placeholder(dtype=tf.int32, shape=[4])

    # Use the same parameters.
    with tf.variable_scope(vs, reuse=True):
      lstm_truncated = self.input.sequence_lstm(13, lengths=lengths)

    with tf.Session() as sess:
      tf.global_variables_initializer().run()

      result = sess.run(base_lstm.sequence + lstm_truncated.sequence,
                        {lengths: [10, 4, 1, 1]})
      base_result = result[:len(base_lstm.sequence)]
      full_result = result[len(base_lstm.sequence):]
      truncated_result = sess.run(lstm_truncated.sequence,
                                  {lengths: [1, 2, 1, 1]})

    for i, (x, y) in enumerate(zip(base_result, truncated_result)):
      if i < 2:
        testing.assert_allclose(x, y, rtol=TOLERANCE)
      else:
        # After the specified output, we check to make sure the same values are
        # propagated forward.
        self.assertFalse(numpy.allclose(x, y, rtol=TOLERANCE))
        testing.assert_allclose(y, truncated_result[i - 1], rtol=TOLERANCE)
    for x, y in zip(base_result, full_result):
      # The later results tend to diverge.  This is something that requires
      # investigation.
      testing.assert_allclose(x, y, atol=0.1)


if __name__ == '__main__':
  unittest.main()
