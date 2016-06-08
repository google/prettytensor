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
"""Test class for PrettyTensor."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import operator
import unittest



import numpy
from numpy import testing
import tensorflow as tf

import prettytensor
from prettytensor import DIM_SAME
from prettytensor import Phase
from prettytensor import pretty_tensor_class
from prettytensor import pretty_tensor_methods
from prettytensor import pretty_tensor_testing

TOLERANCE = 0.000001


# This name is stochastically flagged by pylint....
# pylint: disable=invalid-name
@prettytensor.Register
def AssertHead(head, expected, test):
  test.assertEqual(expected, head)
  return head


class PrettyTensorTest(pretty_tensor_testing.PtTestCase):

  def setUp(self):
    super(self.__class__, self).setUp()
    # Input is 2x3x5, which isn't a natural size for any op.
    self.input_data = numpy.array(
        [[[1, 2, 3, 4, 5],
          [6, 7, 8, 9, 10],
          [10, 12, 13, 14, 15],], [[-1, 2, -3, 4, -5], [6, -7, 8, -9, 10],
                                   [-10, 12, -13, 14, -15]]],
        dtype=numpy.float)
    self.input = tf.constant(self.input_data,
                             shape=list(self.input_data.shape),
                             dtype=tf.float32)
    self.input_layer = self.Wrap(self.input, self.input_data.shape)

  def testInit(self):
    """Tests that the test initiliazes and input_layer is runnable."""
    out = self.RunTensor(self.input_layer)
    testing.assert_allclose(out, self.input_data, rtol=TOLERANCE)

  def testFlatten(self):
    flattened = self.input_layer.flatten()
    out = self.RunTensor(flattened)
    testing.assert_allclose(
        self.input_data.reshape([2, 15]),
        out,
        rtol=TOLERANCE)

  def testBranch(self):
    # Explicit branching is not required, but make sure it works.
    flattened = self.input_layer.flatten()

    self.assertNotEqual(self.input_layer.tensor, flattened.tensor)

  def testDefaultJoinFunction(self):
    pt1 = self.Wrap(numpy.array([[1.0], [2.0]]))
    pt2 = self.Wrap(numpy.array([[2.0, 3.0], [4.0, 5.0]]))
    pt3 = self.Wrap(numpy.array([[5.0], [6.0]]))
    out_pt = prettytensor.join_pretty_tensors([pt1, pt2, pt3], pt1)

    out = self.RunTensor(out_pt)

    testing.assert_allclose(
        out,
        numpy.array([[1.0, 2.0, 3.0, 5.0], [2.0, 4.0, 5.0, 6.0]]),
        rtol=TOLERANCE)

  def testDefaultJoinMethod(self):
    pt1 = self.Wrap(numpy.array([[1.0], [2.0]]))
    pt2 = self.Wrap(numpy.array([[2.0, 3.0], [4.0, 5.0]]))
    pt3 = self.Wrap(numpy.array([[5.0], [6.0]]))
    out_pt = pt1.join([pt2, pt3])

    out = self.RunTensor(out_pt)

    testing.assert_allclose(
        out,
        numpy.array([[1.0, 2.0, 3.0, 5.0], [2.0, 4.0, 5.0, 6.0]]),
        rtol=TOLERANCE)

  def testDefaultJoinAlignmentError(self):
    pt1 = self.Wrap(numpy.array([[1.0], [2.0]]))
    pt2 = self.Wrap(numpy.array([[2.0, 3.0], [4.0, 5.0]]))
    pt3 = self.Wrap(numpy.array([5.0, 6.0]))

    with self.assertRaises(ValueError):
      prettytensor.join_pretty_tensors([pt1, pt2, pt3], pt1)

  def testCustomJoin(self):
    pt1 = self.Wrap(numpy.array([[1.0], [2.0]]))
    pt2 = self.Wrap(numpy.array([[2.0], [4.0]]))
    pt3 = self.Wrap(numpy.array([[5.0], [6.0]]))
    out_pt = prettytensor.join_pretty_tensors(
        [pt1, pt2, pt3],
        pt1,
        join_function=tf.add_n)

    out = self.RunTensor(out_pt)

    testing.assert_allclose(out, numpy.array([[8.0], [12.0]]), rtol=TOLERANCE)

  def testShapeSpecWithPlaceHoldersSame(self):
    reshaped = self.input_layer.reshape([DIM_SAME, 15])
    self.assertEqual([2, 15], reshaped.shape)

    out = self.RunTensor(reshaped)
    self.assertEqual((2, 15), out.shape)

  def testShapeSpecWithPlaceHoldersOutOfBounds(self):
    with self.assertRaises(ValueError):
      self.input_layer.reshape([DIM_SAME, DIM_SAME, DIM_SAME, DIM_SAME])

  def testShapeSpecWithPlaceHoldersRest(self):
    reshaped = self.input_layer.reshape([DIM_SAME, -1])
    self.assertEqual([2, 15], reshaped.shape)

    out = self.RunTensor(reshaped)
    self.assertEqual((2, 15), out.shape)

  def testShapeSpecWithPlaceHoldersRestInMiddle(self):
    reshaped = self.input_layer.reshape([DIM_SAME, -1, 1])
    self.assertEqual([2, 15, 1], reshaped.shape)

    out = self.RunTensor(reshaped)
    self.assertEqual((2, 15, 1), out.shape)

  def testDimRestErrors(self):
    with self.assertRaises(ValueError):
      pretty_tensor_methods._infer_unknown_dims([2, 3, 5], [DIM_SAME, -1, 2])
    with self.assertRaises(ValueError):
      pretty_tensor_methods._infer_unknown_dims([2, 3, 5], [DIM_SAME, -1, -1])
    with self.assertRaises(ValueError):
      pretty_tensor_methods._infer_unknown_dims([2, 3, 5], [DIM_SAME, 14])

  def testShapeSpecWithPlaceHoldersCompact(self):
    reshaped = self.input_layer.reshape('__5')
    self.assertEqual([2, 3, 5], reshaped.shape)

    out = self.RunTensor(reshaped)
    self.assertEqual((2, 3, 5), out.shape)

  def testSequential(self):
    st = self.input_layer.sequential()
    st.flatten()
    out = self.RunTensor(st)
    testing.assert_allclose(
        self.input_data.reshape([2, 15]),
        out,
        rtol=TOLERANCE)

  def testSubdivide(self):
    st = self.input_layer.sequential()
    st.flatten()
    with st.subdivide(2) as [a, b]:
      a.fully_connected(10)
      b.fully_connected(20)

    self.assertEqual([2, 30], st.shape, 'Unexpected shape.')
    self.sess.run(tf.initialize_all_variables())
    combined, a_val, b_val = self.sess.run([st.tensor, a.tensor, b.tensor])
    testing.assert_allclose(combined[:, 0:10], a_val, rtol=TOLERANCE)
    testing.assert_allclose(combined[:, 10:30], b_val, rtol=TOLERANCE)

  def testSubdivideWith(self):
    st = self.input_layer.sequential()
    st.flatten()
    with st.subdivide_with(2, tf.add_n) as [a, b]:
      a.fully_connected(10)
      b.fully_connected(10)

    self.assertEqual([2, 10], st.shape, 'Unexpected shape: %s' % st.shape)
    self.sess.run(tf.initialize_all_variables())
    combined, a_val, b_val = self.sess.run([st.tensor, a.tensor, b.tensor])
    testing.assert_allclose(combined, a_val + b_val, rtol=TOLERANCE)

  def testSoftmax(self):
    input_data = self.input_data.reshape([2, 15])
    label = numpy.zeros_like(input_data)
    label[0, 5] = 1
    label[1, 10] = 1
    label_tensor = tf.constant(label, shape=list(label.shape), dtype=tf.float32)
    layer, cost = self.input_layer.flatten().softmax(label_tensor)

    softmax_out, cost_out = self.sess.run([layer.tensor, cost.tensor])
    expected_softmax = numpy.exp(input_data)
    expected_softmax /= expected_softmax.sum(axis=1, keepdims=True)

    testing.assert_allclose(expected_softmax, softmax_out, rtol=TOLERANCE)
    self.assertSequenceEqual((), cost_out.shape)

  def testMaxPooling(self):
    st = self.input_layer.sequential()
    st.reshape([DIM_SAME, DIM_SAME, DIM_SAME, 1])
    st.max_pool(3, 2)
    result = self.RunTensor(st)

    self.assertSequenceEqual(result.shape, st.shape,
                             'Unexpected shape: %s vs %s' %
                             (result.shape, st.shape))
    expected = numpy.array([[[[7.], [9.], [10.]], [[12.], [14.], [15.]]],
                            [[[6.], [8.], [10.]], [[12.], [14.], [14.]]]])
    testing.assert_allclose(expected, result, rtol=TOLERANCE)

  def testMaxPoolingValid(self):
    st = self.input_layer.sequential()
    st.reshape([DIM_SAME, DIM_SAME, DIM_SAME, 1])
    st.max_pool(3, 2, prettytensor.PAD_VALID)
    result = self.RunTensor(st)

    self.assertSequenceEqual(result.shape, st.shape,
                             'Unexpected shape: %s vs %s' %
                             (result.shape, st.shape))

  def testConv(self):
    st = self.input_layer.sequential()
    st.reshape([DIM_SAME, DIM_SAME, DIM_SAME, 1])
    st.conv2d(3, 2)
    result = self.RunTensor(st)
    self.assertEqual(2, result.shape[-1])
    self.assertTrue(st.layer_parameters['weights'])
    self.assertTrue(st.layer_parameters['bias'])

  def testUnknownShapeConv(self):
    input_data = tf.placeholder(tf.float32)

    with self.assertRaises(ValueError):
      self.Wrap(input_data).conv2d([3, 3], 100)

  def testDepthwiseConv(self):
    # Fixes random number generator and uses a small standard deviation for
    # initializing convolution weights.
    rng = numpy.random.RandomState(1024)
    std = 0.01
    # Depthwise conv2d with input of depth 1 will produce identical result as
    # conv2d.
    pt1 = self.input_layer.reshape([DIM_SAME, DIM_SAME, DIM_SAME, 1])
    weight_init = std * rng.randn(3, 3, 1, 2).astype(numpy.float32)
    pt1 = pt1.depthwise_conv2d(3, 2, init=tf.constant_initializer(weight_init))
    result_pt1 = self.RunTensor(pt1)

    pt2 = self.input_layer.reshape([DIM_SAME, DIM_SAME, DIM_SAME, 1])
    pt2 = pt2.conv2d(3, 2, init=tf.constant_initializer(weight_init))
    result_pt2 = self.RunTensor(pt2)
    testing.assert_allclose(result_pt1, result_pt2, rtol=TOLERANCE)

    # Channel multiplier 3 depthwise conv2d with input of depth 2.
    pt3 = self.input_layer.reshape([1, 3, 5, 2])
    weight_init = std * rng.randn(3, 3, 2, 3).astype(numpy.float32)
    pt3 = pt3.depthwise_conv2d(3, 3, init=tf.constant_initializer(weight_init))
    result_pt3 = self.RunTensor(pt3)
    self.assertEqual(result_pt3.shape, (1, 3, 5, 6))

    # Reference conv2d with kernel shape (3, 3, 2, 3).
    pt4 = self.input_layer.reshape([1, 3, 5, 2])
    pt4 = pt4.conv2d(3, 3, init=tf.constant_initializer(weight_init))
    result_pt4 = self.RunTensor(pt4)
    self.assertEqual(result_pt4.shape, (1, 3, 5, 3))
    # conv2d response should match the sum of corresponding depthwise conv2d
    # response. Uses absolute tolerance here due to float32 precision.
    for depth in range(3):
      testing.assert_allclose(
          result_pt4[..., depth],
          result_pt3[..., depth] + result_pt3[..., 3 + depth],
          atol=TOLERANCE)

  def testConvBatchNorm(self):
    st = self.input_layer.sequential()
    st.reshape([DIM_SAME, DIM_SAME, DIM_SAME, 1])
    with prettytensor.defaults_scope(batch_normalize=True,
                                     learned_moments_update_rate=0.0003,
                                     variance_epsilon=0.001,
                                     scale_after_normalization=True):
      st.conv2d(3, 2)
    self.assertEqual(2,
                     len(tf.get_collection(prettytensor.GraphKeys.UPDATE_OPS)))
    self.assertTrue(tf.get_collection(tf.GraphKeys.VARIABLES, '.*/beta'))
    self.assertTrue(tf.get_collection(tf.GraphKeys.VARIABLES, '.*/gamma'))
    self.assertTrue(tf.get_collection(
        tf.GraphKeys.VARIABLES, '.*/moving_variance'))
    self.assertTrue(tf.get_collection(tf.GraphKeys.VARIABLES, '.*/moving_mean'))

  def testConvBatchNormArgumentOverride(self):
    st = self.input_layer.sequential()
    st.reshape([DIM_SAME, DIM_SAME, DIM_SAME, 1])
    with prettytensor.defaults_scope(batch_normalize=True,
                                     learned_moments_update_rate=0.0003,
                                     variance_epsilon=0.001,
                                     scale_after_normalization=True):
      st.conv2d(3, 2,
                batch_normalize=prettytensor.BatchNormalizationArguments(
                    scale_after_normalization=False))
    self.assertEqual(2,
                     len(tf.get_collection(prettytensor.GraphKeys.UPDATE_OPS)))
    self.assertTrue(tf.get_collection(tf.GraphKeys.VARIABLES, '.*/beta'))
    self.assertFalse(tf.get_collection(tf.GraphKeys.VARIABLES, '.*/gamma'))
    self.assertTrue(tf.get_collection(
        tf.GraphKeys.VARIABLES, '.*/moving_variance'))
    self.assertTrue(tf.get_collection(tf.GraphKeys.VARIABLES, '.*/moving_mean'))

  def testBatchNormalizeUpdatesGraph(self):

    x = prettytensor.wrap((numpy.arange(20)).reshape(10, 2).astype(
        numpy.float32))
    y = x.batch_normalize()

    self.RunTensor(y)
    mean = tf.get_collection(tf.GraphKeys.VARIABLES, '.*/moving_mean')[0]
    var = tf.get_collection(tf.GraphKeys.VARIABLES, '.*/moving_variance')[0]

    testing.assert_allclose([0.9, 1.], self.sess.run(mean), rtol=TOLERANCE)
    testing.assert_allclose([4.200001, 4.200001],
                            self.sess.run(var), rtol=TOLERANCE)

  def testConvBadShape(self):
    with self.assertRaises(ValueError):
      self.input_layer.conv2d(3, 2)

  def testFullBadShape(self):
    with self.assertRaises(ValueError):
      self.input_layer.fully_connected(20)

  def testFull(self):
    st = self.input_layer.sequential()
    st.flatten()
    st.fully_connected(20)
    result = self.RunTensor(st)
    self.assertEqual(20, result.shape[1])
    self.assertTrue(st.layer_parameters['weights'])
    self.assertTrue(st.layer_parameters['bias'])

  def testFullWithWeightMatrix(self):
    weights = numpy.array([[1., 0.], [0., 1.]])
    in_ = numpy.array([[5., 6.]])
    st = prettytensor.wrap(in_).fully_connected(2, init=weights)
    result = self.RunTensor(st)
    testing.assert_allclose(in_, result, rtol=TOLERANCE)

  def testFullTranspose(self):
    st = self.input_layer.sequential()
    st.flatten()
    st.fully_connected(20, transpose_weights=True)
    result = self.RunTensor(st)
    self.assertEqual(20, result.shape[1])
    self.assertTrue(st.layer_parameters['weights'])
    self.assertTrue(st.layer_parameters['bias'])

  def testFullWithVariableStart(self):
    input_layer = prettytensor.wrap(tf.Variable(self.input_data))

    st = input_layer.sequential()
    st.flatten()
    st.fully_connected(20)
    result = self.RunTensor(st)
    self.assertEqual(20, result.shape[1])

  def MultiLayer(self):
    """Builds a multi layer network to verify the impact of scopes."""
    input_layer = prettytensor.wrap(self.input_layer)

    st = input_layer.sequential()
    st.reshape([DIM_SAME, DIM_SAME, DIM_SAME, 1])
    st.conv2d([3, 3], 10, tf.nn.relu)
    st.flatten()
    st.fully_connected(20)
    st.softmax_classifier(2,
                          labels=numpy.array(
                              [[1, 0], [0, 1]],
                              dtype=numpy.float32))

  def testNoSummaries(self):
    with prettytensor.defaults_scope(summary_collections=None):
      self.MultiLayer()
    self.assertEqual([], tf.get_collection(tf.GraphKeys.SUMMARIES))

  def testVariableCollections(self):
    with prettytensor.defaults_scope(variable_collections=['a']):
      self.MultiLayer()
    self.assertTrue(tf.get_collection('a'))
    self.assertEqual(
        tf.get_collection('a'),
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    self.assertTrue(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

  def testVariableNotTrainable(self):
    with prettytensor.defaults_scope(trainable_variables=False):
      self.MultiLayer()
    self.assertFalse(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

  def testChainFull(self):
    chained = self.input_layer.flatten().fully_connected(20).fully_connected(
        30).fully_connected(40)
    result = self.RunTensor(chained)

    self.assertEqual(40, result.shape[1])

  def testAvgPooling(self):
    st = self.input_layer.sequential()
    st.reshape([DIM_SAME, DIM_SAME, DIM_SAME, 1])
    st.average_pool(3, 2)
    result = self.RunTensor(st)

    expected = numpy.array([[[[4.], [5.5], [7.]], [[8.75], [10.5], [12.]]], [
        [[0.], [-0.83333331], [0.]], [[0.25], [0.83333331], [0.]]
    ]])
    testing.assert_allclose(expected, result, rtol=TOLERANCE)

  def testDropoutTraining(self):
    dropout = self.input_layer.dropout(0.0001)
    result = self.RunTensor(dropout)
    zero_count = 0
    flat = result.flatten()
    for x in flat:
      if math.fabs(x) < TOLERANCE:
        zero_count += 1
    # Hopefully 5% tolerance is enough to never have this accidently fail....
    self.assertGreater(zero_count, len(flat) * 0.95)

  def testDropoutNotTraining(self):
    with prettytensor.defaults_scope(phase=Phase.test):
      dropout = self.input_layer.dropout(0.0001)
    result = self.RunTensor(dropout)

    testing.assert_allclose(self.input_data, result, rtol=TOLERANCE)

  def testL2Regression(self):
    label = numpy.zeros_like(self.input_data)
    label[0, 1, 1] = 100
    label[1, 2, 3] = 100
    label_tensor = tf.constant(label, shape=list(label.shape), dtype=tf.float32)
    cost = self.input_layer.l2_regression(label_tensor)

    cost_out = self.RunTensor(cost)
    expected = ((self.input_data - label)**2).sum() / len(label)

    testing.assert_allclose(expected, cost_out, rtol=TOLERANCE)

  # pylint: disable=unused-variable,invalid-name
  def testMethodRegistration(self):

    @prettytensor.Register()
    def test_method1(input_layer):
      self.assertEqual(self.input_layer, input_layer)
      return tf.constant('success')

    result = self.RunTensor(self.input_layer.test_method1())
    self.assertEqual(b'success', result)

  def testMethodRegistrationWithDefaults(self):
    # pylint: disable=unused-variable,invalid-name
    @prettytensor.Register(assign_defaults='funny_name')
    def test_method3(_, funny_name='not none'):
      return tf.constant(funny_name)

    result = self.RunTensor(self.input_layer.test_method3())
    self.assertEqual(b'not none', result)
    result = self.RunTensor(self.input_layer.test_method3(funny_name='other'))
    self.assertEqual(b'other', result)
    with prettytensor.defaults_scope(funny_name='something'):
      result = self.RunTensor(self.input_layer.test_method3())
      self.assertEqual(b'something', result)
      result = self.RunTensor(self.input_layer.test_method3(funny_name='other'))
      self.assertEqual(b'other', result)

  def testMethodRegistrationRepeated(self):
    # pylint: disable=unused-variable,invalid-name
    @prettytensor.Register()
    def test_method4(_):
      pass
    with self.assertRaises(AssertionError):

      @prettytensor.Register(method_name='test_method4')
      def test_method4_repeat(_):
        pass

  def testMethodRegistrationClashesWithLayerWrapper(self):
    # pylint: disable=unused-variable,invalid-name
    with self.assertRaises(AssertionError):

      @prettytensor.Register()
      def tensor(_):
        pass

  def testNamedMethodRegistration(self):
    # pylint: disable=unused-variable,invalid-name
    @prettytensor.Register(method_name='another_test')
    def test_method5(input_layer):
      self.assertEqual(self.input_layer, input_layer)
      return tf.constant('success')

    result = self.RunTensor(self.input_layer.another_test())
    self.assertEqual(b'success', result)

  def testFunctionRegistration(self):
    # pylint: disable=unused-variable,invalid-name
    @prettytensor.Register()
    def test_function1(tensor):
      return tf.nn.relu(tensor)

    test_result = self.input_layer.test_function1()

    result = self.RunTensor(test_result)
    expected = (self.input_data > 0) * self.input_data
    testing.assert_allclose(expected, result, rtol=TOLERANCE)

  def testFunctionRegistrationWithoutParans(self):
    # pylint: disable=unused-variable,invalid-name
    @prettytensor.Register
    def test_function3(tensor):
      return tf.reshape(tensor, [-1])

    result = self.input_layer.test_function3()
    self.assertEqual(result.shape, [30])

  def testUnusedDefaults(self):
    with self.assertRaises(ValueError):
      self.input_layer.with_defaults(garbage=None)

  def testArgumentSpec(self):
    # pylint: disable=unused-variable,invalid-name,unused-argument
    @prettytensor.Register()
    def test_function4(tensor, one, two=None, three=True):
      """I am a doc string."""
      return tf.reshape(tensor, [-1]), '*'

    self.assertEqual(
        self.input_layer.test_function4.__doc__,
        'test_function4(one, two=None, three=True)\n\nI am a doc string.')

  def testNestedDefaultScope(self):
    pretty_tensor_class._defaults['l2loss'] = 0
    with prettytensor.defaults_scope(l2loss=0.001):
      self.assertEqual(0.001, pretty_tensor_class._defaults['l2loss'])
      with prettytensor.defaults_scope(l2loss=5):
        self.assertEqual(5, pretty_tensor_class._defaults['l2loss'])
      self.assertEqual(0.001, pretty_tensor_class._defaults['l2loss'])
    self.assertEqual(0, pretty_tensor_class._defaults['l2loss'])

  def testUnknownBatchDim(self):
    shape = list(self.input_data.shape)
    shape[0] = None
    input_data = tf.placeholder(tf.float32, shape)

    nn = self.Wrap(input_data).flatten().fully_connected(100).fully_connected(
        200)
    self.assertEquals([None, 200], nn.tensor.get_shape().as_list())

  def testUnknownShapeFullyConnected(self):
    input_data = tf.placeholder(tf.float32)

    with self.assertRaises(ValueError):
      self.Wrap(input_data).fully_connected(100)

  def testPartiallyUnknownShape(self):
    shape = list(self.input_data.shape)
    shape[1] = None
    input_data = tf.placeholder(tf.float32, shape)

    flat = self.Wrap(input_data).flatten()
    self.assertEqual([shape[0], None], flat.shape)

    with self.assertRaises(ValueError):
      flat.fully_connected(100)

  def testReshapeWithOneUnknownDim(self):
    shape = pretty_tensor_methods._infer_unknown_dims([2, 3, None], '_*')
    self.assertEquals([2, -1], shape)

  def testReshapeWithUnknownBatch(self):
    shape = pretty_tensor_methods._infer_unknown_dims([None, 3, 5], '_*')
    self.assertEquals(['_', 15], shape)

  def testReshapeWithTwoLegalUnknownDim(self):
    shape = pretty_tensor_methods._infer_unknown_dims([2, None, None], '_*')
    self.assertEquals([2, -1], shape)

  def testReshapeWithTensor(self):
    shape = tf.constant([2, 5]) * tf.constant(1, 3)

    reshaped = self.Wrap(self.input).reshape(shape)
    self.assertEquals([None, None], reshaped.shape)

  def testReshapeWithTooManyUnknownDim(self):
    shape = pretty_tensor_methods._infer_unknown_dims([None, 3, None], '_*')
    self.assertEquals(['_', -1], shape)

  def testSoftmaxEval(self):
    np_prediction = numpy.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=numpy.float)
    actual = numpy.array(
        [
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
        ],
        dtype=numpy.float)
    weights = numpy.array([1, 1, 1, 0, 0], dtype=numpy.float)
    prediction = self.Wrap(np_prediction)

    evaluation = prediction.evaluate_classifier(tf.constant(actual))
    result = self.RunTensor(evaluation)
    testing.assert_allclose(numpy.array([0.6]), result, rtol=TOLERANCE)

    # Ensure that an integer type works.
    evaluation = prediction.evaluate_classifier(tf.constant(actual.astype(
        numpy.int32)))
    result = self.RunTensor(evaluation)
    testing.assert_allclose(numpy.array([0.6]), result, rtol=TOLERANCE)

    evaluation = prediction.evaluate_classifier(
        tf.constant(actual), tf.constant(weights))
    result = self.RunTensor(evaluation)
    testing.assert_allclose(numpy.array([2. / 3.]), result, rtol=TOLERANCE)

  def testEvaluatorCreatesUniqueVariables(self):
    actual = numpy.array([[0, 1, 0], [1, 0, 0],], dtype=numpy.float)
    with tf.variable_scope('scope') as vs:
      (self.input_layer.flatten().softmax_classifier(3))
    with tf.variable_scope(vs, reuse=True):
      # We would have an exception if evaluate_classifier used get_variable to
      # create a new variable. The first example doesn't add
      # evaluate_classifier.
      (self.input_layer.flatten()
       .softmax_classifier(3).softmax.evaluate_classifier(
           actual,
           phase=prettytensor.Phase.test))

  def testBinaryCrossEntropy(self):
    n1 = numpy.array([[2., 3., 4., 5., -6., -7.]], dtype=numpy.float32)
    n2 = numpy.array([[1., 1., 0., 0., 0., 1.]], dtype=numpy.float32)
    ftensor1 = prettytensor.wrap(n1)
    ftensor2 = prettytensor.wrap(n2)
    out = self.RunTensor(ftensor1.binary_cross_entropy_with_logits(ftensor2))
    testing.assert_allclose(out,
                            numpy.sum(n1 *
                                      (1 - n2) + numpy.log(1 + numpy.exp(-n1)),
                                      axis=1),
                            rtol=0.00001)

  def testBinaryCrossEntropyWithPerOutputWeights(self):
    n1 = numpy.array([[2., 3., 4.], [5., -6., -7.]], dtype=numpy.float32)
    n2 = numpy.array([[1., 1., 0.], [0., 0., 1.]], dtype=numpy.float32)
    ftensor1 = prettytensor.wrap(n1)
    ftensor2 = prettytensor.wrap(n2)
    weights = numpy.random.rand(*n1.shape).astype(numpy.float32)
    out = self.RunTensor(ftensor1.binary_cross_entropy_with_logits(
        ftensor2,
        per_output_weights=weights))
    expected = (numpy.sum((n1 * (1 - n2) + numpy.log(1 + numpy.exp(-n1))) *
                          weights) / len(n1))
    testing.assert_allclose(out, expected, rtol=0.00001)

  def testSampledSoftmax(self):
    """Tests that sampled softmax runs properly."""
    actual = numpy.array([[1], [0],], dtype=numpy.int64)
    softmax, loss = (self.input_layer.flatten()
                     .softmax_classifier_with_sampled_loss(3,
                                                           actual,
                                                           num_sampled=2))

    self.sess.run(tf.initialize_all_variables())
    out_softmax, out_loss = self.sess.run((softmax, loss))
    self.assertEqual(out_softmax.shape, (2, 3))
    self.assertEqual(out_loss.shape, ())

  def testWeightedSoftmaxEval(self):
    np_prediction = numpy.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=numpy.float)
    actual = numpy.array(
        [
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
        ],
        dtype=numpy.float)
    prediction = self.Wrap(np_prediction)

    evaluation = prediction.evaluate_classifier(
        tf.constant(actual),
        per_example_weights=[0.0, 0.0, 0.0, 1.0, 1.0])
    result = self.RunTensor(evaluation)
    testing.assert_allclose(numpy.array([0.5]), result, rtol=TOLERANCE)
    evaluation = prediction.evaluate_classifier(
        tf.constant(actual),
        per_example_weights=[1.0, 0.0, 0.0, 1.0, 1.0])
    result = self.RunTensor(evaluation)
    testing.assert_allclose(numpy.array([0.6666667]), result, rtol=TOLERANCE)

  def testPrecisionAndRecall(self):
    np_prediction = numpy.array(
        [
            [0, 1, 1],  # tp = 1, fp = 1, fn = 1, tn = 0
            [0, 0, 1],  # tp = 0, fp = 1, fn = 1, tn = 1
            [1, 0, 0],  # tp = 1, fp = 0, fn = 0, tn = 2
            [0, 1, 0],  # tp = 0, fp = 1, fn = 1, tn = 1
            [0, 1, 0],  # tp = 1, fp = 0, fn = 1, tn = 1
        ],
        dtype=numpy.float)
    actual = numpy.array(
        [
            [1, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
        ],
        dtype=numpy.float)
    prediction = self.Wrap(np_prediction)

    evaluation = prediction.evaluate_precision_recall(tf.constant(actual))
    p, r = self.RunTensor(evaluation)
    testing.assert_allclose(numpy.array(.5), p, rtol=TOLERANCE)
    testing.assert_allclose(numpy.array(3. / 7), r, rtol=TOLERANCE)

    evaluation = prediction.evaluate_precision_recall(
        tf.constant(actual),
        per_example_weights=[0.0, 0.0, 0.0, 1.0, 1.0])
    p, r = self.RunTensor(evaluation)
    testing.assert_allclose(numpy.array(.5), p, rtol=TOLERANCE)
    testing.assert_allclose(numpy.array(1. / 3), r, rtol=TOLERANCE)

    evaluation = prediction.evaluate_precision_recall(
        tf.constant(actual),
        per_example_weights=[1.0, 0.0, 0.0, 1.0, 1.0])
    p, r = self.RunTensor(evaluation)
    testing.assert_allclose(numpy.array(.5), p, rtol=TOLERANCE)
    testing.assert_allclose(numpy.array(.4), r, rtol=TOLERANCE)

  def testToDenseOneHot(self):
    data = numpy.array([1, 5, 2, 9])
    result = prettytensor.funcs.to_dense_one_hot(data, 10)
    dense = self.RunTensor(result)
    expected = numpy.zeros((4, 10))
    expected[numpy.arange(4), data] = 1.0

    testing.assert_allclose(expected, dense, rtol=TOLERANCE)

  def testSoftmaxEvalAtTopK(self):
    np_prediction = numpy.array(
        [
            [0, 1, 0],
            [0.25, 0, 0.75],
            [1, 0, 0],
            [0.25, .65, .10],
            [0, 1, 0],
        ],
        dtype=numpy.float)
    actual = numpy.array(
        [
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
        ],
        dtype=numpy.float)
    prediction = self.Wrap(np_prediction)

    # TopK@1
    evaluation = prediction.evaluate_classifier(tf.constant(actual), topk=1)
    result = self.RunTensor(evaluation)
    testing.assert_allclose(numpy.array([0.6]), result, rtol=TOLERANCE)

    # TopK@2
    evaluation = prediction.evaluate_classifier(tf.constant(actual), topk=2)
    result = self.RunTensor(evaluation)
    testing.assert_allclose(numpy.array([0.8]), result, rtol=TOLERANCE)

    # TopK@3
    evaluation = prediction.evaluate_classifier(tf.constant(actual), topk=3)
    result = self.RunTensor(evaluation)
    testing.assert_allclose(numpy.array([1.0]), result, rtol=TOLERANCE)

  def testSoftmaxEvalAtTopKWithTie(self):
    np_prediction = numpy.zeros((5, 3), dtype=numpy.float)
    actual = numpy.array(
        [
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
        ],
        dtype=numpy.float)
    prediction = self.Wrap(np_prediction)
    # Note: none of these are ideal. k=1 goes left-to-right and 1.0 is just
    # plain wrong for k>1

    # TopK@1
    evaluation = prediction.evaluate_classifier(tf.constant(actual), topk=1)
    result = self.RunTensor(evaluation)
    testing.assert_allclose(numpy.array([0.4]), result, rtol=TOLERANCE)

    # TopK@2
    evaluation = prediction.evaluate_classifier(tf.constant(actual), topk=2)
    result = self.RunTensor(evaluation)
    testing.assert_allclose(numpy.array([1.0]), result, rtol=TOLERANCE)

    # TopK@3
    evaluation = prediction.evaluate_classifier(tf.constant(actual), topk=3)
    result = self.RunTensor(evaluation)
    testing.assert_allclose(numpy.array([1.0]), result, rtol=TOLERANCE)

  def testMathOperators(self):
    operators = [operator.add, operator.sub, operator.mul]

    input2 = self.input * 4
    sequence_input = prettytensor.wrap_sequence([self.input, input2])

    # Test reverse ops
    for op in operators:
      print(op.__name__)
      t1 = op(2., self.input)
      t2 = op(2., self.input_layer)
      seq1 = op([2., 1.], sequence_input)
      seq2 = op(2., sequence_input)

      # Used to validate the sequence.
      t3 = op(1., input2)
      t4 = op(2., input2)

      r1 = self.RunTensor(t1)
      r2 = self.RunTensor(t2)
      r3 = self.RunTensor(t3)
      r4 = self.RunTensor(t4)
      seq_r1 = self.RunTensor(seq1)
      seq_r2 = self.RunTensor(seq2)

      self.assertTrue(isinstance(t2, pretty_tensor_class.Layer))
      testing.assert_allclose(r1, r2, rtol=TOLERANCE)

      testing.assert_allclose(seq_r1[0], r2, rtol=TOLERANCE)
      testing.assert_allclose(seq_r1[1], r3, rtol=TOLERANCE)
      testing.assert_allclose(seq_r2[0], r2, rtol=TOLERANCE)
      testing.assert_allclose(seq_r2[1], r4, rtol=TOLERANCE)

    # Test forward ops
    for op in operators:
      t1 = op(self.input, 2.)
      t2 = op(self.input_layer, 2.)
      seq1 = op(sequence_input, [2., 1.])
      seq2 = op(sequence_input, 2.)

      # Used to validate the sequence.
      t3 = op(input2, 1.)
      t4 = op(input2, 2.)

      r1 = self.RunTensor(t1)
      r2 = self.RunTensor(t2)
      r3 = self.RunTensor(t3)
      r4 = self.RunTensor(t4)
      seq_r1 = self.RunTensor(seq1)
      seq_r2 = self.RunTensor(seq2)

      self.assertTrue(isinstance(t2, pretty_tensor_class.Layer))
      testing.assert_allclose(r1,
                              r2,
                              rtol=TOLERANCE,
                              err_msg='Op: %s' % op.__name__)

      testing.assert_allclose(seq_r1[0], r2, rtol=TOLERANCE)
      testing.assert_allclose(seq_r1[1], r3, rtol=TOLERANCE)
      testing.assert_allclose(seq_r2[0], r2, rtol=TOLERANCE)
      testing.assert_allclose(seq_r2[1], r4, rtol=TOLERANCE)

    operators.extend([operator.truediv])
    for op in operators:
      t1 = op(self.input, self.input_layer)
      t2 = op(self.input_layer, self.input)
      r1 = self.RunTensor(t1)
      r2 = self.RunTensor(t2)

      self.assertFalse(isinstance(t1, pretty_tensor_class.Layer))
      self.assertTrue(isinstance(t2, pretty_tensor_class.Layer))
      testing.assert_allclose(r1,
                              r2,
                              rtol=TOLERANCE,
                              err_msg='Op: %s' % op.__name__)

    unary = [operator.neg, operator.abs]
    for op in unary:
      t1 = op(self.input)
      t2 = op(self.input_layer)
      r1 = self.RunTensor(t1)
      r2 = self.RunTensor(t2)

      seq = op(sequence_input)
      seq_r = self.RunTensor(seq)
      t3 = op(input2)
      r3 = self.RunTensor(t3)

      self.assertTrue(isinstance(t2, pretty_tensor_class.Layer))
      testing.assert_allclose(r1,
                              r2,
                              rtol=TOLERANCE,
                              err_msg='Op: %s' % op.__name__)
      testing.assert_allclose(r1,
                              seq_r[0],
                              rtol=TOLERANCE,
                              err_msg='Op: %s' % op.__name__)
      testing.assert_allclose(r3,
                              seq_r[1],
                              rtol=TOLERANCE,
                              err_msg='Op: %s' % op.__name__)

  def testMathOperatorSideEffects(self):
    seq = self.input_layer.sequential()

    t1 = seq + 1

    self.assertTrue(t1.tensor is not seq.tensor,
                    '%s %s' % (t1.tensor, seq.tensor))

    seq += 1
    r1 = self.RunTensor(t1)
    r2 = self.RunTensor(seq)
    testing.assert_allclose(r1, r2, rtol=TOLERANCE)


if __name__ == '__main__':
  unittest.main()
