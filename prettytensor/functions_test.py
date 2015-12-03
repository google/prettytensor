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
"""Test class for functions."""
import unittest



import numpy
from numpy import testing
import numpy.linalg
from scipy.spatial import distance
import tensorflow as tf

from prettytensor import functions

TOLERANCE = 0.000001


class TensorFlowOpTest(unittest.TestCase):

  def setUp(self):
    unittest.TestCase.setUp(self)
    self.sess = tf.Session('')

  def tearDown(self):
    unittest.TestCase.tearDown(self)
    self.sess.close()

  def Run(self, tensors):
    if isinstance(tensors, tf.Tensor):
      tensors = [tensors]
    return self.sess.run(tensors)

  def testEveryOther(self):
    tensor = tf.constant([[1, 2], [3, 4]])
    out = self.Run(functions.every_other(tensor))
    testing.assert_array_equal(out[0], numpy.array([1, 3], dtype=numpy.int32))
    tensor = tf.constant([[1, 2, 3, 4]])
    out = self.Run(functions.every_other(tensor))
    testing.assert_array_equal(out[0], numpy.array([1, 3], dtype=numpy.int32))

  def testL1RegressionLoss(self):
    ftensor1 = tf.constant([1., 2., 3., 4.])
    ftensor2 = tf.constant([5., 6., 7., -8.])
    out = self.Run(functions.l1_regression_loss(ftensor1, ftensor2))
    testing.assert_array_equal(out[0], numpy.array([4., 4., 4., 12.]))

  def testL2SqRegressionLoss(self):
    ftensor1 = tf.constant([1., 2., 3., 4.])
    ftensor2 = tf.constant([5., 6., 7., -8.])
    out = self.Run(functions.l2_regression_sq_loss(ftensor1, ftensor2))
    testing.assert_array_equal(out[0], numpy.array([16., 16., 16, 144]))

  def testL2RegressionLoss(self):
    ftensor1 = tf.constant([1., 2., 3., 4.])
    ftensor2 = tf.constant([5., 6., 7., -8.])
    out = self.Run(functions.l2_regression_loss(ftensor1, ftensor2))
    testing.assert_allclose(
        out[0],
        numpy.array([4., 4., 4., 12.]),
        rtol=TOLERANCE)

  def testBinaryCorssEntropyLossWithLogits(self):
    n1 = numpy.array([2., 3., 4., 5., -6., -7.], dtype=numpy.float32)
    n2 = numpy.array([1., 1., 0., 0., 0., 1.], dtype=numpy.float32)
    ftensor1 = tf.constant(n1)
    ftensor2 = tf.constant(n2)
    out = self.Run(functions.binary_cross_entropy_loss_with_logits(ftensor1,
                                                                   ftensor2))
    testing.assert_allclose(
        out[0],
        n1 * (1-n2) + numpy.log(1 + numpy.exp(-n1)),
        rtol=TOLERANCE)

  def testSoftPlus(self):
    # 100 overflows naive implementations in float
    values = (
        numpy.array(
            [-100., -10., 1., 0, 1., 10., 100.],
            dtype=numpy.float32))
    out = self.Run(
        functions.softplus(
            tf.constant(
                values,
                dtype=tf.float32),
            1.))
    np_values = numpy.log(1. + numpy.exp(values))
    np_values[6] = 100.
    testing.assert_allclose(out[0], np_values, rtol=TOLERANCE)

    out = self.Run(functions.softplus(tf.constant(values), 2.))
    np_values = numpy.log(1. + numpy.exp(values * 2.)) / 2.
    np_values[6] = 100.
    testing.assert_allclose(out[0], np_values, rtol=TOLERANCE)

  def testCosDistance(self):
    n1 = numpy.array([[1., 2., 3., 4.], [1., 1., 1., 1.]], dtype=numpy.float32)
    n2 = numpy.array([[5., 6., 7., -8.], [1., 1., 1., 1.]], dtype=numpy.float32)
    out = self.Run(functions.cos_distance(n1, n2))
    testing.assert_allclose(
        out[0],
        numpy.array(
            [distance.cosine(n1[0], n2[0]), distance.cosine(n1[1], n2[1])]),
        rtol=TOLERANCE)

  def testL1Distance(self):
    n1 = numpy.array([[1., 2., 3., 4.], [1., 1., 1., 1.]], dtype=numpy.float32)
    n2 = numpy.array([[5., 6., 7., -8.], [1., 1., 1., 1.]], dtype=numpy.float32)
    out = self.Run(functions.l1_distance(n1, n2))
    testing.assert_allclose(
        out[0],
        numpy.array(
            [distance.cityblock(n1[0], n2[0]), distance.cityblock(n1[1], n2[1])
            ]),
        rtol=TOLERANCE)

  def testL2Distance(self):
    n1 = numpy.array([[1., 2., 3., 4.], [1., 1., 1., 1.]], dtype=numpy.float32)
    n2 = numpy.array([[5., 6., 7., -8.], [1., 1., 1., 1.]], dtype=numpy.float32)
    out = self.Run(functions.l2_distance(n1, n2))
    testing.assert_allclose(
        out[0],
        numpy.array(
            [distance.euclidean(n1[0], n2[0]), distance.euclidean(n1[1], n2[1])
            ]),
        rtol=TOLERANCE)

  def testL2DistanceSq(self):
    n1 = numpy.array([[1., 2., 3., 4.], [1., 1., 1., 1.]], dtype=numpy.float32)
    n2 = numpy.array([[5., 6., 7., -8.], [1., 1., 1., 1.]], dtype=numpy.float32)
    out = self.Run(functions.l2_distance_sq(n1, n2))
    testing.assert_allclose(
        out[0],
        numpy.power(
            numpy.array(
                [distance.euclidean(n1[0], n2[0]), distance.euclidean(
                    n1[1], n2[1])]), 2),
        rtol=TOLERANCE)

  def testDotDistance(self):
    n1 = numpy.array([[1., 2., 3., 4.], [1., 1., 1., 1.]], dtype=numpy.float32)
    n2 = numpy.array([[5., 6., 7., -8.], [1., 1., 1., 1.]], dtype=numpy.float32)
    out = self.Run(functions.dot_distance(n1, n2))
    testing.assert_allclose(
        out[0],
        numpy.array(-numpy.sum(n1 * n2,
                               axis=1)),
        rtol=TOLERANCE)

  def testCosDistanceWithBroadcast(self):
    n1 = numpy.array([[[1., 2., 3., 4.], [1., 1., 1., 1.]], [[5., 6., 7., 8.],
                                                             [1., 1., 1., 2.]]],
                     dtype=numpy.float32)
    n2 = numpy.array([[5., 6., 7., -8.], [1., 1., 1., 1.]], dtype=numpy.float32)
    out = self.Run(functions.cos_distance(n1, n2))
    expected = numpy.array(
        [[distance.cosine(n1[0, 0], n2[0]), distance.cosine(n1[0, 1], n2[1])],
         [distance.cosine(n1[1, 0], n2[0]), distance.cosine(n1[1, 1], n2[1])]])
    testing.assert_allclose(expected, out[0], atol=TOLERANCE)

  def testL1DistanceWithBroadcast(self):
    n1 = numpy.array([[[1., 2., 3., 4.], [1., 1., 1., 1.]], [[5., 6., 7., 8.],
                                                             [1., 1., 1., 2.]]],
                     dtype=numpy.float32)
    n2 = numpy.array([[5., 6., 7., -8.], [1., 1., 1., 1.]], dtype=numpy.float32)
    out = self.Run(functions.l1_distance(n1, n2))
    expected = numpy.array(
        [[distance.cityblock(n1[0, 0], n2[0]), distance.cityblock(
            n1[0, 1], n2[1])], [distance.cityblock(n1[1, 0], n2[0]),
                                distance.cityblock(n1[1, 1], n2[1])]])
    testing.assert_allclose(expected, out[0], atol=TOLERANCE)

  def testL2DistanceWithBroadcast(self):
    n1 = numpy.array([[[1., 2., 3., 4.], [1., 1., 1., 1.]], [[5., 6., 7., 8.],
                                                             [1., 1., 1., 2.]]],
                     dtype=numpy.float32)
    n2 = numpy.array([[5., 6., 7., -8.], [1., 1., 1., 1.]], dtype=numpy.float32)
    out = self.Run(functions.l2_distance(n1, n2))
    expected = numpy.array(
        [[distance.euclidean(n1[0, 0], n2[0]), distance.euclidean(
            n1[0, 1], n2[1])], [distance.euclidean(n1[1, 0], n2[0]),
                                distance.euclidean(n1[1, 1], n2[1])]])
    testing.assert_allclose(expected, out[0], atol=TOLERANCE)

  def testL2DistanceSqWithBroadcast(self):
    n1 = numpy.array([[[1., 2., 3., 4.], [1., 1., 1., 1.]], [[5., 6., 7., 8.],
                                                             [1., 1., 1., 2.]]],
                     dtype=numpy.float32)
    n2 = numpy.array([[5., 6., 7., -8.], [1., 1., 1., 1.]], dtype=numpy.float32)
    out = self.Run(functions.l2_distance_sq(n1, n2))
    expected = numpy.array(
        [[distance.euclidean(n1[0, 0], n2[0]), distance.euclidean(
            n1[0, 1], n2[1])], [distance.euclidean(n1[1, 0], n2[0]),
                                distance.euclidean(n1[1, 1], n2[1])]])
    expected = numpy.power(expected, 2)
    testing.assert_allclose(expected, out[0], atol=TOLERANCE)

  def testDotDistanceWithBroadcast(self):
    n1 = numpy.array([[[1., 2., 3., 4.], [1., 1., 1., 1.]], [[5., 6., 7., 8.],
                                                             [1., 1., 1., 2.]]],
                     dtype=numpy.float32)
    n2 = numpy.array([[5., 6., 7., -8.], [1., 1., 1., 1.]], dtype=numpy.float32)
    out = self.Run(functions.dot_distance(n1, n2))
    testing.assert_allclose(
        out[0],
        numpy.array(-numpy.sum(n1 * n2,
                               axis=2)),
        rtol=TOLERANCE)

  def testL2Normalize(self):
    n1 = numpy.array([[1., 2., 3., 4.], [1., 1., 1., 1.]], dtype=numpy.float32)
    t1 = tf.constant(n1)
    out = self.Run(functions.l2_normalize(t1, 1))
    testing.assert_allclose(
        out[0],
        n1 / numpy.linalg.norm(n1,
                               2,
                               axis=1).reshape((2, 1)),
        rtol=TOLERANCE)

  def testL1Normalize(self):
    n1 = numpy.array([[1., 2., 3., 4.], [1., 1., 1., 1.]], dtype=numpy.float32)
    t1 = tf.constant(n1)
    out = self.Run(functions.l1_normalize(t1, 1))
    testing.assert_allclose(
        out[0],
        n1 / numpy.linalg.norm(n1,
                               1,
                               axis=1).reshape((2, 1)),
        rtol=TOLERANCE)

  def testLeakyRelu(self):
    values = (
        numpy.array(
            [-100., -10., 1., 0, 1., 10., 100.],
            dtype=numpy.float32))
    tensor = tf.constant(values)
    out = self.Run(functions.leaky_relu(tensor))
    for i in range(len(values)):
      values[i] *= 0.01 if values[i] < 0 else 1
    testing.assert_allclose(out[0], values, rtol=TOLERANCE)

  def testUnzip(self):
    n1 = numpy.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]],
                     dtype=numpy.float32)
    t1 = tf.constant(n1)
    out = self.Run(functions.unzip(t1, 0, 4, 2))

    expected = numpy.array([[1., 2.], [5., 6.]], dtype=numpy.float32)
    testing.assert_allclose(expected, out[0], rtol=TOLERANCE)
    expected = numpy.array([[3., 4.], [7., 8.]], dtype=numpy.float32)
    testing.assert_allclose(expected, out[1], rtol=TOLERANCE)

  def testSplit(self):
    """Testing TF functionality to highlight difference with Unzip."""
    n1 = numpy.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]],
                     dtype=numpy.float32)
    t1 = tf.constant(n1)
    out = self.Run(tf.split(0, 2, t1))
    expected = numpy.array([[1., 2.], [3., 4.]], dtype=numpy.float32)
    testing.assert_allclose(expected, out[0], rtol=TOLERANCE)
    expected = numpy.array([[5., 6.], [7., 8.]], dtype=numpy.float32)
    testing.assert_allclose(expected, out[1], rtol=TOLERANCE)


if __name__ == '__main__':
  unittest.main()
