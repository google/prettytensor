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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import numpy
from numpy import linalg
from numpy import testing
import tensorflow as tf

from prettytensor import functions

TOLERANCE = 0.00001


# Distance functions used in tests.  These are defined here instead of using
# scipy so the open source tests don't depend on such a huge module for 3
# 1 line functions.
def cosine(u, v):  # pylint: disable=invalid-name
  return 1.0 - numpy.dot(u, v) / (linalg.norm(u, ord=2) * linalg.norm(v, ord=2))


def cityblock(u, v):  # pylint: disable=invalid-name
  return numpy.abs(u - v).sum()


def euclidean(u, v):  # pylint: disable=invalid-name
  return linalg.norm(u - v, ord=2)


class TensorFlowOpTest(tf.test.TestCase):

  def eval_tensor(self, tensors):
    if isinstance(tensors, tf.Tensor):
      tensors = [tensors]
    with self.test_session() as sess:
      return sess.run(tensors)

  def test_every_other(self):
    tensor = tf.constant([[1, 2], [3, 4]])
    out = self.eval_tensor(functions.every_other(tensor))
    testing.assert_array_equal(out[0], numpy.array([1, 3], dtype=numpy.int32))
    tensor = tf.constant([[1, 2, 3, 4]])
    out = self.eval_tensor(functions.every_other(tensor))
    testing.assert_array_equal(out[0], numpy.array([1, 3], dtype=numpy.int32))

  def test_l1_regression_loss(self):
    ftensor1 = tf.constant([1., 2., 3., 4.])
    ftensor2 = tf.constant([5., 6., 7., -8.])
    out = self.eval_tensor(functions.l1_regression_loss(ftensor1, ftensor2))
    testing.assert_array_equal(out[0], numpy.array([4., 4., 4., 12.]))

  def test_l2_sq_regression_loss(self):
    ftensor1 = tf.constant([1., 2., 3., 4.])
    ftensor2 = tf.constant([5., 6., 7., -8.])
    out = self.eval_tensor(functions.l2_regression_sq_loss(ftensor1, ftensor2))
    testing.assert_array_equal(out[0], numpy.array([16., 16., 16, 144]))

  def test_l2_regression_loss(self):
    ftensor1 = tf.constant([1., 2., 3., 4.])
    ftensor2 = tf.constant([5., 6., 7., -8.])
    out = self.eval_tensor(functions.l2_regression_loss(ftensor1, ftensor2))
    testing.assert_allclose(
        out[0],
        numpy.array([4., 4., 4., 12.]),
        rtol=TOLERANCE)

  def test_binary_cross_entropy_loss_with_logits(self):
    n1 = numpy.array([2., 3., 4., 5., -6., -7.], dtype=numpy.float32)
    n2 = numpy.array([1., 1., 0., 0., 0., 1.], dtype=numpy.float32)
    ftensor1 = tf.constant(n1)
    ftensor2 = tf.constant(n2)
    out = self.eval_tensor(functions.binary_cross_entropy_loss_with_logits(
        ftensor1, ftensor2))
    testing.assert_allclose(
        out[0],
        n1 * (1-n2) + numpy.log(1 + numpy.exp(-n1)),
        rtol=0.00001)

  def test_soft_plus(self):
    # 100 overflows naive implementations in float
    values = (
        numpy.array(
            [-100., -10., 1., 0, 1., 10., 100.],
            dtype=numpy.float32))
    out = self.eval_tensor(
        functions.softplus(
            tf.constant(
                values,
                dtype=tf.float32),
            1.))
    np_values = numpy.log(1. + numpy.exp(values))
    np_values[6] = 100.
    testing.assert_allclose(out[0], np_values, rtol=TOLERANCE)

    out = self.eval_tensor(functions.softplus(tf.constant(values), 2.))
    np_values = numpy.log(1. + numpy.exp(values * 2.)) / 2.
    np_values[6] = 100.
    testing.assert_allclose(out[0], np_values, rtol=TOLERANCE)

  def test_cos_distance(self):
    n1 = numpy.array([[1., 2., 3., 4.], [1., 1., 1., 1.]], dtype=numpy.float32)
    n2 = numpy.array([[5., 6., 7., -8.], [1., 1., 1., 1.]], dtype=numpy.float32)
    out = self.eval_tensor(functions.cos_distance(n1, n2))

    testing.assert_allclose(
        out[0],
        numpy.array([cosine(n1[0], n2[0]), cosine(n1[1], n2[1])]),
        rtol=TOLERANCE)

  def test_l1_distance(self):
    n1 = numpy.array([[1., 2., 3., 4.], [1., 1., 1., 1.]], dtype=numpy.float32)
    n2 = numpy.array([[5., 6., 7., -8.], [1., 1., 1., 1.]], dtype=numpy.float32)
    out = self.eval_tensor(functions.l1_distance(n1, n2))
    testing.assert_allclose(
        out[0],
        numpy.array(
            [cityblock(n1[0], n2[0]), cityblock(n1[1], n2[1])
            ]),
        rtol=TOLERANCE)

  def test_l2_distance(self):
    n1 = numpy.array([[1., 2., 3., 4.], [1., 1., 1., 1.]], dtype=numpy.float32)
    n2 = numpy.array([[5., 6., 7., -8.], [1., 1., 1., 1.]], dtype=numpy.float32)
    out = self.eval_tensor(functions.l2_distance(n1, n2))
    testing.assert_allclose(
        out[0],
        numpy.array(
            [euclidean(n1[0], n2[0]),
             1e-6  # Epsilon sets the minimum distance so use that instead of 0.
            ]),
        rtol=TOLERANCE)

  def test_l2_distance_sq(self):
    n1 = numpy.array([[1., 2., 3., 4.], [1., 1., 1., 1.]], dtype=numpy.float32)
    n2 = numpy.array([[5., 6., 7., -8.], [1., 1., 1., 1.]], dtype=numpy.float32)
    out = self.eval_tensor(functions.l2_distance_sq(n1, n2))
    testing.assert_allclose(
        out[0],
        numpy.power(
            numpy.array(
                [euclidean(n1[0], n2[0]), euclidean(
                    n1[1], n2[1])]), 2),
        rtol=TOLERANCE)

  def test_dot_distance(self):
    n1 = numpy.array([[1., 2., 3., 4.], [1., 1., 1., 1.]], dtype=numpy.float32)
    n2 = numpy.array([[5., 6., 7., -8.], [1., 1., 1., 1.]], dtype=numpy.float32)
    out = self.eval_tensor(functions.dot_distance(n1, n2))
    testing.assert_allclose(
        out[0],
        numpy.array(-numpy.sum(n1 * n2,
                               axis=1)),
        rtol=TOLERANCE)

  def test_cos_distance_with_broadcast(self):
    n1 = numpy.array([[[1., 2., 3., 4.], [1., 1., 1., 1.]], [[5., 6., 7., 8.],
                                                             [1., 1., 1., 2.]]],
                     dtype=numpy.float32)
    n2 = numpy.array([[5., 6., 7., -8.], [1., 1., 1., 1.]], dtype=numpy.float32)
    out = self.eval_tensor(functions.cos_distance(n1, n2))
    expected = numpy.array(
        [[cosine(n1[0, 0], n2[0]), cosine(n1[0, 1], n2[1])],
         [cosine(n1[1, 0], n2[0]), cosine(n1[1, 1], n2[1])]])
    testing.assert_allclose(expected, out[0], atol=TOLERANCE)

  def test_l1_distance_with_broadcast(self):
    n1 = numpy.array([[[1., 2., 3., 4.], [1., 1., 1., 1.]], [[5., 6., 7., 8.],
                                                             [1., 1., 1., 2.]]],
                     dtype=numpy.float32)
    n2 = numpy.array([[5., 6., 7., -8.], [1., 1., 1., 1.]], dtype=numpy.float32)
    out = self.eval_tensor(functions.l1_distance(n1, n2))
    expected = numpy.array(
        [[cityblock(n1[0, 0], n2[0]), cityblock(
            n1[0, 1], n2[1])], [cityblock(n1[1, 0], n2[0]),
                                cityblock(n1[1, 1], n2[1])]])
    testing.assert_allclose(expected, out[0], atol=TOLERANCE)

  def test_l2_distance_with_broadcast(self):
    n1 = numpy.array([[[1., 2., 3., 4.], [1., 1., 1., 1.]], [[5., 6., 7., 8.],
                                                             [1., 1., 1., 2.]]],
                     dtype=numpy.float32)
    n2 = numpy.array([[5., 6., 7., -8.], [1., 1., 1., 1.]], dtype=numpy.float32)
    out = self.eval_tensor(functions.l2_distance(n1, n2))
    expected = numpy.array(
        [[euclidean(n1[0, 0], n2[0]), euclidean(
            n1[0, 1], n2[1])], [euclidean(n1[1, 0], n2[0]),
                                euclidean(n1[1, 1], n2[1])]])
    testing.assert_allclose(expected, out[0], atol=TOLERANCE)

  def test_l2_distance_sq_with_broadcast(self):
    n1 = numpy.array([[[1., 2., 3., 4.], [1., 1., 1., 1.]], [[5., 6., 7., 8.],
                                                             [1., 1., 1., 2.]]],
                     dtype=numpy.float32)
    n2 = numpy.array([[5., 6., 7., -8.], [1., 1., 1., 1.]], dtype=numpy.float32)
    out = self.eval_tensor(functions.l2_distance_sq(n1, n2))
    expected = numpy.array(
        [[euclidean(n1[0, 0], n2[0]), euclidean(
            n1[0, 1], n2[1])], [euclidean(n1[1, 0], n2[0]),
                                euclidean(n1[1, 1], n2[1])]])
    expected = numpy.power(expected, 2)
    testing.assert_allclose(expected, out[0], atol=TOLERANCE)

  def test_dot_distance_with_broadcast(self):
    n1 = numpy.array([[[1., 2., 3., 4.], [1., 1., 1., 1.]], [[5., 6., 7., 8.],
                                                             [1., 1., 1., 2.]]],
                     dtype=numpy.float32)
    n2 = numpy.array([[5., 6., 7., -8.], [1., 1., 1., 1.]], dtype=numpy.float32)
    out = self.eval_tensor(functions.dot_distance(n1, n2))
    testing.assert_allclose(
        out[0],
        numpy.array(-numpy.sum(n1 * n2,
                               axis=2)),
        rtol=TOLERANCE)

  def test_l2_normalize(self):
    n1 = numpy.array([[1., 2., 3., 4.], [1., 1., 1., 1.]], dtype=numpy.float32)
    t1 = tf.constant(n1)
    out = self.eval_tensor(functions.l2_normalize(t1, 1))
    testing.assert_allclose(
        out[0],
        n1 / linalg.norm(n1, 2, axis=1).reshape((2, 1)),
        rtol=TOLERANCE)

  def test_l1_normalize(self):
    n1 = numpy.array([[1., 2., 3., 4.], [1., 1., 1., 1.]], dtype=numpy.float32)
    t1 = tf.constant(n1)
    out = self.eval_tensor(functions.l1_normalize(t1, 1))
    testing.assert_allclose(
        out[0],
        n1 / linalg.norm(n1, 1, axis=1).reshape((2, 1)),
        rtol=TOLERANCE)

  def test_leaky_relu(self):
    values = (
        numpy.array(
            [-100., -10., 1., 0, 1., 10., 100.],
            dtype=numpy.float32))
    tensor = tf.constant(values)
    out = self.eval_tensor(functions.leaky_relu(tensor))
    for i, value in enumerate(values):
      if value < 0:
        values[i] *= 0.01
    testing.assert_allclose(out[0], values, rtol=TOLERANCE)

  def test_unzip(self):
    n1 = numpy.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]],
                     dtype=numpy.float32)
    t1 = tf.constant(n1)
    out = self.eval_tensor(functions.unzip(t1, 0, 4, 2))

    expected = numpy.array([[1., 2.], [5., 6.]], dtype=numpy.float32)
    testing.assert_allclose(expected, out[0], rtol=TOLERANCE)
    expected = numpy.array([[3., 4.], [7., 8.]], dtype=numpy.float32)
    testing.assert_allclose(expected, out[1], rtol=TOLERANCE)

  def test_split(self):
    """Testing TF functionality to highlight difference with Unzip."""
    n1 = numpy.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]],
                     dtype=numpy.float32)
    t1 = tf.constant(n1)
    out = self.eval_tensor(tf.split(0, 2, t1))
    expected = numpy.array([[1., 2.], [3., 4.]], dtype=numpy.float32)
    testing.assert_allclose(expected, out[0], rtol=TOLERANCE)
    expected = numpy.array([[5., 6.], [7., 8.]], dtype=numpy.float32)
    testing.assert_allclose(expected, out[1], rtol=TOLERANCE)


if __name__ == '__main__':
  tf.test.main()
