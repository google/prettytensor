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
"""Tests for templating in PrettyTensor."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest



import numpy
from numpy import testing
import tensorflow as tf

import prettytensor
from prettytensor import pretty_tensor_class
from prettytensor import pretty_tensor_testing

KEY = 'random_key'
TOLERANCE = 0.000001


@prettytensor.Register(assign_defaults='value')
def ValidateMethod(input_tensor, test_class, value):
  test_class.assertEqual(KEY, value)
  return input_tensor


class TemplatedPrettyTensorTest(pretty_tensor_testing.PtTestCase):

  def setUp(self):
    super(self.__class__, self).setUp()
    # Input is 2x3x5, which isn't a natural size for any op.
    self.input_data = numpy.array(
        [[[1, 2, 3, 4, 5],
          [6, 7, 8, 9, 10],
          [10, 12, 13, 14, 15],], [[-1, 2, -3, 4, -5], [6, -7, 8, -9, 10],
                                   [-10, 12, -13, 14, -15]]],
        dtype=numpy.float)
    self.input = tf.constant(self.input_data, dtype=tf.float32)

  def Template(self, key):
    return prettytensor.template(key, self.bookkeeper)

  def testSimpleTemplate(self):
    template = self.Template(KEY)

    x = template.construct(random_key=self.input)
    out = self.RunTensor(x)
    testing.assert_allclose(self.input_data, out, rtol=TOLERANCE)

  def testSingleMethod(self):
    template = self.Template(KEY).flatten()

    x = template.construct(random_key=self.input)
    out = self.RunTensor(x)
    testing.assert_allclose(
        self.input_data.reshape([2, 15]),
        out,
        rtol=TOLERANCE)

  def testSequential(self):
    seq = self.Template(KEY).sequential()
    seq.flatten()
    seq.fully_connected(100)
    out = self.RunTensor(seq.as_layer().construct(random_key=self.input))
    self.assertSequenceEqual([2, 100], out.shape)

  def testAttach(self):
    input_pt = self.Wrap(self.input)
    template = self.Template('input').flatten().fully_connected(100)
    out = self.RunTensor(input_pt.attach_template(template, 'input'))
    self.assertSequenceEqual([2, 100], out.shape)

  def testUnboundVariableForParameter(self):
    input_pt = self.Wrap(self.input)
    template = input_pt.flatten().fully_connected(prettytensor.UnboundVariable(
        'width'))
    self.assertTrue(isinstance(template, pretty_tensor_class._DeferredLayer))
    out = self.RunTensor(template.construct(width=200))
    self.assertSequenceEqual([2, 200], out.shape)

  def testMissingUnboundVariable(self):
    input_pt = self.Wrap(self.input)
    template = input_pt.flatten().fully_connected(prettytensor.UnboundVariable(
        'width'))
    with self.assertRaises(ValueError):
      template.construct()

  def testUnboundVariableReused(self):
    """The same unbound_var can be used multiple times in a graph."""
    input_pt = self.Wrap(self.input)
    unbound_var = prettytensor.UnboundVariable('width')
    template = (input_pt.flatten().fully_connected(unbound_var)
                .fully_connected(unbound_var))
    out = self.RunTensor(template.construct(width=200))
    self.assertSequenceEqual([2, 200], out.shape)

  def testAttachToTemplate(self):
    input_pt = self.Wrap(self.input)
    template1 = self.Template('input').flatten()
    template2 = self.Template('input').fully_connected(100)

    joined = template1.attach_template(template2, 'input')
    out = self.RunTensor(input_pt.attach_template(joined, 'input'))
    self.assertSequenceEqual([2, 100], out.shape)

  def testUnboundVariableAsDefault(self):
    """The same unbound_var can be used multiple times in a graph."""
    input_pt = self.Wrap(self.input)
    with prettytensor.defaults_scope(
        value=prettytensor.UnboundVariable('key')):
      x = input_pt.ValidateMethod(self)
    self.assertTrue(isinstance(x, pretty_tensor_class._DeferredLayer))
    x.construct(key=KEY)

  def testConflictingUnboundVariables(self):
    """Two unbound_vars with the same name are considered conflicting."""
    input_pt = self.Wrap(self.input)
    with self.assertRaises(ValueError):
      (input_pt.flatten()
       .fully_connected(prettytensor.UnboundVariable('width'))
       .fully_connected(prettytensor.UnboundVariable('width')))

  def testMultipleUnboundVariables(self):
    input_pt = self.Wrap(self.input)
    template = (input_pt.flatten()
                .fully_connected(prettytensor.UnboundVariable('width'))
                .fully_connected(prettytensor.UnboundVariable('width2')))
    out = self.RunTensor(template.construct(width=200, width2=100))
    self.assertSequenceEqual([2, 100], out.shape)

  def testExtraValues(self):
    input_pt = self.Wrap(self.input)
    template = (input_pt.flatten()
                .fully_connected(prettytensor.UnboundVariable('width')))
    with self.assertRaises(ValueError):
      template.construct(width=200, width2=100)

  def testIncompatibleUnboundVariableValues(self):
    """Ensures that an error is thrown if a var is given incompatible values.

    Since the primary use case of templates is parameter sharing, it is
    important that substitutions don't conflict.
    """
    input_pt = self.Wrap(self.input)
    full = input_pt.flatten().fully_connected(prettytensor.UnboundVariable(
        'width'))
    full.construct(width=100)
    with self.assertRaises(ValueError):
      full.construct(width=200)

  def BuildLargishGraph(self, input_pt):
    seq = input_pt.sequential()
    seq.reshape('___1')
    seq.conv2d(1, 10)
    with seq.subdivide(2) as [a, b]:
      a.with_name('a').conv2d(1, 5)
      b.with_name('b').conv2d(1, 15)
    seq.with_name('wow')
    seq.flatten()
    seq.fully_connected(100, name='a_funny_name')
    return seq.as_layer()

  def testGraphMatchesImmediate(self):
    """Ensures that the vars line up between the two modes."""
    with tf.Graph().as_default():
      input_pt = prettytensor.wrap(
          tf.constant(self.input_data, dtype=tf.float32))
      self.BuildLargishGraph(input_pt)
      normal_names = sorted([v.name for v in tf.all_variables()])

    with tf.Graph().as_default():
      template = prettytensor.template('input')
      self.BuildLargishGraph(template).construct(input=prettytensor.wrap(
          tf.constant(self.input_data, dtype=tf.float32)))
      template_names = sorted([v.name for v in tf.all_variables()])

    self.assertSequenceEqual(normal_names, template_names)

  def testVariablesAreShared(self):
    """Ensures that adding the graph twice shares variables."""
    input_pt = self.Wrap(self.input)
    template = self.Template('input').flatten().fully_connected(10)

    l1 = template.construct(input=input_pt)
    l2 = template.construct(input=input_pt)
    self.assertNotEqual(l1.tensor, l2.tensor)

    v1 = self.RunTensor(l1, init=True)
    v2 = self.RunTensor(l2, init=False)
    testing.assert_allclose(v1, v2, rtol=TOLERANCE)

  def testBind(self):
    input_pt = self.Wrap(self.input)
    template = self.Template('input').flatten().fully_connected(10)

    l1 = template.bind(input=input_pt).construct()
    l2 = template.construct(input=input_pt)
    v1 = self.RunTensor(l1, init=True)
    v2 = self.RunTensor(l2, init=False)
    testing.assert_allclose(v1, v2, rtol=TOLERANCE)

  def testBindTuple(self):
    labels = numpy.array([[0., 1.], [1., 0.]], dtype=numpy.float32)
    template = self.Template('input').flatten().softmax_classifier(2, labels)
    bound = template.bind(input=self.input)

    tuple1 = bound.construct()
    tuple2 = template.construct(input=self.input)

    self.assertNotEqual(tuple1.softmax.tensor, tuple2.softmax.tensor)
    softmax1 = self.RunTensor(tuple1.softmax, init=True)
    loss1 = self.RunTensor(tuple1.loss, init=False)
    softmax2 = self.RunTensor(tuple2.softmax, init=False)
    loss2 = self.RunTensor(tuple2.loss, init=False)
    testing.assert_allclose(softmax1, softmax2, rtol=TOLERANCE)
    testing.assert_allclose(loss1, loss2, rtol=TOLERANCE)

  def testConstructAllWithConflictingValues(self):
    labels = numpy.array([[0., 1.], [1., 0.]], dtype=numpy.float32)
    template = self.Template('input').flatten().softmax_classifier(2, labels)

    softmax = template.softmax.bind(input=self.input)
    loss = template.loss.bind(input=labels)
    with self.assertRaises(ValueError):
      prettytensor.construct_all([softmax, loss])


if __name__ == '__main__':
  unittest.main()
