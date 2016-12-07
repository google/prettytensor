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

import unittest
import six
import tensorflow as tf

import prettytensor


# Count of tests for unique root namespaces.
_count = 0


class PtTestCase(unittest.TestCase):
  """Contains shared setUp/tearDown and convenience methods.

  This adds the following attributes to self:

  self.bookkeeper
  self.sess
  """

  def __init__(self, *args):
    super(PtTestCase, self).__init__(*args)
    self.bookkeeper = None
    self.sess = None
    self._graph_with = None

  def RunTensor(self, layer, init=True):
    """Convenience method to run a tensor."""
    if init:
      self.sess.run(tf.global_variables_initializer())
    if isinstance(layer, (tf.Tensor, six.string_types)):
      return self.sess.run(layer)
    elif layer.is_sequence():
      return self.sess.run(layer.sequence)
    else:
      return self.sess.run(layer)

  def Wrap(self, tensor, tensor_shape=None):
    """Convenience for prettytensor.wrap(tensor, self.bookkeeper)."""
    return prettytensor.wrap(tensor, self.bookkeeper, tensor_shape)

  def setUp(self):
    unittest.TestCase.setUp(self)
    self.SetBookkeeper(prettytensor.bookkeeper_for_new_graph())

  def tearDown(self):
    self.TearDownBookkeeper()
    unittest.TestCase.tearDown(self)

  def SetBookkeeper(self, m):
    """Used to set custom bookkeeper code."""
    self.TearDownBookkeeper()
    self.bookkeeper = m
    self._graph_with = self.bookkeeper.g.as_default()
    self._graph_with.__enter__()
    global _count
    _count += 1

    self.sess = tf.Session('')

  def TearDownBookkeeper(self):
    if self._graph_with:
      self._graph_with.__exit__(None, None, None)
      self._graph_with = None
    if self.sess:
      self.sess.close()
      self.sess = None
    self.bookkeeper = None
