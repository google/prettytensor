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
"""Test class for bookkeepers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest


import tensorflow as tf

from prettytensor import bookkeeper
from prettytensor import pretty_tensor_testing


class BookkeeperTest(pretty_tensor_testing.PtTestCase):

  def setUp(self):
    super(self.__class__, self).setUp()

  def assertSameContents(self, list1, list2, msg):
    self.assertEqual(len(list1), len(list2), msg)
    self.assertEqual(set(list1), set(list2), msg)

  def testGraphIsReused(self):
    b1 = bookkeeper.for_default_graph()
    b2 = bookkeeper.for_default_graph()
    self.assertTrue(b1 is b2)

  def testPassingArgsCausesError(self):
    b1 = bookkeeper.for_new_graph()
    with b1.g.as_default(), self.assertRaises(ValueError):
      bookkeeper.for_default_graph(global_step=None)

  def testGlobalStep(self):
    v = tf.Variable(0)
    b1 = bookkeeper.for_new_graph(global_step=v)
    with b1.g.as_default():
      self.assertEqual(v, bookkeeper.global_step())
    with self.assertRaises(ValueError):
      bookkeeper.for_new_graph(global_step=tf.Variable(1.0))

  def testUniqueBookkeeperPerGraph(self):
    b1 = bookkeeper.for_default_graph()
    with tf.Graph().as_default():
      b2 = bookkeeper.for_default_graph()
    self.assertFalse(b1 is b2)

  def testBareVarName(self):
    name = 'hello'
    var = tf.Variable([1], name=name)
    self.assertEquals(name, bookkeeper._bare_var_name(var))
    self.assertEquals(name,
                      bookkeeper._bare_var_name(var._as_graph_element()))


if __name__ == '__main__':
  unittest.main()
