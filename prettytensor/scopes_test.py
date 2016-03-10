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
"""Tests for scopes."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import traceback
import unittest


import tensorflow as tf

from prettytensor import scopes


def var_scoped_function():
  return tf.get_variable('dummy',
                         shape=[1],
                         initializer=tf.zeros_initializer)


class ScopesTest(unittest.TestCase):

  def test_skip_stack_frames(self):
    first = traceback.format_stack()
    second = traceback.format_stack()
    result = scopes.skip_common_stack_elements(first, second)
    self.assertEqual(1, len(result))
    self.assertNotEqual(len(first), len(result))

  def test_get_current_name_scope(self):
    self.assertEquals('/', scopes.get_current_name_scope())
    self.assertEquals('', scopes._get_last_part_of_name_scope('/'))
    with tf.name_scope('one') as scope:
      self.assertEquals(scope, scopes.get_current_name_scope())
      self.assertEquals('one', scopes._get_last_part_of_name_scope(scope))

    with tf.name_scope('one') as scope:
      self.assertEquals(scope, scopes.get_current_name_scope())
      self.assertEquals('one_1', scopes._get_last_part_of_name_scope(scope))
      with tf.name_scope('two') as nested_scope:
        self.assertEquals(nested_scope, scopes.get_current_name_scope())
      self.assertEquals('two',
                        scopes._get_last_part_of_name_scope(nested_scope))

  def test_template_without_name(self):
    tmpl1 = scopes.Template(None, var_scoped_function)

    v1 = tmpl1()
    v2 = tmpl1()
    self.assertEqual(v1, v2)
    self.assertEqual('dummy:0', v1.name)

  def test_template_with_name(self):
    tmpl1 = scopes.Template('s1', var_scoped_function)
    tmpl2 = scopes.Template('s1', var_scoped_function)

    v1 = tmpl1()
    v2 = tmpl1()
    v3 = tmpl2()
    self.assertEqual(v1, v2)
    self.assertNotEqual(v1, v3)
    self.assertEqual('s1/dummy:0', v1.name)
    self.assertEqual('s1_2/dummy:0', v3.name)

  def test_var_and_name_scope(self):
    with tf.Graph().as_default():
      with scopes.var_and_name_scope(('one', None)) as (ns, vs):
        self.assertEqual('one/', ns)
        self.assertEqual('one', vs.name)
      with scopes.var_and_name_scope(('one', None)) as (ns, vs):
        self.assertEqual('one_1/', ns)
        self.assertEqual('one_1', vs.name)
      with scopes.var_and_name_scope(('one/two', None)) as (ns, vs):
        self.assertEqual('one/two/', ns)
        self.assertEqual('one/two', vs.name)


if __name__ == '__main__':
  unittest.main()
