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
"""Test class for ChainDict."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from prettytensor import chain_dict


class ChainDictTest(unittest.TestCase):

  def testSet(self):
    d = chain_dict.ChainDict(None)
    d['KEY'] = 'VALUE'
    self.assertEqual({'KEY': 'VALUE'}, d._map)

  def testGetNoParent(self):
    d = chain_dict.ChainDict(None)
    d['KEY'] = 'VALUE'
    self.assertEqual('VALUE', d['KEY'])

  def testGetAbsentNoParent(self):
    d = chain_dict.ChainDict(None)
    with self.assertRaises(KeyError):
      # pylint: disable=pointless-statement
      d['KEY']

  def testGetInParent(self):
    parent = chain_dict.ChainDict(None)
    d = chain_dict.ChainDict(parent)
    parent['KEY'] = 'VALUE'
    self.assertEqual('VALUE', parent['KEY'])
    self.assertEqual('VALUE', d['KEY'])

  def testGetNotInParent(self):
    parent = chain_dict.ChainDict(None)
    d = chain_dict.ChainDict(parent)
    with self.assertRaises(KeyError):
      # pylint: disable=pointless-statement
      parent['KEY']
    with self.assertRaises(KeyError):
      # pylint: disable=pointless-statement
      d['KEY']

  def testGetOverridden(self):
    parent = chain_dict.ChainDict(None)
    d = chain_dict.ChainDict(parent)
    parent['KEY'] = 'VALUE'
    d['KEY'] = 'OTHER_VALUE'
    self.assertEqual('VALUE', parent['KEY'])
    self.assertEqual('OTHER_VALUE', d['KEY'])

  def testLen(self):
    parent = chain_dict.ChainDict(None)
    d = chain_dict.ChainDict(parent)
    self.assertEqual(0, len(d))
    self.assertEqual(0, len(parent))

    parent['KEY'] = 'VALUE'
    self.assertEqual(1, len(d))
    self.assertEqual(1, len(parent))

    d['KEY'] = 'OTHER_VALUE'
    self.assertEqual(1, len(d))
    self.assertEqual(1, len(parent))

    d['OTHER_KEY'] = 'YAV'
    self.assertEqual(2, len(d))
    self.assertEqual(1, len(parent))

  def testIteration(self):
    parent = chain_dict.ChainDict(None)
    d = chain_dict.ChainDict(parent)
    parent['KEY'] = 'VALUE'
    d['KEY'] = 'OTHER_VALUE'
    d['OTHER_KEY'] = 'YAV'

    self.assertEqual([('KEY', 'OTHER_VALUE'), ('OTHER_KEY', 'YAV')],
                     sorted(d.items()))
    self.assertEqual([('KEY', 'VALUE')], sorted(parent.items()))

  def testPlainDictParent(self):
    d = chain_dict.ChainDict({'KEY': 'VALUE'})
    self.assertEqual('VALUE', d['KEY'])
    self.assertEqual(len(d), 1)
    # In Python 3, items produces an iterator.
    self.assertEqual(list(d.items()), [('KEY', 'VALUE')])

if __name__ == '__main__':
  unittest.main()
