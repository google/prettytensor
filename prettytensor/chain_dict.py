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
"""Creates a dict with a parent so that missing values are sent up."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


class ChainDict(collections.MutableMapping):
  """The Name class."""

  def __init__(self, parent):
    self._map = {}
    self._parent = parent
    self._dead_count = 0

  def __getitem__(self, key):
    if key in self._map:
      return self._map[key]
    elif self._parent:
      return self._parent[key]
    else:
      raise KeyError('Key not found: %s' % key)

  def __setitem__(self, key, value):
    self._map[key] = value

  def __delitem__(self, key):
    raise Exception('Deleting items not supported.')

  def _full_map(self):
    """Creates a full mapping of this and all parent key, value pairs."""
    result = {}
    if self._parent:
      result.update(self._parent)
    result.update(self._map)
    return result

  def __iter__(self):
    return self._full_map().__iter__()

  def __len__(self):
    return len(self._full_map())
