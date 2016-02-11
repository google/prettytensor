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
"""Provides a class that just implements a sequence with a delta count."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


class SequenceWithDeltas(collections.MutableSequence):
  """Provides a sequence with a count of modifications."""

  def __init__(self, other_seq=None):
    if other_seq is None:
      self._seq = []
    else:
      self._seq = list(other_seq)
    self._mods = len(self._seq)
    self._mark = 0

  def __getitem__(self, key):
    return self._seq[key]

  def __setitem__(self, key, value):
    self._mods += 1
    self._seq[key] = value

  def __delitem__(self, key):
    self._mods += 1
    del self._seq[key]

  def __len__(self):
    return len(self._seq)

  def insert(self, key, value):
    self._mods += 1
    self._seq.insert(key, value)

  @property
  def deltas(self):
    return self._mods

  def mark(self):
    """Marks this sequence at the current number of deltas."""
    self._mark = self._mods

  def has_changed(self):
    """Returns if it has changed since the last mark."""
    return self._mark == self._mods
