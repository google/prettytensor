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
"""Utility class for testing recurrent networks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class SequenceInputMock(object):
  """A sequence input mock for testing recurrent networks."""

  def __init__(self, bookkeeper, input_list, label_list, node_depth):
    self.inputs = self.for_constants(input_list)
    self.targets = self.for_constants(label_list)
    self.node_depth = node_depth
    self.batch_size = input_list[0].shape[0]
    self.requested_tensors = {}
    self.bookkeeper = bookkeeper
    self.num_timesteps = len(input_list)

  def for_constants(self, ls):
    return [tf.constant(x, dtype=tf.float32) for x in ls]

  def state(self, state_name):
    """Returns, creating if necessary, a state variable with the given name."""
    if state_name not in self.requested_tensors:
      count = tf.get_variable('count_%s' % state_name,
                              [],
                              tf.int32,
                              tf.zeros_initializer(),
                              trainable=False)
      value = tf.get_variable(state_name, [self.batch_size, self.node_depth],
                              tf.float32, tf.zeros_initializer())
      self.requested_tensors[state_name] = (count, value)

    return self.requested_tensors[state_name][1]

  def save_state(self, state_name, unused_value, name='SaveState'):
    return tf.assign_add(self.requested_tensors[state_name][0], 1, name=name)

