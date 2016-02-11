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
"""Provides helpers for feeding in numpy data to a TF graph.

These methods are intended to aid experimentation.  For large datasets consider
using readers and queues.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools



from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from prettytensor import bookkeeper


def feed_numpy(batch_size, *arrays):
  """Given a set of numpy arrays, produce slices of batch_size.

  Note: You can use itertools.cycle to have this repeat forever.

  Args:
    batch_size: The batch_size for each array.
    *arrays: A list of arrays.
  Yields:
    A list of slices from the arrays of length batch_size except the last one
    which will contain the rest.
  Raises:
    ValueError: If arrays aren't all the same length or no arrays are provided.
  """
  if not arrays:
    raise ValueError('Arrays cannot be empty.')
  size = len(arrays[0])
  for a in arrays:
    if size != len(a):
      raise ValueError('All arrays must be the same size.')
  count = int(size / batch_size)

  for i in xrange(count):
    start = i * batch_size
    end = start + batch_size
    yield [x[start:end] for x in arrays]
  if count * batch_size < size:
    yield [x[end:] for x in arrays]


def batch(input_iter, batch_size=32):
  """Batches data from an iterator that returns single items at a time."""
  input_iter = iter(input_iter)
  next_ = list(itertools.islice(input_iter, batch_size))
  while next_:
    yield next_
    next_ = list(itertools.islice(input_iter, batch_size))


def slice_constant(data, batch_size=32, name='constant_data', global_step=None):
  """Provide a slice based on the global_step.

  This is useful when the entire data array can be stored in memory because it
  allows you to feed the data very efficiently.

  Args:
    data: A numpy array or tensor.
    batch_size: The batch size for the produced data.
    name: An optional name for this data.
    global_step: A global step variable that is used to read the data. If None
      then the default prettytensor global_step is used.
  Returns:
    A tensor that produces the given data.
  """
  with tf.name_scope(name):
    all_data = tf.convert_to_tensor(data)
    global_step = global_step or bookkeeper.global_step()

    count = len(data) / batch_size
    extra = len(data) - count * batch_size

    if extra:
      offset = tf.mod(global_step, count)
      return tf.slice(all_data, offset * batch_size, batch_size)
    else:
      offset = tf.mod(global_step, count + 1)
      return tf.slice(all_data, offset * batch_size,
                      tf.select(tf.equal(offset, count), extra, batch_size))
