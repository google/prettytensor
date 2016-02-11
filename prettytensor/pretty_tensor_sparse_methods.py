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
"""Holds PrettyTensor methods related to sparse data types."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from prettytensor import pretty_tensor_class as prettytensor


@prettytensor.Register
def to_dense_one_hot(labels, class_count):
  """Converts a vector that specified one-hot per batch into a dense version.

  Args:
    labels: The labels input.
    class_count: The number of classes as an int.
  Returns:
    One dense vector for each item in the batch.
  Raises:
    ValueError: If labels is not rank 1.
    TypeError: If class_count is not an integer or labels is not an integer
      Tensor.
  """
  if not isinstance(class_count, tf.compat.integral_types):
    raise TypeError('class_count must be an integer type.')
  if labels.dtype.base_dtype not in (tf.int32, tf.int64):
    raise TypeError('Labels must be an integer: %s' % labels.dtype)
  if labels.get_shape().ndims != 1:
    raise ValueError('Labels must be a rank 1 tensor: %s' % labels.get_shape())

  dtype = labels.dtype.base_dtype
  class_tensor = tf.convert_to_tensor(
      class_count, dtype=dtype, name='class_count')

  # Extract the batch from the shape so this is batch independent.
  batch = tf.gather(tf.shape(labels), 0)
  count = tf.expand_dims(tf.range(0, limit=batch), 1)
  labels = tf.expand_dims(labels, 1)
  batch = tf.gather(tf.shape(labels), 0)

  if dtype != tf.int32:
    count = tf.cast(count, dtype)
    batch = tf.cast(batch, dtype)

  result = tf.sparse_to_dense(tf.concat(1, [count, labels]),
                              tf.concat(0, [tf.expand_dims(batch, 0),
                                            tf.expand_dims(class_tensor, 0)]),
                              1.0, 0.0)
  result.set_shape([labels.get_shape().dims[0], class_count])
  return result
