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
"""Reusable TensorFlow functions.

This file is divided into roughly 4 parts:

* _regression functions with signature: `fn(y, target, name=)` where y is
  the output of your DNN and target is the regression target as a tensor and
  the result is a tensor with shape [1].
* _distance functions with signature: `fn(t1, t2, name=) where t1 and t2 are
  both tensors and the result is a tensor with shape [N] where N is the first
  dimension of t1 and t2.
* Activation functions with signature `fn(x, name=)` where x is a tensor
  and the result is a tensor of the same shape.
* Utility functions. These include normalizations that are used in embedding
  models as a non-linearity and a few others.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# To improve numerical stability, we want to not do an exponential above this
# value.
_SOFTPLUS_STABILITY_LIMIT = 20.0


def l1_regression_loss(y, target, name=None):
  """Calculates the sum of absolute errors between y and target.

  Args:
    y: the calculated values.
    target: the desired values.
    name: the name for this op, defaults to l1_regression
  Returns:
    A tensorflow op.
  """
  with tf.op_scope([y, target], name, 'l1_regression') as scope:
    y = tf.convert_to_tensor(y, name='y')
    target = tf.convert_to_tensor(target, name='target')
    return reduce_batch_sum(tf.abs(y - target), name=scope)


def l2_regression_sq_loss(y, target, name=None):
  """Calculates the sum of squared errors between y and target.

  Args:
    y: the calculated values.
    target: the desired values.
    name: the name for this op, defaults to l2_regression
  Returns:
    A tensorflow op.
  """
  with tf.op_scope([y, target], name, 'l2_regression_sq') as scope:
    y = tf.convert_to_tensor(y, name='y')
    target = tf.convert_to_tensor(target, name='target')
    return reduce_batch_sum(tf.square(y - target), name=scope)


def reduce_batch_sum(x, name=None):
  with tf.op_scope([x], name, 'reduce_batch_sum') as scope:
    ndims = x.get_shape().ndims
    if ndims == 0:
      raise ValueError('Cannot reduce a scalar into batches.')
    elif ndims == 1:
      return x  # Don't include a useless sum.
    elif ndims:
      reduction_indices = list(range(1, x.get_shape().ndims))
      shape = [x.get_shape().dims[0]]
    else:
      reduction_indices = tf.range(1, tf.size(tf.shape(x)))
      shape = [None]  # We don't know much about the shape, but it is rank 1.
    result = tf.reduce_sum(x, reduction_indices=reduction_indices, name=scope)

    # Give a shape hint in case we have extra information.
    result.set_shape(shape)
    return result


def l2_regression_loss(y, target, name=None):
  """Calculates the square root of the SSE between y and target.

  Args:
    y: the calculated values.
    target: the desired values.
    name: the name for this op, defaults to l2_regression
  Returns:
    A tensorflow op.
  """
  with tf.op_scope([y, target], name, 'l2_regression') as scope:
    y = tf.convert_to_tensor(y, name='y')
    target = tf.convert_to_tensor(target, name='target')
    return tf.sqrt(l2_regression_sq_loss(y, target, name=scope))


def binary_cross_entropy_loss_with_logits(x, target, name=None):
  """Calculates the binary cross entropy between sigmoid(x) and target.

  Expects unscaled logits. Do not pass in results of sigmoid operation.

  Args:
    x: the calculated pre-sigmoid values
    target: the desired values.
    name: the name for this op, defaults to binary_cross_entropy_with_logits
  Returns:
    -(target * -softplus(-x) + (1-target) * (-x - softplus(-x)))
  Raises:
    ValueError: If shapes are incompatible.
  """
  with tf.op_scope([x, target], name,
                   'binary_cross_entropy_with_logits') as scope:
    x.get_shape().assert_is_compatible_with(target.get_shape())
    neg_softplus = -tf.nn.softplus(-x)
    return -tf.add(
        tf.mul(target, neg_softplus),
        tf.mul(1 - target, -x + neg_softplus),
        name=scope)


def cos_distance(t1, t2, epsilon=1e-12, name=None):
  """Cos distance between t1 and t2 and caps the gradient of the Square Root.

  Args:
    t1: A tensor
    t2: A tensor that can be multiplied by t1.
    epsilon: A lower bound value for the distance. The square root is used as
      the normalizer.
    name: Optional name for this op.
  Returns:
    The cos distance between t1 and t2.
  """
  with tf.op_scope([t1, t2], name, 'cos_distance') as scope:
    t1 = tf.convert_to_tensor(t1, name='t1')
    t2 = tf.convert_to_tensor(t2, name='t2')
    x_inv_norm = tf.rsqrt(tf.maximum(length_squared(t1) * length_squared(t2),
                                     epsilon))
    return tf.sub(1.0,
                  dot_product(t1, t2) * x_inv_norm,
                  name=scope)


def dot_distance(t1, t2, name=None):
  """dot "distance" between t1 and t2.

  Args:
    t1: A tensor.
    t2: A tensor that is the same size as t1.
    name: Optional name for this op.
  Returns:
    The dot distance between t1 and t2.
  """
  with tf.op_scope([t1, t2], name, 'dot_distance') as scope:
    return -dot_product(t1, t2, name=scope)


def l2_distance_sq(t1, t2, name=None):
  """Square of l2 distance between t1 and t2.

  Args:
    t1: A tensor.
    t2: A tensor that is the same size as t1.
    name: Optional name for this op.
  Returns:
    The l2 distance between t1 and t2.
  """
  with tf.op_scope([t1, t2], name, 'l2_distance_sq') as scope:
    t1 = tf.convert_to_tensor(t1, name='t1')
    t2 = tf.convert_to_tensor(t2, name='t2')
    return length_squared(tf.sub(t1, t2), name=scope)


def l2_distance(t1, t2, epsilon=1e-12, name=None):
  """l2 distance between t1 and t2 and caps the gradient of the Square Root.

  Args:
    t1: A tensor.
    t2: A tensor that is the same size as t1.
    epsilon: A lower bound for distance, useful to avoid sqrt of very small
      values that can blow up gradients.
    name: Optional name for this op.
  Returns:
    The l2 distance between t1 and t2.
  """
  with tf.op_scope([t1, t2], name, 'l2_distance') as scope:
    t1 = tf.convert_to_tensor(t1, name='t1')
    t2 = tf.convert_to_tensor(t2, name='t2')
    return tf.sqrt(tf.maximum(l2_distance_sq(t1, t2, scope), epsilon))


def l1_distance(t1, t2, name=None):
  """l1 distance between t1 and t2.

  Args:
    t1: A tensor.
    t2: A tensor that is the same size as t1.
    name: Optional name for this op.
  Returns:
    The l1 distance between t1 and t2.
  """
  with tf.op_scope([t1, t2], name, 'l1_distance') as scope:
    t1 = tf.convert_to_tensor(t1, name='t1')
    t2 = tf.convert_to_tensor(t2, name='t2')
    sub = tf.sub(t1, t2)
    reduction_dim = _last_index(sub, 1)
    return tf.reduce_sum(tf.abs(sub), reduction_dim, name=scope)


def leaky_relu(x, name=None):
  """Creates a leaky_relu.

  This is an alternate non-linearity to relu. The leaky part of the relu may
  prevent dead Neurons in a model since the gradient doesn't go completely to
  0.

  Args:
    x: The input tensor.
    name: Optional name for this op.
  Returns:
    x if x > 0 otherwise 0.01 * x.
  """
  with tf.op_scope([x], name, 'leaky_relu') as scope:
    x = tf.convert_to_tensor(x, name='x')
    return tf.select(tf.less(x, 0.0), 0.01 * x, x, name=scope)


def softplus(x, scale=1.0, name=None):
  """Computes softplus with a scale factor to sharpen of the hinge.

  This is an alternate non-linearity to relu. It has a similar shape, but
  it has a smooth transition from the linear part to 0.

  Args:
    x: A tensor.
    scale: A float that sharpens the curve.
    name: Optional name.
  Returns:
    y = log(1 + exp(x))

  """
  with tf.op_scope([x], name, 'softplus') as scope:
    x = tf.convert_to_tensor(x, name='x')
    # y ~= x when x > _SOFTPLUS_STABILITY_LIMIT.  This improves numerical
    # stability.
    return tf.select(
        tf.greater(x, _SOFTPLUS_STABILITY_LIMIT / scale),
        x,
        tf.log(tf.exp(x * scale) + 1.0) / scale,
        name=scope)


# Copied to keep API consistency with other functions.
l2_normalize = tf.nn.l2_normalize


def l1_normalize(x, dim, epsilon=1e-12, name=None):
  """l1 normalizes x.

  Args:
    x: The tensor to normalize.
    dim: The dimension to normalize along.
    epsilon: Lower bound on the norm, used to avoid exploding gradients as the
      norm approaches 0.
    name: Optional name for this op.
  Returns:
    x normalized along dim.
  """
  with tf.op_scope([x], name, 'l1_normalize') as scope:
    x = tf.convert_to_tensor(x, name='x')
    x = tf.verify_tensor_all_finite(x, 'Error at input %s' % scope)
    x_norm = tf.maximum(tf.reduce_sum(tf.abs(x), [dim], keep_dims=True),
                        epsilon)
    return tf.div(x, x_norm, name=scope)


def every_other(x, name=None):
  """Drops every other value from the tensor and returns a 1D tensor.

  This is useful if you are running multiple inputs through a model tower
  before splitting them and you want to line it up with some other data.

  Args:
    x: the target tensor.
    name: the name for this op, defaults to every_other
  Returns:
    A tensorflow op.
  """
  with tf.op_scope([x], name, 'every_other') as scope:
    x = tf.convert_to_tensor(x, name='x')
    return tf.reshape(
        tf.slice(
            tf.reshape(x, [-1, 2]), [0, 0], [-1, 1]),
        [-1],
        name=scope)


def dot_product(t1, t2, keep_dims=False, name=None, reduction_dim=None):
  """Computes the dot product of t1 and t2.

  Args:
    t1: A rank 2 tensor.
    t2: A tensor that is the same size as t1.
    keep_dims: If true, reduction does not change the rank of the input.
    name: Optional name for this op.
    reduction_dim: The dimension to reduce, by default choose the last one
      and if no shape is specified guess 1.
  Returns:
    The dot product.
  """
  with tf.op_scope([t1, t2], name, 'dot') as scope:
    t1 = tf.convert_to_tensor(t1, name='t1')
    t2 = tf.convert_to_tensor(t2, name='t2')
    mul = tf.mul(t1, t2)
    if not reduction_dim:
      reduction_dim = _last_index(mul, 1)
    return tf.reduce_sum(mul, reduction_dim, name=scope, keep_dims=keep_dims)


def length_squared(x, keep_dims=False, name=None, reduction_dim=None):
  """Computes the squared length of x.

  Args:
    x: A tensor.
    keep_dims: If true, reduction does not change the rank of the input.
    name: Optional name for this op.
    reduction_dim: The dimension to reduce, by default choose the last one
      and if no shape is specified guess 1.
  Returns:
    The squared length of x.
  """
  with tf.op_scope([x], name, 'length_squared') as scope:
    x = tf.convert_to_tensor(x, name='x')
    if not reduction_dim:
      reduction_dim = _last_index(x, 1)
    return tf.reduce_sum(
        tf.square(x),
        reduction_dim,
        keep_dims=keep_dims,
        name=scope)


def unzip(x, split_dim, current_length, num_splits=2, name=None):
  """Splits a tensor by unzipping along the split_dim.

  For example the following array split into 2 would be:
      [1, 2, 3, 4, 5, 6] -> [1, 3, 5], [2, 4, 6]
  and by 3:
      [1, 2, 3, 4] -> [1, 4], [2], [3]

  Args:
    x: The tensor to split.
    split_dim: The dimension to split along.
    current_length: Current length along the split_dim.
    num_splits: The number of splits.
    name: Optional name for this op.
  Returns:
    A length num_splits sequence.
  """
  with tf.op_scope([x], name, 'unzip') as scope:
    x = tf.convert_to_tensor(x, name='x')
    # There is probably a more efficient way to do this.
    all_splits = tf.split(split_dim, current_length, x, name=scope)
    splits = [[] for _ in xrange(num_splits)]
    for i in xrange(current_length):
      splits[i % num_splits].append(all_splits[i])
    return [tf.concat(split_dim, s) for s in splits]


def _last_index(x, default_dim):
  """Returns the last dimension's index or default_dim if x has no shape."""
  if x.get_shape().ndims is not None:
    return len(x.get_shape()) - 1
  else:
    return default_dim


def _all_dims(x, default_dims=None):
  """Returns a list of dims in x or default_dims if the rank is unknown."""
  if x.get_shape().ndims is not None:
    return list(xrange(x.get_shape().ndims))
  else:
    return default_dims
