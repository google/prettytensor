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
"""Image methods for PrettyTensor."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf

from prettytensor import layers
from prettytensor import pretty_tensor_class as prettytensor
from prettytensor import pretty_tensor_normalization_methods
from prettytensor.pretty_tensor_class import PAD_SAME
from prettytensor.pretty_tensor_class import PROVIDED


def _pool(input_layer, pool_fn, kernel, stride, edges, name):
  """Applies a pooling function."""
  input_layer.get_shape().assert_has_rank(4)
  if input_layer.get_shape().ndims not in (None, 4):
    raise ValueError('Pooling requires a rank 4 tensor: %s' %
                     input_layer.get_shape())
  kernel = _kernel(kernel)
  stride = _stride(stride)
  size = [1, kernel[0], kernel[1], 1]

  new_head = pool_fn(input_layer.tensor, size, stride, edges, name=name)
  return input_layer.with_tensor(new_head)


@prettytensor.Register
def average_pool(input_layer, kernel, stride, edges=PAD_SAME, name=PROVIDED):
  """Performs average pooling. The current head must be a rank 4 Tensor.

  Args:
    input_layer: The chainable object, supplied.
    kernel: The size of the patch for the pool, either an int or a length 1 or
      2 sequence (if length 1 or int, it is expanded).
    stride: The strides as a length 1, 2 or 4 sequence or an integer. If an
      int, length 1 or 2, the stride in the first and last dimensions are 1.
    edges: Either PAD_SAME' or PAD_VALID to control the padding.
    name: The name for this operation is also used to create/find the
      parameter variables.
  Returns:
    Handle to this layer.
  """
  return _pool(input_layer, tf.nn.avg_pool, kernel, stride, edges, name)


@prettytensor.Register
def max_pool(input_layer, kernel, stride, edges=PAD_SAME, name=PROVIDED):
  """Performs max pooling. The current head must be a rank 4 Tensor.

  Args:
    input_layer: The chainable object, supplied.
    kernel: The size of the patch for the pool, either an int or a length 1 or
      2 sequence (if length 1 or int, it is expanded).
    stride: The strides as a length 1, 2 or 4 sequence or an integer. If an
      int, length 1 or 2, the stride in the first and last dimensions are 1.
    edges: Either PAD_SAME or PAD_VALID to control the padding.
    name: The name for this operation is also used to create/find the
      parameter variables.
  Returns:
    Handle to this layer.
  """
  return _pool(input_layer, tf.nn.max_pool, kernel, stride, edges, name)


# pylint: disable=redefined-outer-name,invalid-name
@prettytensor.Register(
    assign_defaults=('activation_fn', 'l2loss', 'stddev', 'batch_normalize'))
class conv2d(prettytensor.VarStoreMethod):

  def __call__(self,
               input_layer,
               kernel,
               depth,
               activation_fn=None,
               stride=None,
               l2loss=None,
               init=None,
               stddev=None,
               bias=True,
               bias_init=tf.zeros_initializer,
               edges=PAD_SAME,
               batch_normalize=False,
               name=PROVIDED):
    """Adds a convolution to the stack of operations.

    The current head must be a rank 4 Tensor.

    Args:
      input_layer: The chainable object, supplied.
      kernel: The size of the patch for the pool, either an int or a length 1 or
        2 sequence (if length 1 or int, it is expanded).
      depth: The depth of the new Tensor.
      activation_fn: A tuple of (activation_function, extra_parameters). Any
        function that takes a tensor as its first argument can be used. More
        common functions will have summaries added (e.g. relu).
      stride: The strides as a length 1, 2 or 4 sequence or an integer. If an
        int, length 1 or 2, the stride in the first and last dimensions are 1.
      l2loss: Set to a value greater than 0 to use L2 regularization to decay
        the weights.
      init: An optional initialization. If not specified, uses Xavier
        initialization.
      stddev: A standard deviation to use in parameter initialization.
      bias: Set to False to not have a bias.
      bias_init: An initializer for the bias or a Tensor.
      edges: Either SAME to use 0s for the out of bounds area or VALID to shrink
        the output size and only uses valid input pixels.
      batch_normalize: Supply a BatchNormalizationArguments to set the
        parameters for batch normalization.
      name: The name for this operation is also used to create/find the
        parameter variables.
    Returns:
      Handle to the generated layer.
    Raises:
      ValueError: If head is not a rank 4 tensor or the  depth of the input
        (4th dim) is not known.
    """
    if input_layer.get_shape().ndims != 4:
      raise ValueError('conv2d requires a rank 4 Tensor with a known depth %s' %
                       input_layer.get_shape())
    if input_layer.shape[3] is None:
      raise ValueError('Input depth must be known')
    kernel = _kernel(kernel)
    stride = _stride(stride)
    size = [kernel[0], kernel[1], input_layer.shape[3], depth]

    books = input_layer.bookkeeper
    if init is None:
      if stddev is None:
        patch_size = size[0] * size[1]
        init = layers.he_init(size[2] * patch_size, size[3] * patch_size,
                              activation_fn)
      else:
        tf.logging.warning(
            'Passing `stddev` to initialize weight variable is deprecated and '
            'will be removed in the future. Pass '
            'tf.truncated_normal_initializer(stddev=stddev) or '
            'tf.zeros_initializer to `init` instead.')
        if stddev:
          init = tf.truncated_normal_initializer(stddev=stddev)
        else:
          init = tf.zeros_initializer
    elif stddev is not None:
      raise ValueError('Do not set both init and stddev.')
    dtype = input_layer.tensor.dtype
    params = self.variable('weights', size, init, dt=dtype)
    y = tf.nn.conv2d(input_layer, params, stride, edges)
    layers.add_l2loss(books, params, l2loss)
    if bias:
      y += self.variable('bias', [size[-1]], bias_init, dt=dtype)
    books.add_scalar_summary(
        tf.reduce_mean(layers.spatial_slice_zeros(y)),
        '%s/zeros_spatial' % y.op.name)
    y = pretty_tensor_normalization_methods.batch_normalize_with_arguments(
        y, batch_normalize)
    if activation_fn is not None:
      if not isinstance(activation_fn, collections.Sequence):
        activation_fn = (activation_fn,)
      y = layers.apply_activation(books,
                                  y,
                                  activation_fn[0],
                                  activation_args=activation_fn[1:])
    books.add_histogram_summary(y, '%s/activations' % y.op.name)
    return input_layer.with_tensor(y, parameters=self.vars)


@prettytensor.Register(
    assign_defaults=('activation_fn', 'l2loss', 'stddev', 'batch_normalize'))
class depthwise_conv2d(prettytensor.VarStoreMethod):

  def __call__(self,
               input_layer,
               kernel,
               channel_multiplier,
               activation_fn=None,
               stride=None,
               l2loss=None,
               init=None,
               stddev=None,
               bias=True,
               bias_init=tf.zeros_initializer,
               edges=PAD_SAME,
               batch_normalize=False,
               name=PROVIDED):
    """Adds a depth-wise convolution to the stack of operations.

    The current head must be a rank 4 Tensor.

    Args:
      input_layer: The chainable object, supplied.
      kernel: The size of the patch for the pool, either an int or a length 1 or
        2 sequence (if length 1 or int, it is expanded).
      channel_multiplier: Output channels will be a multiple of input channels.
      activation_fn: A tuple of (activation_function, extra_parameters). Any
        function that takes a tensor as its first argument can be used. More
        common functions will have summaries added (e.g. relu).
      stride: The strides as a length 1, 2 or 4 sequence or an integer. If an
        int, length 1 or 2, the stride in the first and last dimensions are 1.
      l2loss: Set to a value greater than 0 to use L2 regularization to decay
        the weights.
      init: An optional initialization. If not specified, uses Xavier
        initialization.
      stddev: A standard deviation to use in parameter initialization.
      bias: Set to False to not have a bias.
      bias_init: An initializer for the bias or a Tensor.
      edges: Either SAME to use 0s for the out of bounds area or VALID to shrink
        the output size and only uses valid input pixels.
      batch_normalize: Supply a BatchNormalizationArguments to set the
        parameters for batch normalization.
      name: The name for this operation is also used to create/find the
        parameter variables.
    Returns:
      Handle to the generated layer.
    Raises:
      ValueError: If head is not a rank 4 tensor or the depth of the input
        (4th dim) is not known.
    """
    if input_layer.get_shape().ndims != 4:
      raise ValueError(
          'depthwise_conv2d requires a rank 4 Tensor with a known depth %s' %
          input_layer.get_shape())
    if input_layer.shape[3] is None:
      raise ValueError('Input depth must be known')
    kernel = _kernel(kernel)
    stride = _stride(stride)
    size = [kernel[0], kernel[1], input_layer.shape[3], channel_multiplier]

    books = input_layer.bookkeeper
    if init is None:
      if stddev is None:
        patch_size = size[0] * size[1]
        init = layers.he_init(size[2] * patch_size, size[3] * patch_size,
                              activation_fn)
      else:
        tf.logging.warning(
            'Passing `stddev` to initialize weight variable is deprecated and '
            'will be removed in the future. Pass '
            'tf.truncated_normal_initializer(stddev=stddev) or '
            'tf.zeros_initializer to `init` instead.')
        if stddev:
          init = tf.truncated_normal_initializer(stddev=stddev)
        else:
          init = tf.zeros_initializer
    elif stddev is not None:
      raise ValueError('Do not set both init and stddev.')
    dtype = input_layer.tensor.dtype
    params = self.variable('weights', size, init, dt=dtype)
    y = tf.nn.depthwise_conv2d(input_layer, params, stride, edges)
    layers.add_l2loss(books, params, l2loss)
    if bias:
      y += self.variable('bias', [input_layer.shape[3] * channel_multiplier],
                         bias_init,
                         dt=dtype)
    books.add_scalar_summary(
        tf.reduce_mean(layers.spatial_slice_zeros(y)),
        '%s/zeros_spatial' % y.op.name)
    y = pretty_tensor_normalization_methods.batch_normalize_with_arguments(
        y, batch_normalize)
    if activation_fn is not None:
      if not isinstance(activation_fn, collections.Sequence):
        activation_fn = (activation_fn,)
      y = layers.apply_activation(books,
                                  y,
                                  activation_fn[0],
                                  activation_args=activation_fn[1:])
    books.add_histogram_summary(y, '%s/activations' % y.op.name)
    return input_layer.with_tensor(y, parameters=self.vars)
# pylint: enable=redefined-outer-name,invalid-name

# Helper methods


def _kernel(kernel_spec):
  """Expands the kernel spec into a length 2 list.

  Args:
    kernel_spec: An integer or a length 1 or 2 sequence that is expanded to a
      list.
  Returns:
    A length 2 list.
  """
  if isinstance(kernel_spec, tf.compat.integral_types):
    return [kernel_spec, kernel_spec]
  elif len(kernel_spec) == 1:
    return [kernel_spec[0], kernel_spec[0]]
  else:
    assert len(kernel_spec) == 2
    return kernel_spec


def _stride(stride_spec):
  """Expands the stride spec into a length 4 list.

  Args:
    stride_spec: If length 0, 1 or 2 then assign the inner dimensions, otherwise
      return stride_spec if it is length 4.
  Returns:
    A length 4 list.
  """
  if stride_spec is None:
    return [1, 1, 1, 1]
  elif isinstance(stride_spec, tf.compat.integral_types):
    return [1, stride_spec, stride_spec, 1]
  elif len(stride_spec) == 1:
    return [1, stride_spec[0], stride_spec[0], 1]
  elif len(stride_spec) == 2:
    return [1, stride_spec[0], stride_spec[1], 1]
  else:
    assert len(stride_spec) == 4
    return stride_spec
