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
from prettytensor import parameters
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
  """Performs average pooling.

  `kernel` is the patch that will be pooled and it describes the pooling along
  each of the 4 dimensions.  `stride` is how big to take each step.

  Because more often than not, pooling is only done
  on the width and height of the image, the following shorthands are supported:

  * scalar (e.g. 3): Square pooling on the image
      (`[b, c, r, d] = [1, 3, 3, 1]`).
  * singleton list (e.g. [3]): Square pooling on the image
      (`[b, c, r, d] = [1, 3, 3, 1]`).
  * list of length 2 (e.g. [3, 2]): Square pooling on the image
      (`[b, c, r, d] = [1, 3, 2, 1]`).

  Args:
    input_layer: The chainable object, supplied.
    kernel: The size of the patch for the pool, either an int or a length 1 or
      2 sequence (if length 1 or int, it is expanded).
    stride: The strides as a length 1, 2 or 4 sequence or an integer. If an
      int, length 1 or 2, the stride in the first and last dimensions are 1.
    edges: Either `pt.PAD_SAME`' or `pt.PAD_VALID` to control the padding.
    name: The name for this operation is also used to create/find the
      parameter variables.
  Returns:
    Handle to this layer.
  """
  return _pool(input_layer, tf.nn.avg_pool, kernel, stride, edges, name)


@prettytensor.Register
def max_pool(input_layer, kernel, stride, edges=PAD_SAME, name=PROVIDED):
  """Performs max pooling.

  `kernel` is the patch that will be pooled and it describes the pooling along
  each of the 4 dimensions.  `stride` is how big to take each step.

  Because more often than not, pooling is only done
  on the width and height of the image, the following shorthands are supported:

  * scalar (e.g. 3): Square pooling on the image
      (`[b, c, r, d] = [1, 3, 3, 1]`).
  * singleton list (e.g. [3]): Square pooling on the image
      (`[b, c, r, d] = [1, 3, 3, 1]`).
  * list of length 2 (e.g. [3, 2]): Square pooling on the image
      (`[b, c, r, d] = [1, 3, 2, 1]`).

  Args:
    input_layer: The chainable object, supplied.
    kernel: The size of the patch for the pool, either an int or a length 1 or
      2 sequence (if length 1 or int, it is expanded).
    stride: The strides as a length 1, 2 or 4 sequence or an integer. If an
      int, length 1 or 2, the stride in the first and last dimensions are 1.
    edges: Either `pt.PAD_SAME` or `pt.PAD_VALID` to control the padding.
    name: The name for this operation is also used to create/find the
      parameter variables.
  Returns:
    Handle to this layer.
  """
  return _pool(input_layer, tf.nn.max_pool, kernel, stride, edges, name)


@prettytensor.Register
def bilinear_sampling(input_layer, x, y, name=PROVIDED):
  """Performs bilinear sampling. This must be a rank 4 Tensor.

  Implements the differentiable sampling mechanism with bilinear kernel
  in https://arxiv.org/abs/1506.02025.

  Given (x, y) coordinates for each output pixel, use bilinear sampling on
  the input_layer to fill the output.

  Args:
    input_layer: The chainable object, supplied.
    x: A tensor of size [batch_size, height, width, 1] representing the sampling
      x coordinates normalized to range [-1,1].
    y: A tensor of size [batch_size, height, width, 1] representing  the
      sampling y coordinates normalized to range [-1,1].
    name: The name for this operation is also used to create/find the
      parameter variables.
  Returns:
    Handle to this layer
  """
  input_layer.get_shape().assert_has_rank(4)
  return _interpolate(im=input_layer, x=x, y=y, name=name)


# pylint: disable=redefined-outer-name,invalid-name
@prettytensor.Register(
    assign_defaults=('activation_fn', 'l2loss', 'batch_normalize',
                     'parameter_modifier', 'phase'))
class conv2d(prettytensor.VarStoreMethod):

  def __call__(self,
               input_layer,
               kernel,
               depth,
               activation_fn=None,
               stride=(1, 1),
               l2loss=None,
               weights=None,
               bias=tf.zeros_initializer,
               edges=PAD_SAME,
               batch_normalize=False,
               phase=prettytensor.Phase.train,
               parameter_modifier=parameters.identity,
               name=PROVIDED):
    """Adds a convolution to the stack of operations.

    `kernel` is the patch that will be pooled and it describes the pooling
    along each of the 4 dimensions.  The stride is how big to take each step.

    * scalar (e.g. 3): Square pooling on the image
        (`[b, c, r, d] = [1, 3, 3, 1]`).
    * singleton list (e.g. [3]): Square pooling on the image
        (`[b, c, r, d] = [1, 3, 3, 1]`).
    * list of length 2 (e.g. [3, 2]): Square pooling on the image
        (`[b, c, r, d] = [1, 3, 2, 1]`).

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
      weights:  An initializer for weights or a Tensor. If not specified,
        uses He's initialization.
      bias: An initializer for the bias or a Tensor. No bias if set to None.
      edges: Either SAME to use 0s for the out of bounds area or VALID to shrink
        the output size and only uses valid input pixels.
      batch_normalize: Supply a BatchNormalizationArguments to set the
        parameters for batch normalization.
      phase: The phase of graph construction.  See `pt.Phase`.
      parameter_modifier: A function to modify parameters that is applied after
        creation and before use.
      name: The name for this operation is also used to create/find the
        parameter variables.
    Returns:
      Handle to the generated layer.
    Raises:
      ValueError: If input_layer is not a rank 4 tensor or the  depth of the
        input (4th dim) is not known.
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
    if weights is None:
      patch_size = size[0] * size[1]
      weights = layers.he_init(size[2] * patch_size, size[3] * patch_size,
                               activation_fn)

    dtype = input_layer.tensor.dtype
    params = parameter_modifier(
        'weights',
        self.variable('weights', size, weights, dt=dtype),
        phase)
    y = tf.nn.conv2d(input_layer, params, stride, edges)
    layers.add_l2loss(books, params, l2loss)
    if bias is not None:
      y += parameter_modifier('bias',
                              self.variable('bias', [size[-1]],
                                            bias,
                                            dt=dtype),
                              phase)
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
    assign_defaults=('activation_fn', 'l2loss', 'batch_normalize',
                     'parameter_modifier', 'phase'))
class depthwise_conv2d(prettytensor.VarStoreMethod):

  def __call__(self,
               input_layer,
               kernel,
               channel_multiplier,
               activation_fn=None,
               stride=None,
               l2loss=None,
               weights=None,
               bias=tf.zeros_initializer,
               edges=PAD_SAME,
               batch_normalize=False,
               phase=prettytensor.Phase.train,
               parameter_modifier=parameters.identity,
               name=PROVIDED):
    """Adds a depth-wise convolution to the stack of operations.

    A depthwise convolution performs the convolutions one channel at a time and
    produces an output with depth `channel_multiplier * input_depth`.

    `kernel` is the patch that will be pooled and it describes the pooling
    along each of the 4 dimensions.  The stride is how big to take each step.

    * scalar (e.g. 3): Square pooling on the image
        (`[b, c, r, d] = [1, 3, 3, 1]`).
    * singleton list (e.g. [3]): Square pooling on the image
        (`[b, c, r, d] = [1, 3, 3, 1]`).
    * list of length 2 (e.g. [3, 2]): Square pooling on the image
        (`[b, c, r, d] = [1, 3, 2, 1]`).

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
      weights:  An initializer for weights or a Tensor. If not specified,
        uses He's initialization.
      bias: An initializer for the bias or a Tensor. No bias if set to None.
      edges: Either `pt.DIM_SAME` to use 0s for the out of bounds area or
        `pt.DIM_VALID` to shrink the output size and only uses valid input
        pixels.
      batch_normalize: Supply a BatchNormalizationArguments to set the
        parameters for batch normalization.
      phase: The phase of graph construction.  See `pt.Phase`.
      parameter_modifier: A function to modify parameters that is applied after
        creation and before use.
      name: The name for this operation is also used to create/find the
        parameter variables.
    Returns:
      Handle to the generated layer.
    Raises:
      ValueError: If input_layer is not a rank 4 tensor or the depth of the
        input (4th dim) is not known.
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
    if weights is None:
      patch_size = size[0] * size[1]
      weights = layers.he_init(size[2] * patch_size, size[3] * patch_size,
                               activation_fn)

    dtype = input_layer.tensor.dtype
    params = parameter_modifier(
        'weights',
        self.variable('weights', size, weights, dt=dtype),
        phase)
    y = tf.nn.depthwise_conv2d(input_layer, params, stride, edges)
    layers.add_l2loss(books, params, l2loss)
    if bias is not None:
      y += parameter_modifier(
          'bias',
          self.variable('bias', [input_layer.shape[3] * channel_multiplier],
                        bias,
                        dt=dtype),
          phase)
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


def _interpolate(im, x, y, name):
  """Perform bilinear sampling on im given x,y coordiantes.

  Implements the differentiable sampling mechanism with bilinear kerenl
  in https://arxiv.org/abs/1506.02025.

  Modified from https://github.com/tensorflow/models/tree/master/transformer

  x,y are tensors specifying normalized coordinates [-1,1] to sampled on im.
  (e.g.) (-1,-1) in x,y corresponds to pixel location (0,0) in im, and
  (1,1) in x,y corresponds to the bottom right pixel in im.

  Args:
    im: A tensor of size [batch_size, height, width, channels]
    x: A tensor of size [batch_size, height, width, 1] representing the sampling
      x coordinates normalized to range [-1,1].
    y: A tensor of size [batch_size, height, width, 1] representing  the
      sampling y coordinates normalized to range [-1,1].
    name: The name for this operation is also used to create/find the
        parameter variables.
  Returns:
    A tensor of size [batch_size, height, width, channels]
  """
  with tf.variable_scope(name):
    x = tf.reshape(x, [-1])
    y = tf.reshape(y, [-1])

    # constants
    num_batch = tf.shape(im)[0]
    _, height, width, channels = im.get_shape().as_list()

    x = tf.to_float(x)
    y = tf.to_float(y)
    height_f = tf.cast(height, 'float32')
    width_f = tf.cast(width, 'float32')
    zero = tf.constant(0, dtype=tf.int32)
    max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
    max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

    # scale indices from [-1, 1] to [0, width-1/height-1]
    x = (x + 1.0) * (width_f - 1.0) / 2.0
    y = (y + 1.0) * (height_f - 1.0) / 2.0

    # do sampling
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)
    dim2 = width
    dim1 = width * height

    # Create base index
    base = tf.range(num_batch) * dim1
    base = tf.reshape(base, [-1, 1])
    base = tf.tile(base, [1, height * width])
    base = tf.reshape(base, [-1])

    base_y0 = base + y0 * dim2
    base_y1 = base + y1 * dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # use indices to lookup pixels in the flat image and restore channels dim
    im_flat = tf.reshape(im, tf.stack([-1, channels]))
    im_flat = tf.to_float(im_flat)
    pixel_a = tf.gather(im_flat, idx_a)
    pixel_b = tf.gather(im_flat, idx_b)
    pixel_c = tf.gather(im_flat, idx_c)
    pixel_d = tf.gather(im_flat, idx_d)

    # and finally calculate interpolated values
    x1_f = tf.to_float(x1)
    y1_f = tf.to_float(y1)

    wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
    wb = tf.expand_dims((x1_f - x) * (1.0 - (y1_f - y)), 1)
    wc = tf.expand_dims(((1.0 - (x1_f - x)) * (y1_f - y)), 1)
    wd = tf.expand_dims(((1.0 - (x1_f - x)) * (1.0 - (y1_f - y))), 1)

    output = tf.add_n([wa * pixel_a, wb * pixel_b, wc * pixel_c, wd * pixel_d])
    output = tf.reshape(
        output, shape=tf.stack([num_batch, height, width, channels]))
    return output
