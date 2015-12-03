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
import collections

import tensorflow as tf

from prettytensor import layers
from prettytensor import pretty_tensor_class as prettytensor
from prettytensor.pretty_tensor_class import PAD_SAME
from prettytensor.pretty_tensor_class import Phase
from prettytensor.pretty_tensor_class import PROVIDED


# pylint: disable=invalid-name
@prettytensor.Register(
    assign_defaults=('learned_moments_update_rate', 'variance_epsilon',
                     'scale_after_normalization', 'phase'))
class batch_normalize(prettytensor.VarStoreMethod):

  def __call__(self,
               input_layer,
               name=PROVIDED,
               learned_moments_update_rate=None,
               variance_epsilon=None,
               scale_after_normalization=None,
               phase=Phase.train):
    """Batch normalize this layer.

    This only supports global batch normalization and it can be enabled for all
    convolutional layers by setting the default 'batch_normalize' to True.
    learned_moments_update_rate, variance_epsilon and scale_after_normalization
    need to either be set here or be set in defaults as well.

    Args:
      input_layer: The chainable object, supplied.
      name: The name for this operation is also used to create/find the
        parameter variables.
      learned_moments_update_rate: Update rate for the learned moments.
      variance_epsilon: A float. A small float number to avoid dividing by 0.
      scale_after_normalization: A bool indicating whether the resulted tensor
        needs to be multiplied with gamma.
      phase: The phase of construction.
    Returns:
      Handle to the generated layer.
    """
    assert isinstance(learned_moments_update_rate, (int, long, float))
    assert isinstance(variance_epsilon, (int, long, float))
    assert scale_after_normalization is not None

    # Allocate variables to hold the moving averages of the moments.
    params_shape = [input_layer.shape[-1]]

    # Allocate parameters for the beta and gamma of the normalization.
    beta = self.variable('beta', params_shape, tf.constant_initializer(0.0))
    gamma = self.variable('gamma', params_shape, tf.constant_initializer(1.0))
    moving_mean = (self.variable('moving_mean',
                                 params_shape,
                                 tf.constant_initializer(0.0),
                                 train=False))
    moving_variance = self.variable('moving_variance',
                                    params_shape,
                                    tf.constant_initializer(1.0),
                                    train=False)

    if phase == Phase.train:
      # Calculate the moments based on the individual batch.
      mean, variance = tf.nn.moments(input_layer.tensor, [0, 1, 2])
      input_layer.bookkeeper.add_histogram_summary(mean)
      input_layer.bookkeeper.add_histogram_summary(variance)

      input_layer.bookkeeper.exponential_moving_average(
          mean, moving_mean, 1.0 - learned_moments_update_rate)
      input_layer.bookkeeper.exponential_moving_average(
          variance, moving_variance, 1.0 - learned_moments_update_rate)
    else:
      # Load the mean and variance as the 'moving average' moments
      # from the checkpoint.
      mean = moving_mean
      variance = moving_variance

    # Normalize the activations.
    y = tf.nn.batch_norm_with_global_normalization(
        input_layer.tensor,
        mean,
        variance,
        beta,
        gamma,
        name=name,
        scale_after_normalization=scale_after_normalization,
        variance_epsilon=variance_epsilon)

    return input_layer.with_tensor(y)
# pylint: enable=invalid-name


def _pool(input_layer, pool_fn, kernel, stride, edges, name):
  """Applies a pooling function."""
  assert len(input_layer.shape) == 4
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
               name=PROVIDED,
               stride=None,
               activation_fn=None,
               l2loss=None,
               init=None,
               stddev=None,
               bias=True,
               edges=PAD_SAME,
               batch_normalize=False):
    """Adds a convolution to the stack of operations.

    The current head must be a rank 4 Tensor.

    Args:
      input_layer: The chainable object, supplied.
      kernel: The size of the patch for the pool, either an int or a length 1 or
        2 sequence (if length 1 or int, it is expanded).
      depth: The depth of the new Tensor.
      name: The name for this operation is also used to create/find the
        parameter variables.
      stride: The strides as a length 1, 2 or 4 sequence or an integer. If an
        int, length 1 or 2, the stride in the first and last dimensions are 1.
      activation_fn: A tuple of (activation_function, extra_parameters). Any
        function that takes a tensor as its first argument can be used. More
        common functions will have summaries added (e.g. relu).
      l2loss: Set to a value greater than 0 to use L2 regularization to decay
        the weights.
      init: An optional initialization. If not specified, uses Xavier
        initialization.
      stddev: A standard deviation to use in parameter initialization.
      bias: Set to False to not have a bias.
      edges: Either SAME to use 0s for the out of bounds area or VALID to shrink
        the output size and only uses valid input pixels.
      batch_normalize: Set to True to batch_normalize this layer.
    Returns:
      Handle to the generated layer.
    Raises:
      ValueError: If head is not a rank 4 tensor or the  depth of the input
        (4th dim) is not known.
    """
    if len(input_layer.shape) != 4:
      raise ValueError(
          'Cannot perform conv2d on tensor with shape %s' % input_layer.shape)
    if input_layer.shape[3] is None:
      raise ValueError('Input depth must be known')
    kernel = _kernel(kernel)
    stride = _stride(stride)
    size = [kernel[0], kernel[1], input_layer.shape[3], depth]

    books = input_layer.bookkeeper
    if init is None:
      if stddev is None:
        patch_size = size[0] * size[1]
        init = layers.xavier_init(size[2] * patch_size, size[3] * patch_size)
      elif stddev:
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
      y += self.variable(
          'bias',
          [size[-1]],
          tf.zeros_initializer,
          dt=dtype)
    books.add_scalar_summary(
        tf.reduce_mean(
            layers.spatial_slice_zeros(y)), '%s/zeros_spatial' % y.op.name)
    if batch_normalize:
      y = input_layer.with_tensor(y).batch_normalize()
    if activation_fn is not None:
      if not isinstance(activation_fn, collections.Sequence):
        activation_fn = (activation_fn,)
      y = layers.apply_activation(
          books,
          y,
          activation_fn[0],
          activation_args=activation_fn[1:])
    return input_layer.with_tensor(y)
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
  if isinstance(kernel_spec, (int, long)):
    return [kernel_spec, kernel_spec]
  elif len(kernel_spec) == 1:
    return [kernel_spec[0], kernel_spec[0]]
  else:
    assert len(kernel_spec) == 2
    return kernel_spec


def _stride(stride_spec):
  """Expands the stride spec into a length 4 list.

  Args:
    stride_spec: None, an integer or a length 1, 2, or 4 sequence.
  Returns:
    A length 4 list.
  """
  if stride_spec is None:
    return [1, 1, 1, 1]
  elif isinstance(stride_spec, (int, long)):
    return [1, stride_spec, stride_spec, 1]
  elif len(stride_spec) == 1:
    return [1, stride_spec[0], stride_spec[0], 1]
  elif len(stride_spec) == 2:
    return [1, stride_spec[0], stride_spec[1], 1]
  else:
    assert len(stride_spec) == 4
    return stride_spec
