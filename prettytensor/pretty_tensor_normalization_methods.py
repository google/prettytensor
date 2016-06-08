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
"""Batch Normalization and eventually some friends for PrettyTensor."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf

from prettytensor import pretty_tensor_class as prettytensor
from prettytensor.pretty_tensor_class import Phase
from prettytensor.pretty_tensor_class import PROVIDED


BatchNormalizationArguments = collections.namedtuple(
    'BatchNormalizationArguments',
    ('learned_moments_update_rate', 'variance_epsilon',
     'scale_after_normalization'))

BatchNormalizationArguments.__new__.__defaults__ = (None, None, None)


def batch_normalize_with_arguments(x, arguments):
  """Applies batch normalization to x as specified in arguments.

  Args:
    x: A Pretty Tensor.
    arguments: Either a boolean to batch_normalize or a
      BatchNormalizationArguments

  Returns:
    x with batch normalization applied.
  """
  x = prettytensor.wrap(x)
  # Backwards compatibility.
  if isinstance(arguments, bool):
    if arguments:
      return x.batch_normalize()
    else:
      return x

  # pylint: disable=protected-access
  kwargs = arguments._asdict()
  defaults = prettytensor._defaults
  # pylint: enable=protected-access
  for arg in ('learned_moments_update_rate', 'variance_epsilon',
              'scale_after_normalization'):
    if kwargs.get(arg, None) is None:
      if arg in defaults:
        kwargs[arg] = defaults[arg]
      else:
        del kwargs
  return x.batch_normalize(**kwargs)


# pylint: disable=invalid-name
@prettytensor.Register(
    assign_defaults=('learned_moments_update_rate', 'variance_epsilon',
                     'scale_after_normalization', 'phase'))
class batch_normalize(prettytensor.VarStoreMethod):

  def __call__(self,
               input_layer,
               name=PROVIDED,
               learned_moments_update_rate=0.0003,
               variance_epsilon=0.001,
               scale_after_normalization=False,
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
    # Allocate variables to hold the moving averages of the moments.
    params_shape = [input_layer.shape[-1]]

    # Allocate parameters for the beta and gamma of the normalization.
    beta = self.variable('beta', params_shape, tf.constant_initializer(0.0))
    if scale_after_normalization:
      gamma = self.variable('gamma', params_shape, tf.constant_initializer(1.0))
    else:
      gamma = None
    moving_mean = self.variable('moving_mean',
                                params_shape,
                                tf.constant_initializer(0.0),
                                train=False)
    moving_variance = self.variable('moving_variance',
                                    params_shape,
                                    tf.constant_initializer(1.0),
                                    train=False)

    if phase == Phase.train:
      # Calculate the moments based on the individual batch.
      mean, variance = tf.nn.moments(
          input_layer.tensor, list(range(len(input_layer.get_shape()) - 1)))
      input_layer.bookkeeper.add_histogram_summary(mean)
      input_layer.bookkeeper.add_histogram_summary(variance)

      avg_mean = input_layer.bookkeeper.exponential_moving_average(
          mean, moving_mean, 1.0 - learned_moments_update_rate)
      avg_variance = input_layer.bookkeeper.exponential_moving_average(
          variance, moving_variance, 1.0 - learned_moments_update_rate)
      with tf.control_dependencies([avg_variance, avg_mean]):
        y = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma,
                                      variance_epsilon)
    else:
      # Load the mean and variance as the 'moving average' moments
      # from the checkpoint.
      y = tf.nn.batch_normalization(input_layer, moving_mean, moving_variance,
                                    beta, gamma, variance_epsilon)

    return input_layer.with_tensor(y)
# pylint: enable=invalid-name
