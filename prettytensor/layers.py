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
"""Utility functions for adding layers to a Model.

NB: This is used by PrettyTensor, but it will be deprecated.  Please do not use!
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

import tensorflow as tf

from prettytensor import functions

# Implementation note: this takes a tuple for an activation instead of
# encouraging lambdas so that we can inspect the actual function and add
# appropriate summaries.


def apply_activation(
    books,
    x,
    activation,
    activation_args=(),
    activation_kwargs=None):
  """Returns activation(x, *activation_args, **activation_kwargs).

  This applies the given activation and adds useful summaries specific to the
  activation.

  Args:
    books: The bookkeeper.
    x: The tensor to apply activation to.
    activation: An activation function.
    activation_args: Optional additional arguments for the activation.
    activation_kwargs: Optional keyword args for activation.
  Returns:
    A tensor with activation applied to x.
  """
  if activation is None:
    return x
  if activation_kwargs is None:
    activation_kwargs = {}
  y = activation(x, *activation_args, **activation_kwargs)
  if activation in (tf.nn.relu, functions.leaky_relu, functions.softplus):
    books.add_scalar_summary(
        tf.reduce_mean(tf.cast(tf.less(x, 0.0), tf.float32)),
        '%s/zeros' % y.op.name)
  elif activation is tf.nn.relu6:
    books.add_scalar_summary(
        tf.reduce_mean(tf.cast(tf.less(x, 0.0), tf.float32)),
        '%s/zeros' % y.op.name)
    books.add_scalar_summary(
        tf.reduce_mean(tf.cast(tf.greater(x, 6.0), tf.float32)),
        '%s/sixes' % y.op.name)
  elif activation in (functions.l2_normalize, tf.nn.l2_normalize,
                      functions.l1_normalize):
    books.add_scalar_summary(
        tf.reduce_mean(tf.sqrt(tf.reduce_sum(
            tf.square(x), 1))), '%s/length' % y.op.name)
  return y


def add_l2loss(books, params, l2loss, name='weight_decay'):
  if l2loss:
    books.add_loss(
        tf.multiply(
            tf.nn.l2_loss(params), l2loss, name=name),
        regularization=True,
        add_summaries=False)


def he_init(n_inputs, n_outputs, activation_fn, uniform=True):
  """Sets the parameter initialization using the method described.

  This method is designed to keep the scale of the gradients roughly the same
  in all layers with ReLU activations.

  He et al. (2015):
           Delving deep into rectifiers: surpassing human-level performance on
           imageNet classification. International Conference on Computer Vision.

  For activations other than ReLU and ReLU6, this method uses Xavier
  initialization as in xavier_init().

  Args:
    n_inputs: The number of input nodes into each output.
    n_outputs: The number of output nodes for each input.
    activation_fn: Activation function used in this layer.
    uniform: If uniform distribution will be used for Xavier initialization.
             Normal distribution will be used if False.
  Returns:
    An initializer.
  """
  def in_relu_family(activation_fn):
    if isinstance(activation_fn, collections.Sequence):
      activation_fn = activation_fn[0]
    return activation_fn in (tf.nn.relu, tf.nn.relu6)

  if in_relu_family(activation_fn):
    stddev = math.sqrt(2.0 / n_inputs)
    # TODO(): Evaluates truncated_normal_initializer.
    return tf.random_normal_initializer(stddev=stddev)
  else:
    return xavier_init(n_inputs, n_outputs, uniform)


def xavier_init(n_inputs, n_outputs, uniform=True):
  """Set the parameter initialization using the method described.

  This method is designed to keep the scale of the gradients roughly the same
  in all layers.

  Xavier Glorot and Yoshua Bengio (2010):
           Understanding the difficulty of training deep feedforward neural
           networks. International conference on artificial intelligence and
           statistics.
  Args:
    n_inputs: The number of input nodes into each output.
    n_outputs: The number of output nodes for each input.
    uniform: If true use a uniform distribution, otherwise use a normal.
  Returns:
    An initializer.
  """
  if uniform:
    # 6 was used in the paper.
    init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
    return tf.random_uniform_initializer(-init_range, init_range)
  else:
    # 3 gives us approximately the same limits as above since this repicks
    # values greater than 2 standard deviations from the mean.
    stddev = math.sqrt(3.0 / (n_inputs + n_outputs))
    return tf.truncated_normal_initializer(stddev=stddev)


def spatial_slice_zeros(x):
  """Experimental summary that shows how many planes are unused for a batch."""
  return tf.cast(tf.reduce_all(tf.less_equal(x, 0.0), [0, 1, 2]),
                 tf.float32)
