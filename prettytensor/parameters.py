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
r"""Provides some standard functions to modify parameter variables.

These are applied before the variables are used within the graph; the basic
function signature is:

    def my_func(var_name, variable, phase):
      \"""A function to apply to a model's parameters.

      Args:
        var_name: The short name for the variable.
        variable: A `Tensor` that can be used in the model. Note: this is often
          a tf.Variable, but may not be and the name only usually contains
          var_name (it doesn't in the case of reuse).
        phase: The phase of model construction.

      Returns:
        A `Variable` or `Tensor` with the same shape and type as `variable` to
        use.
      \"""
      return something_done_to_variable
"""

import re


import tensorflow as tf

from prettytensor import pretty_tensor_class as pt


def identity(unused_var_name, variable, unused_phase):
  return variable


def regularizer(name, regularization_fn, name_filter='weights'):
  """Wraps a regularizer in a parameter-function.

  Args:
    name: The name scope for this regularizer.
    regularization_fn: A function with signature:
      fn(variable) -> loss `Tensor` or `None`.
    name_filter: A regex that will be used to filter variables by name.

  Returns:
    A parameter modification function that adds the loss to the
    REGULARIZATION_LOSSES graph key.
  """
  regex = re.compile(name_filter)
  def fn(var_name, variable, phase):
    if phase is pt.Phase.train and regex.search(var_name):
      with tf.name_scope(None, name, [variable]):
        loss = regularization_fn(variable)
      if loss is not None:
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, loss)
    return variable
  return fn


def l2_regularizer(decay, name_filter='weights'):
  """Create an l2 regularizer."""
  return regularizer(
      'l2_regularizer',
      lambda x: tf.nn.l2_loss(x) * decay,
      name_filter=name_filter)


def l1_regularizer(decay, name_filter='weights'):
  """Create an l1 regularizer."""
  return regularizer(
      'l1_regularizer',
      lambda x: tf.reduce_sum(tf.abs(x)) *  decay,
      name_filter=name_filter)


def compose(*parameter_functions):
  """Composes multiple modification functions in order.

  Args:
    *parameter_functions: The functions to compose.

  Returns:
    A parameter modification function that consists of applying all the provided
    functions.
  """
  def composed_fn(var_name, variable, phase):
    for fn in parameter_functions:
      variable = fn(var_name, variable, phase)
    return variable
  return composed_fn


class Noise(object):
  """Regularize the model by applying gaussian noise to the variables."""

  def __init__(self, stddev):
    self._stddev = stddev

  def __call__(self, var_name, variable, phase):
    if phase is pt.Phase.train:
      return variable * tf.random_normal(
          tf.shape(variable), mean=1., stddev=self._stddev)
    else:
      return variable


class DropConnect(object):
  """Drop out some connections.

  See the paper: http://www.matthewzeiler.com/pubs/icml2013/icml2013.pdf
  """

  def __init__(self, keep_prob):
    self._keep_prob = keep_prob

  def __call__(self, var_name, variable, phase):
    if 'bias' not in var_name or phase is not pt.Phase.train:
      return variable
    return tf.nn.dropout(variable, self._keep_prob)
