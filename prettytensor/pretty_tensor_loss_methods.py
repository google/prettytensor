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
"""Default regression functions for PrettyTensor."""
import collections
import tensorflow as tf

from prettytensor import functions
from prettytensor import bookkeeper
from prettytensor import pretty_tensor_class as prettytensor
from prettytensor.pretty_tensor_class import Phase
from prettytensor.pretty_tensor_class import PROVIDED


class SoftmaxResult(
    collections.namedtuple('SoftmaxResult', ['softmax', 'loss']),
    prettytensor.PrettyTensorTupleMixin):
  """Holds a softmax activation and a cross entropy loss.

  This also provides binding and construction if the result contains a template.
  """
  pass


def apply_regression(input_layer,
                     regression_fn,
                     target,
                     regression_args=(),
                     regression_kwargs=None,
                     name=PROVIDED,
                     loss_weight=None,
                     per_example_weights=None):
  """Applies the given regression and adds the loss to the bookkeeper.

  This does not change tensor.
  Args:
    input_layer: The chainable object.
    regression_fn: A function that takes (in order) tensor, labels.
    target: The targe of the regression.
    regression_args: Other arguments for the regression.
    regression_kwargs: Keyword args for the regression.
    name: The name, also added to regression_kwargs.
    loss_weight: A scalar multiplier for the loss.
    per_example_weights: A Tensor with a weight per example.
  Returns:
    The loss tensor's name.
  Raises:
    ValueError: If the target is not a compatible shape with input_layer.
  """
  if regression_kwargs is None:
    regression_kwargs = {}
  if name is not None and 'name' not in regression_kwargs:
    regression_kwargs['name'] = name
  elif name is None:
    name = input_layer.tensor.op.name

  target = tf.convert_to_tensor(target, dtype=input_layer.dtype)
  if not input_layer.get_shape().is_compatible_with(target.get_shape()):
    raise ValueError('target and input_layer are not compatible: %s != %s' %
                     (input_layer.get_shape(), target.get_shape()))
  tensor = input_layer.tensor
  loss = regression_fn(tensor, target, *regression_args, **regression_kwargs)
  if loss_weight is not None:
    loss *= loss_weight
  if per_example_weights is not None:
    loss *= per_example_weights
  # Use mean so that the learning rate is independent of the batch size.
  if name is None:
    name = loss.op.name
  if tensor.get_shape()[0].value is not None:
    # Try to use division instead of reduce_mean because reduce_mean doesn't
    # work on GPU.
    avg_loss = tf.reduce_sum(loss) / tensor.get_shape()[0].value
  else:
    avg_loss = tf.reduce_mean(loss)
  return input_layer.add_loss(avg_loss, name=name)


@prettytensor.Register
def l2_regression(
    input_layer, target, name=PROVIDED, loss_weight=None,
    per_example_weights=None):
  """Applies an L2 Regression (Sum of Squared Error) to the target."""
  if target.dtype not in (tf.float32, tf.float64):
    raise ValueError('Unexpected type for target:  %s' % target.dtype)
  return apply_regression(input_layer,
                          functions.l2_regression_sq_loss,
                          target,
                          [],
                          name='%s_loss' % name,
                          loss_weight=loss_weight,
                          per_example_weights=per_example_weights)


@prettytensor.Register
def l1_regression(
    input_layer, target, name=PROVIDED, loss_weight=None,
    per_example_weights=None):
  """Applies an L1 Regression (Sum of Absolute Error) to the target."""
  if target.dtype not in (tf.float32, tf.float64):
    raise ValueError('Unexpected type for target: %s' % target.dtype)
  return apply_regression(input_layer,
                          functions.l1_regression_loss,
                          target,
                          [],
                          name='%s_loss' % name,
                          loss_weight=loss_weight,
                          per_example_weights=per_example_weights)


@prettytensor.Register
def softmax_activation(input_layer):
  """Computes the softmax.

  Args:
    input_layer: The input PrettyTensor.
  Returns:
    A new Pretty Tensor with the softmax applied.
  """
  return input_layer.with_tensor(tf.nn.softmax(input_layer))


@prettytensor.Register
def cross_entropy(input_layer,
                  labels,
                  name=PROVIDED,
                  loss_weight=None,
                  per_example_weights=None):
  """Calculates the Cross Entropy of the input_layer vs inputs.

  Args:
    input_layer: The input PrettyTensor.
    labels: A Float or Double tensor containing the labels.
    name: The optional name.
    loss_weight: A weight to scale the loss. Used when there are multiple
      losses.
    per_example_weights: A weighting for each example.
  Returns:
    A loss.
  Raises:
    ValueError: if labels is None or the type is not float or double.
  """
  if labels is None:
    raise ValueError('Labels must be set')
  labels = tf.convert_to_tensor(labels, name='labels')
  if labels.dtype not in (tf.float32, tf.float64):
    raise ValueError('Unexpected type for target:  %s' % labels.dtype)
  correct_predictions, examples = _compute_average_correct(
      input_layer, labels, per_example_weights)
  correct_ratio = correct_predictions / examples
  if correct_ratio.get_shape().is_fully_defined():
    input_layer.bookkeeper.add_average_summary(
        correct_ratio, 'average_accuracy_%s' % name)
  return apply_regression(input_layer,
                          tf.nn.softmax_cross_entropy_with_logits,
                          labels,
                          [],
                          name='%s_loss' % name,
                          loss_weight=loss_weight,
                          per_example_weights=per_example_weights)


@prettytensor.Register
def binary_cross_entropy_with_logits(input_layer,
                                     target,
                                     name=PROVIDED,
                                     loss_weight=None,
                                     per_example_weights=None):
  """Calculates the binary cross entropy of the input_layer vs inputs.

  Expects unscaled logits. Do not pass in results of sigmoid operation.

  Args:
    input_layer: The input pre-sigmoid PrettyTensor.
    target: A Float or Double tensor containing class label probabilities. Note
      that binary cross entropy is equivalent to logistic loss.
    name: The optional name.
    loss_weight: A scalar multiplier for the loss.
    per_example_weights: A Tensor with a weight per example.
  Returns:
    Binary cross entropy loss after sigmoid operation.
  Raises:
    ValueError: if target is None or the type is not float or double.
  """
  if target is None:
    raise ValueError('Labels must be set')
  if target.dtype not in (tf.float32, tf.float64):
    raise ValueError('Unexpected type for target:  %s' % target.dtype)
  return apply_regression(input_layer,
                          functions.binary_cross_entropy_loss_with_logits,
                          target,
                          [],
                          name='%s_bce_loss' % name,
                          loss_weight=loss_weight,
                          per_example_weights=per_example_weights)


@prettytensor.RegisterCompoundOp
def softmax_classifier(
    input_layer, class_count, labels=None, name=PROVIDED, loss_weight=None,
    per_example_weights=None):
  """Creates a fully-connected linear layer followed by a softmax.

  Args:
    input_layer: The chainable object, supplied.
    class_count: The number of classes.
    labels: The target labels to learn as a float tensor.  Use None to not
      include a training loss.
    name: The optional name.
    loss_weight: A scalar multiplier for the loss.
    per_example_weights: A Tensor with a weight per example.
  Returns:
    A tuple of the softmax's name and the loss tensor's name in m.bits.
  Raises:
    ValueError: If the datatype is wrong.
  """
  full = input_layer.fully_connected(class_count, activation_fn=None, name=name)
  return full.softmax(labels=labels,
                      loss_weight=loss_weight,
                      per_example_weights=per_example_weights,
                      name=name)


@prettytensor.RegisterCompoundOp
def softmax(input_layer,
            labels=None,
            name=PROVIDED,
            loss_weight=None,
            per_example_weights=None):
  """Applies softmax and if labels is not None, then it also adds a loss.

  Args:
    input_layer: The chainable object, supplied.
    labels: The target labels to learn as a float tensor.  Use None to not
      include a training loss.
    name: The optional name.
    loss_weight: A scalar multiplier for the loss.
    per_example_weights: A Tensor with a weight per example.
  Returns:
    A tuple of the a handle to softmax and a handle to the loss tensor.
  Raises:
    ValueError: If the datatype is wrong.
  """
  if labels is not None:
    # Cache the current layer because we only want softmax to change the head.
    full = input_layer.as_layer()
    return SoftmaxResult(input_layer.softmax_activation(),
                         full.cross_entropy(
                             labels,
                             name=name,
                             loss_weight=loss_weight,
                             per_example_weights=per_example_weights))
  else:
    return SoftmaxResult(input_layer.softmax_activation(), None)


@prettytensor.Register
def evaluate_classifier(input_layer, labels, per_example_weights=None,
                        topk=1, name=PROVIDED, phase=Phase.train):
  """Calculates the total ratio of correct predictions across all examples seen.

  In test and infer mode, this creates variables in the graph collection
  pt.GraphKeys.TEST_VARIABLES and does not add them to
  tf.GraphKeys.ALL_VARIABLES.  This means that you must initialize them
  separately from tf.initialize_all_variables().

  Args:
    input_layer: The input_layer.
    labels: A float or double tensor containing the target for this layer.
    per_example_weights: Weights that are applied to every example.
    topk: Integer k for 'accuracy at top k' metric.
    name: The name of this layer.
    phase: In training mode the batch accuracy is returned and in eval/infer
      modes a total average is calculated.
  Returns:
    A Pretty Tensor with the ratio of correct to total examples seen..
  """
  correct_predictions, examples = _compute_average_correct(
      input_layer, labels, per_example_weights, topk=topk)
  if phase != Phase.train:
    dtype = tf.float32
    # Create the variables in all cases so that the load logic is easier.
    count = tf.get_variable('count_%d' % topk, [], dtype, tf.zeros_initializer,
                            collections=[bookkeeper.GraphKeys.TEST_VARIABLES],
                            trainable=False)
    correct = tf.get_variable('correct_%d' % topk, [], dtype,
                              tf.zeros_initializer,
                              collections=[bookkeeper.GraphKeys.TEST_VARIABLES],
                              trainable=False)
    with input_layer.g.device(count.device):
      examples = tf.assign_add(count, examples)
    with input_layer.g.device(correct.device):
      correct_predictions = tf.assign_add(correct, correct_predictions)
  return input_layer.with_tensor(
      tf.div(correct_predictions, examples, name=name))


def _compute_average_correct(input_layer, labels, per_example_weights, topk=1):
  """Returns the numerator and denominator of classifier accuracy."""
  dtype = tf.float32
  _, true_labels = tf.nn.top_k(labels, k=1)
  true_labels = tf.reshape(true_labels, [-1])
  in_topk = tf.nn.in_top_k(tf.cast(input_layer, dtype), true_labels, k=topk)
  correct_predictions = tf.cast(in_topk, dtype)

  # If individual examples are weighted, then we want to normalize by that.
  if per_example_weights:
    per_example_weights = tf.convert_to_tensor(per_example_weights,
                                               name='per_example_weights')
    if ((input_layer.get_shape() and not per_example_weights.get_shape(
    ).is_compatible_with([input_layer.get_shape().dims[0]])) or
        per_example_weights.get_shape().ndims != 1):
      raise ValueError(
          'per_example_weights must be a vector of the same length as '
          'labels: %s' % per_example_weights.get_shape())
    float_weights = tf.cast(per_example_weights, dtype)
    # TODO(eiderman): This should use an op that doesn't support broadcasting.
    correct_predictions *= float_weights
    num_examples = tf.reduce_sum(float_weights)
  else:
    # shape only holds ints, but we want to always return the same type
    # for num_examples to make everything compatible.
    num_examples = tf.cast(tf.gather(tf.shape(input_layer), 0), dtype)
  return tf.reduce_sum(correct_predictions), num_examples
