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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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


class SampledSoftmaxResult(
    collections.namedtuple('SoftmaxResult', ['logits', 'loss']),
    prettytensor.PrettyTensorTupleMixin):
  """Holds logits and a sampled cross entropy loss.

  This also provides binding and construction if the result contains a template.
  """
  pass


def _convert_and_assert_tensors_compatible(input_layer, target):
  target = tf.convert_to_tensor(target, dtype=input_layer.dtype)
  if not input_layer.get_shape().is_compatible_with(target.get_shape()):
    raise ValueError('target and input_layer are not compatible: %s != %s' %
                     (input_layer.get_shape(), target.get_shape()))
  return target


def _convert_and_assert_per_example_weights_compatible(
    input_layer, per_example_weights, dtype):
  """Converts per_example_weights to a tensor and validates the shape."""
  per_example_weights = tf.convert_to_tensor(
      per_example_weights, name='per_example_weights', dtype=dtype)
  if input_layer.get_shape().ndims:
    expected_length = input_layer.get_shape().dims[0]
    message = ('per_example_weights must have rank 1 and length %s, but was: %s'
               % (expected_length, per_example_weights.get_shape()))
  else:
    expected_length = None
    message = ('per_example_weights must have rank 1 and length equal to the '
               'first dimension of inputs (unknown), but was: %s'
               % per_example_weights.get_shape())

  if per_example_weights.get_shape().ndims not in (1, None):
    raise ValueError(message)

  if not per_example_weights.get_shape().is_compatible_with((expected_length,)):
    raise ValueError(message)

  return per_example_weights


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

  tensor = input_layer.tensor
  loss = regression_fn(tensor, target, *regression_args, **regression_kwargs)
  if loss_weight is not None:
    loss *= loss_weight
  if per_example_weights is not None:
    per_example_weights = _convert_and_assert_per_example_weights_compatible(
        input_layer,
        per_example_weights,
        dtype=loss.dtype)
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
  target = _convert_and_assert_tensors_compatible(input_layer, target)
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
  target = _convert_and_assert_tensors_compatible(input_layer, target)
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
  labels = _convert_and_assert_tensors_compatible(input_layer, labels)

  if per_example_weights is not None:
    per_example_weights = _convert_and_assert_per_example_weights_compatible(
        input_layer,
        per_example_weights,
        dtype=input_layer.dtype)

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
                                     per_example_weights=None,
                                     per_output_weights=None):
  """Calculates the binary cross entropy of the input_layer vs inputs.

  Expects unscaled logits. Do not pass in results of sigmoid operation.

  Args:
    input_layer: The input pre-sigmoid PrettyTensor.
    target: A Float or Double tensor containing class label probabilities. Note
      that binary cross entropy is equivalent to logistic loss.
    name: The optional name.
    loss_weight: A scalar multiplier for the loss.
    per_example_weights: A `Tensor` with a weight per example.
    per_output_weights: A weight `Tensor` that is the same shape as the
      input_layer that can be used to scale individual prediction losses.  See
      `tf.tile` to turn a per-column weight vector into a `per_output_weights`
      `Tensor`.
  Returns:
    Binary cross entropy loss after sigmoid operation.
  Raises:
    ValueError: if target is None or the type is not float or double.
  """
  if target is None:
    raise ValueError('target must be set')
  target = _convert_and_assert_tensors_compatible(input_layer, target)

  with tf.name_scope('stats'):
    selected, sum_retrieved, sum_relevant = _compute_precision_recall(
        input_layer, target, 0, per_example_weights)
    precision = selected / sum_retrieved
    recall = selected / sum_relevant
    if precision.get_shape().is_fully_defined():
      input_layer.bookkeeper.add_average_summary(
          precision, 'average_precision_%s' % name)
    if recall.get_shape().is_fully_defined():
      input_layer.bookkeeper.add_average_summary(
          recall, 'average_recall_%s' % name)
    input_layer.bookkeeper.add_scalar_summary(
        tf.reduce_sum(tf.to_float(tf.greater(input_layer, 0))), 'activations')
    if per_output_weights is not None:
      per_output_weights = tf.convert_to_tensor(
          per_output_weights,
          name='per_output_weights',
          dtype=input_layer.dtype.base_dtype)
      input_layer.get_shape().assert_is_compatible_with(
          per_output_weights.get_shape())

  def _batch_sum_bce(x, target, name='binary_cross_entropy'):
    logits = functions.binary_cross_entropy_loss_with_logits(x,
                                                             target,
                                                             name=name)
    if per_output_weights is not None:
      logits *= per_output_weights
    return functions.reduce_batch_sum(logits)

  return apply_regression(
      input_layer,
      _batch_sum_bce,
      target,
      [],
      name='%s_bce_loss' % name,
      loss_weight=loss_weight,
      per_example_weights=per_example_weights)


@prettytensor.RegisterCompoundOp
def softmax_classifier_with_sampled_loss(inputs,
                                         num_classes,
                                         labels,
                                         num_sampled,
                                         num_true=None,
                                         sampled_values=None,
                                         remove_accidental_hits=True,
                                         loss_weight=None,
                                         per_example_weights=None,
                                         weight_init=None,
                                         bias_init=tf.zeros_initializer,
                                         name='softmax_classifier'):
  """Applies softmax and if labels is not None, then it adds a sampled loss.

  This is a faster way to train a softmax classifier over a huge number of
  classes. It is generally an underestimate of the full softmax loss.

  At inference time, you can compute full softmax probabilities with the
  expression `tf.nn.softmax(tf.matmul(inputs, weights) + biases)`.

  See `tf.nn.sampled_softmax_loss` for more details.

  Also see Section 3 of [Jean et al., 2014](http://arxiv.org/abs/1412.2007)
  ([pdf](http://arxiv.org/pdf/1412.2007.pdf)) for the math.

  Note: If you depend on the softmax part of the loss, then you will lose most
  of the speed benefits of sampling the loss. It should be used for evaluation
  only and not executed on every update op.

  Note: This is not checkpoint compatible with `softmax_classifier` since it
  optimizes a transpose by pushing it down to the `fully_connected` layer.

  Args:
    inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
        activations of the input network.
    num_classes: An `int`. The number of possible classes.
    labels: A `Tensor` of type `int64` and shape `[batch_size,
        num_true]`. The target classes.  Note that this format differs from
        the `labels` argument of `nn.softmax_cross_entropy_with_logits`.
    num_sampled: An `int`.  The number of classes to randomly sample per batch.
    num_true: An `int`.  The number of target classes per training example,
      defaults to the second dim of labels if known or 1.
    sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
        `sampled_expected_count`) returned by a `*_candidate_sampler` function.
        (if None, we default to `log_uniform_candidate_sampler`)
    remove_accidental_hits:  A `bool`.  whether to remove "accidental hits"
        where a sampled class equals one of the target classes.  Default is
        True.
    loss_weight: A scalar multiplier for the loss.
    per_example_weights: A Tensor with a weight per example.
    weight_init: The initializer for the weights (see `fully_connected`). Note:
      This is the transpose of a normal fully_connected input layer!
    bias_init: The initializer for the bias (see `fully_connected`).
    name: The optional name.
  Returns:
    A tuple of handles to the logits (fully connected layer) and loss.
  Raises:
    ValueError: If inputs or labels do not have the right shape.
  """
  # Compound ops need to respect sequential, so take a snapshot.
  input_copy = inputs.as_layer()

  with tf.name_scope('sampled_softmax'):
    full = inputs.fully_connected(num_classes,
                                  activation_fn=None,
                                  name=name,
                                  transpose_weights=True,
                                  init=weight_init,
                                  bias_init=bias_init)
    if labels is not None:
      labels = tf.convert_to_tensor(labels, dtype=tf.int64, name='labels')
      labels.get_shape().assert_is_compatible_with([input_copy.get_shape()[0],
                                                    num_true])
      if num_true is None:
        if labels.get_shape().ndims and labels.get_shape().dims[1]:
          num_true = labels.get_shape().dims[1].value
        else:
          num_true = 1

      def _loss(input_, labels, name=None):
        return tf.nn.sampled_softmax_loss(
            full.layer_parameters['weights'],
            full.layer_parameters['bias'],
            input_,
            labels,
            num_sampled=num_sampled,
            num_classes=num_classes,
            num_true=num_true,
            sampled_values=sampled_values,
            remove_accidental_hits=remove_accidental_hits,
            name=name)

      loss = apply_regression(input_copy,
                              _loss,
                              labels,
                              [],
                              name='%s_sampled_loss' % name,
                              loss_weight=loss_weight,
                              per_example_weights=per_example_weights)
    else:
      loss = None

  return SampledSoftmaxResult(full, loss)


@prettytensor.RegisterCompoundOp
def softmax_classifier(
    input_layer, class_count, labels=None, name=PROVIDED, loss_weight=None,
    per_example_weights=None, weight_init=None, bias_init=tf.zeros_initializer):
  """Creates a fully-connected linear layer followed by a softmax.

  Args:
    input_layer: The chainable object, supplied.
    class_count: The number of classes.
    labels: The target labels to learn as a float tensor.  Use None to not
      include a training loss.
    name: The optional name.
    loss_weight: A scalar multiplier for the loss.
    per_example_weights: A Tensor with a weight per example.
    weight_init: The initializer for the weights (see `fully_connected`).
    bias_init: The initializer for the bias (see `fully_connected`).
  Returns:
    A tuple of the softmax's name and the loss tensor's name in m.bits.
  Raises:
    ValueError: If the datatype is wrong.
  """
  full = input_layer.fully_connected(class_count, activation_fn=None, name=name,
                                     init=weight_init, bias_init=bias_init)
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


@prettytensor.Register(assign_defaults=('phase',))
def evaluate_precision_recall(input_layer,
                              labels,
                              threshold=0.5,
                              per_example_weights=None,
                              name=PROVIDED,
                              phase=Phase.train):
  """Computes the precision and recall of the prediction vs the labels.

  Args:
    input_layer: A Pretty Tensor object.
    labels: The target labels to learn as a float tensor.
    threshold: The threshold to use to decide if the prediction is true.
    per_example_weights: A Tensor with a weight per example.
    name: An optional name.
    phase: The phase of this model; non training phases compute a total across
      all examples.
  Returns:
    Precision and Recall.
  """
  _ = name  # Eliminate warning, name used for namescoping by PT.
  selected, sum_retrieved, sum_relevant = _compute_precision_recall(
      input_layer, labels, threshold, per_example_weights)

  if phase != Phase.train:
    dtype = tf.float32
    # Create the variables in all cases so that the load logic is easier.
    relevant_count = tf.get_variable(
        'relevant_count',
        [],
        dtype,
        tf.zeros_initializer,
        collections=[bookkeeper.GraphKeys.TEST_VARIABLES],
        trainable=False)
    retrieved_count = tf.get_variable(
        'retrieved_count',
        [],
        dtype,
        tf.zeros_initializer,
        collections=[bookkeeper.GraphKeys.TEST_VARIABLES],
        trainable=False)
    selected_count = tf.get_variable(
        'selected_count',
        [],
        dtype,
        tf.zeros_initializer,
        collections=[bookkeeper.GraphKeys.TEST_VARIABLES],
        trainable=False)

    with input_layer.g.device(selected_count.device):
      selected = tf.assign_add(selected_count, selected)
    with input_layer.g.device(retrieved_count.device):
      sum_retrieved = tf.assign_add(retrieved_count, sum_retrieved)
    with input_layer.g.device(relevant_count.device):
      sum_relevant = tf.assign_add(relevant_count, sum_relevant)

  return (tf.select(tf.equal(sum_retrieved, 0),
                    tf.zeros_like(selected),
                    selected/sum_retrieved),
          tf.select(tf.equal(sum_relevant, 0),
                    tf.zeros_like(selected),
                    selected/sum_relevant))


@prettytensor.Register(assign_defaults=('phase',))
def evaluate_classifier(input_layer, labels, per_example_weights=None,
                        topk=1, name=PROVIDED, phase=Phase.train):
  """Calculates the total ratio of correct predictions across all examples seen.

  In test and infer mode, this creates variables in the graph collection
  pt.GraphKeys.TEST_VARIABLES and does not add them to
  tf.GraphKeys.ALL_VARIABLES.  This means that you must initialize them
  separately from tf.initialize_all_variables().

  In the case of `topk == 1`, this breaks ties left-to-right, in all other cases
  it follows `tf.nn.in_top_k`. *Note*: the tie behavior will change in the
  future.

  Args:
    input_layer: The input_layer.
    labels: A float or double tensor containing the target for this layer.
    per_example_weights: Weights that are applied to every example.
    topk: Integer k for 'accuracy at top k' metric.
    name: The name of this layer.
    phase: In training mode the batch accuracy is returned and in eval/infer
      modes a total average is calculated.
  Returns:
    A Pretty Tensor with the ratio of correct to total examples seen.
  """
  correct_predictions, examples = _compute_average_correct(
      input_layer, labels, per_example_weights, topk=topk)
  parameters = {}
  if phase != Phase.train:
    dtype = tf.float32
    # Create the variables using tf.Variable because we don't want to share.
    count = tf.Variable(tf.constant(0, dtype=dtype),
                        name='count_%d' % topk,
                        collections=[bookkeeper.GraphKeys.TEST_VARIABLES],
                        trainable=False)
    correct = tf.Variable(tf.constant(0, dtype=dtype),
                          name='correct_%d' % topk,
                          collections=[bookkeeper.GraphKeys.TEST_VARIABLES],
                          trainable=False)
    parameters['count'] = count
    parameters['correct'] = correct
    with input_layer.g.device(count.device):
      examples = tf.assign_add(count, examples)
    with input_layer.g.device(correct.device):
      correct_predictions = tf.assign_add(correct, correct_predictions)
  return input_layer.with_tensor(
      tf.div(correct_predictions, examples, name=name), parameters)


def _compute_precision_recall(input_layer, labels, threshold,
                              per_example_weights):
  """Returns the numerator of both, the denominator of precision and recall."""

  # To apply per_example_weights, we need to collapse each row to a scalar, but
  # we really want the sum.
  labels.get_shape().assert_is_compatible_with(input_layer.get_shape())
  relevant = tf.to_float(tf.greater(labels, 0))
  retrieved = tf.to_float(tf.greater(input_layer, threshold))
  selected = relevant * retrieved

  if per_example_weights is not None:
    per_example_weights = _convert_and_assert_per_example_weights_compatible(
        input_layer,
        per_example_weights,
        dtype=None)
    per_example_weights = tf.to_float(tf.greater(per_example_weights, 0))
    selected = functions.reduce_batch_sum(selected) * per_example_weights
    relevant = functions.reduce_batch_sum(relevant) * per_example_weights
    retrieved = functions.reduce_batch_sum(retrieved) * per_example_weights
  sum_relevant = tf.reduce_sum(relevant)
  sum_retrieved = tf.reduce_sum(retrieved)
  selected = tf.reduce_sum(selected)
  return selected, sum_retrieved, sum_relevant


def _compute_average_correct(input_layer, labels, per_example_weights, topk=1):
  """Returns the numerator and denominator of classifier accuracy."""
  dtype = tf.float32
  if topk == 1:
    true_labels = tf.argmax(input_layer, 1)
    predictions = tf.argmax(labels, 1)
    in_topk = tf.equal(true_labels, predictions)
  else:
    _, true_labels = tf.nn.top_k(labels, k=1)
    true_labels = tf.reshape(true_labels, [-1])
    in_topk = tf.nn.in_top_k(tf.cast(input_layer, dtype), true_labels, k=topk)
  correct_predictions = tf.cast(in_topk, dtype)

  # If individual examples are weighted, then we want to normalize by that.
  if per_example_weights is not None:
    per_example_weights = _convert_and_assert_per_example_weights_compatible(
        input_layer,
        per_example_weights,
        dtype=None)
    float_weights = tf.cast(per_example_weights, dtype)
    # TODO(eiderman): This should use an op that doesn't support broadcasting.
    correct_predictions *= float_weights
    num_examples = tf.reduce_sum(float_weights)
  else:
    # shape only holds ints, but we want to always return the same type
    # for num_examples to make everything compatible.
    num_examples = tf.cast(tf.gather(tf.shape(input_layer), 0), dtype)
  return tf.reduce_sum(correct_predictions), num_examples
