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
"""Methods for Pretty Tensor."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools

import tensorflow as tf

from prettytensor import functions
from prettytensor import layers
from prettytensor import parameters
from prettytensor import pretty_tensor_class as prettytensor
from prettytensor.pretty_tensor_class import DIM_REST
from prettytensor.pretty_tensor_class import DIM_SAME
from prettytensor.pretty_tensor_class import Phase
from prettytensor.pretty_tensor_class import PROVIDED


# TODO(eiderman): many of these functions are different depending on if the
# underlying PT holds a sequence or a tensor. The underlying PT should be
# separate types so that the dispatch is handled by type instead of if
# statement.


def _infer_unknown_dims(old_shape, shape_spec):
  """Attempts to replace DIM_REST (if present) with a value.

  Because of `pt.DIM_SAME`, this has more information to compute a shape value
  than the default reshape's shape function.

  Args:
    old_shape: The current shape of the Tensor as a list.
    shape_spec: A shape spec, see `pt.reshape`.
  Returns:
    A list derived from `shape_spec` with `pt.DIM_SAME` replaced by the value
    from old_shape (if possible) and `pt.DIM_REST` computed (if possible).
  Raises:
    ValueError: If there are two many unknown dimensions or the shape_spec
    requires out of range DIM_SAME.
    TypeError: If shape_spec if not iterable.
  """

  # To compute the dimension of an unknown, we need to track which of the values
  # from old_shape are not copied for the numerator and any values specified as
  # integers for the denominator.
  #
  # After the loop, if any input dimension is unknown and not DIM_SAME, the
  # numerator will be 0. Otherwise it is the product of all non-DIM_SAME
  # dimensions.  This means that the dimension of DIM_REST is
  # numerator / denominator
  numerator_elements = [x if x else 0 for x in old_shape]
  denominator = 1
  unknowns = 0

  normalized_shape_spec = []
  for s in shape_spec:
    # Equality of tf.Dimension is broken and upstream fix hasn't been accepted.
    if isinstance(s, tf.Dimension):
      normalized_shape_spec.append(s.value)
    elif isinstance(s, tf.TensorShape):
      for dim in s:
        normalized_shape_spec.append(dim.value)
    else:
      normalized_shape_spec.append(s)
  result = []
  for i, s in enumerate(normalized_shape_spec):
    if s == DIM_SAME:
      if i >= len(old_shape):
        raise ValueError('%d exceeds the input shape' % i)
      if old_shape[i] is None:
        result.append(DIM_SAME)
      else:
        result.append(old_shape[i])
      numerator_elements[i] = 1
    elif s in (DIM_REST, -1, None):
      result.append(-1)
      unknowns += 1
    else:
      x = int(s)
      result.append(x)
      denominator *= x

  numerator = 1
  for x in numerator_elements:
    numerator *= x
  if unknowns > 1:
    raise ValueError('Only one unknown value (-1 or *) is allowed: %s' %
                     shape_spec)
  elif numerator % denominator != 0:
    raise ValueError('Input (%s) cannot be reshaped to %s.' %
                     (old_shape, shape_spec))
  elif unknowns == 0 and numerator > 0 and numerator != denominator:
    raise ValueError('Input (%s) cannot be reshaped to %s.' %
                     (old_shape, shape_spec))
  if numerator and unknowns:
    unknown_elements = int(numerator / denominator)
    return [unknown_elements if x == -1 else x for x in result]
  else:
    return result


@prettytensor.Register
def reshape(input_layer, shape_spec):
  """Reshapes this tensor to the given spec.

  This provides additional functionality over the basic `tf.reshape`. In
  particular, it provides the ability to specify some dimensions as unchanged
  (`pt.DIM_SAME`) which can greatly aid in inferring the extra dimensions
  (`pt.DIM_REST`) and help maintain more shape information going forward.

  A shape_spec can be a list or tuple of numbers specifying the new shape, but
  also may include the following shorthands for using values from the shape of
  the input:

  1. `pt.DIM_SAME` ('_') will use the corresponding value from the current
      shape.
  2. One -1 or `pt.DIM_REST` ('*') can be used to specify the remainder of the
      values.
  3. An integer will be used as is.

  A compact syntax is also supported for setting shapes. If the new shape is
  only composed of DIM_SAME, DIM_REST/-1 and single digit integers, then a
  string can be passed in. Integers larger than 9 must be passed in as part of a
  sequence.

  1. Flatten to a batch dimension (first by convention): [DIM_SAME, -1] or '_*'.
  2. Expand a Rank 2 Tensor so that it can be used as an image: '_11*'.
  The primary difference between this and `tf.reshape` is that `DIM_SAME` allows
  more shape inference possibilities. For example: given a shape of
  **[None, 3, 7]** if flattening were desired then the caller would have to
  compute the shape and request a reshape of **[-1, 21]** to flatten. Instead of
  brittle or repeated code, this can be inferred if we know that the first dim
  is being copied.

  Another example that is impossible to express as a list of integers is if the
  starting shape were **[None, 3, None]** and we wanted to do the same
  flattening. While the shape cannot be inferred, this can still be expressed as
  '_*' (A.K.A. [DIM_SAME, DIM_REST]).

  Args:
    input_layer: The Pretty Tensor object, supplied.
    shape_spec: The spec for the new shape.
  Returns:
    A Pretty Tensor with the reshaped tensor.
  Raises:
    ValueError: If there are two many unknown dimensions or the shape_spec
    requires out of range DIM_SAME.
  """
  old_shape = input_layer.get_shape().as_list()

  # Extract both a tensor that sets the new shape and as much of the new
  # shape is known. This lets us merge in any extra information we have about
  # the shape.
  try:
    new_shape = _infer_unknown_dims(old_shape, shape_spec)
  except TypeError:
    # shape_spec is not iterable, it is probably a tensor or variable.
    return tf.reshape(input_layer, shape_spec)
  reshape_tensor = []

  # To avoid bloating the graph, we want to capture consecutive integers into
  # a single tf.constant. This allows us to eliminate tf.concat when we know the
  # shape.
  runner = []

  for i, s in enumerate(new_shape):
    if s is DIM_SAME:
      new_shape[i] = None
      if runner:
        reshape_tensor.append(tf.constant(runner))
        runner = []
      # Since we can't statically infer the value, compute it from the graph.
      reshape_tensor.append(tf.gather(tf.shape(input_layer), [i]))
    else:
      runner.append(s)
      if s == -1:
        new_shape[i] = None
  if runner:
    reshape_tensor.append(tf.constant(runner))

  if len(reshape_tensor) == 1:
    reshape_tensor = reshape_tensor[0]
  else:
    reshape_tensor = tf.concat(0, reshape_tensor)
  result = tf.reshape(input_layer, reshape_tensor)
  result.set_shape(new_shape)

  return input_layer.with_tensor(result)


@prettytensor.Register
def flatten(input_layer, preserve_batch=True):
  """Flattens this.

  If preserve_batch is True, the result is rank 2 and the first dim (batch) is
  unchanged. Otherwise the result is rank 1.

  Args:
    input_layer: The Pretty Tensor object, supplied.
    preserve_batch: If True (the default), then preserve the first dimension.
  Returns:
    A LayerWrapper with the flattened tensor.
  """
  if preserve_batch:
    return reshape(input_layer, [DIM_SAME, -1])
  else:
    return reshape(input_layer, [-1])


@prettytensor.Register
def stop_gradient(input_layer):
  """Cuts off the gradient at this point.

  This works on both sequence and regular Pretty Tensors.

  Args:
    input_layer: The input.
  Returns:
    A new Pretty Tensor of the same type with stop_gradient applied.
  """
  if input_layer.is_sequence():
    result = [tf.stop_gradient(t) for t in input_layer.sequence]
    return input_layer.with_sequence(result)
  else:
    return tf.stop_gradient(input_layer)


@prettytensor.Register(assign_defaults='phase')
def dropout(input_layer, keep_prob, phase=Phase.train, name=PROVIDED):
  """Aplies dropout if this is in the train phase."""
  if phase == Phase.train:
    return tf.nn.dropout(input_layer, keep_prob, name=name)
  else:
    return input_layer


# TODO(eiderman): Give a good name for this function: Maybe InnerProductIsh ?
# pylint: disable=invalid-name
@prettytensor.Register(
    assign_defaults=('l2loss', 'parameter_modifier', 'phase'))
class diagonal_matrix_mul(prettytensor.VarStoreMethod):
  """Diagonal Matrix Multiplication."""

  def __call__(self,
               input_layer,
               weights=None,
               l2loss=None,
               phase=prettytensor.Phase.train,
               parameter_modifier=parameters.identity):
    """Performs a diagonal matrix multiplication with a learned vector.

    This creates the parameter vector.

    Args:
      input_layer: The input_layer.
      weights:  An initializer for weights or a Tensor. If not specified,
        uses Xavier initialization.
      l2loss: An l2 weight decay to apply.
      phase: The phase of graph construction.  See `pt.Phase`.
      parameter_modifier: A function to modify parameters that is applied after
        creation and before use.
    Returns:
      A Pretty Tensor handle to the layer.
    Raises:
      ValueError: if this is not rank 2 or the number of input nodes
      (second dim) is not known.
    """
    size = input_layer.shape[-1]
    if weights is None:
      weights = layers.xavier_init(size, 0)

    param = parameter_modifier('weights', self.variable('weights', [size],
                                                        weights), phase)
    layers.add_l2loss(input_layer.bookkeeper, param, l2loss)

    return input_layer.with_tensor(input_layer * param, parameters=self.vars)
# pylint: enable=invalid-name


# pylint: disable=invalid-name
@prettytensor.Register(assign_defaults=('activation_fn', 'l2loss',
                                        'parameter_modifier', 'phase'))
class fully_connected(prettytensor.VarStoreMethod):

  def __call__(self,
               input_layer,
               size,
               activation_fn=None,
               l2loss=None,
               weights=None,
               bias=tf.zeros_initializer,
               transpose_weights=False,
               phase=prettytensor.Phase.train,
               parameter_modifier=parameters.identity,
               name=PROVIDED):
    """Adds the parameters for a fully connected layer and returns a tensor.

    The current PrettyTensor must have rank 2.

    Args:
      input_layer: The Pretty Tensor object, supplied.
      size: The number of neurons
      activation_fn: A tuple of (activation_function, extra_parameters). Any
        function that takes a tensor as its first argument can be used. More
        common functions will have summaries added (e.g. relu).
      l2loss: Set to a value greater than 0 to use L2 regularization to decay
        the weights.
      weights:  An initializer for weights or a Tensor. If not specified,
        uses He's initialization.
      bias: An initializer for the bias or a Tensor. No bias if set to None.
      transpose_weights: Flag indicating if weights should be transposed;
        this is useful for loading models with a different shape.
      phase: The phase of graph construction.  See `pt.Phase`.
      parameter_modifier: A function to modify parameters that is applied after
        creation and before use.
      name: The name for this operation is also used to create/find the
        parameter variables.
    Returns:
      A Pretty Tensor handle to the layer.
    Raises:
      ValueError: if the Pretty Tensor is not rank 2  or the number of input
        nodes (second dim) is not known.
    """
    if input_layer.get_shape().ndims != 2:
      raise ValueError(
          'fully_connected requires a rank 2 Tensor with known second '
          'dimension: %s' % input_layer.get_shape())
    in_size = input_layer.shape[1]
    if input_layer.shape[1] is None:
      raise ValueError('Number of input nodes must be known.')
    books = input_layer.bookkeeper
    if weights is None:
      weights = layers.he_init(in_size, size, activation_fn)

    dtype = input_layer.tensor.dtype
    weight_shape = [size, in_size] if transpose_weights else [in_size, size]

    params = parameter_modifier(
        'weights',
        self.variable('weights', weight_shape,
                      weights, dt=dtype),
        phase)
    y = tf.matmul(input_layer, params, transpose_b=transpose_weights)
    layers.add_l2loss(books, params, l2loss)
    if bias is not None:
      y += parameter_modifier(
          'bias',
          self.variable('bias', [size], bias, dt=dtype),
          phase)

    if activation_fn is not None:
      if not isinstance(activation_fn, collections.Sequence):
        activation_fn = (activation_fn,)
      y = layers.apply_activation(books,
                                  y,
                                  activation_fn[0],
                                  activation_args=activation_fn[1:])
    books.add_histogram_summary(y, '%s/activations' % y.op.name)
    return input_layer.with_tensor(y, parameters=self.vars)
# pylint: enable=invalid-name


@prettytensor.Register
def apply_with_summary(input_layer, operation, *op_args, **op_kwargs):
  """Applies the given operation to `input_layer` and create a summary.

  Args:
    input_layer: The input layer for this op.
    operation: An operation that takes a tensor and the supplied args.
    *op_args: Extra arguments for operation.
    **op_kwargs: Keyword arguments for the operation.
  Returns:
    A new layer with operation applied.
  """
  return layers.apply_activation(input_layer.bookkeeper,
                                 input_layer.tensor,
                                 operation,
                                 activation_args=op_args,
                                 activation_kwargs=op_kwargs)


@prettytensor.Register()
def _rapply(input_layer, operation, *op_args, **op_kwargs):
  """Applies the given operation to this after expanding op_args.

  Args:
    input_layer: The input layer for this op.
    operation: An operation that takes a tensor and the supplied args.
    *op_args: Extra arguments for operation.
    **op_kwargs: Keyword arguments for the operation.
  Returns:
    A new layer with operation applied.
  """
  op_args = list(op_args)
  op_args.append(input_layer.tensor)
  return input_layer.with_tensor(operation(*op_args, **op_kwargs))


@prettytensor.Register(method_name='apply')
def apply_op(input_layer, operation, *op_args, **op_kwargs):
  """Applies the given operation to this before without adding any summaries.

  Args:
    input_layer: The input layer for this op.
    operation: An operation that takes a tensor and the supplied args.
    *op_args: Extra arguments for operation.
    **op_kwargs: Keyword arguments for the operation.
  Returns:
    A new layer with operation applied.
  """
  return input_layer.with_tensor(
      operation(input_layer.tensor, *op_args, **op_kwargs))


@prettytensor.Register
def __getitem__(input_layer, key):  # pylint: disable=invalid-name
  if input_layer.is_sequence():
    return input_layer.with_tensor(input_layer.sequence[key])
  else:
    return input_layer.tensor[key]


@prettytensor.Register
def join(input_layer, others, include_self=True, join_function=None):
  """Joins the provided PrettyTensors with this using the join function.

  Args:
    input_layer: The input layer for this op.
    others: Sequence of PrettyTensor objects.
    include_self: Whether or not this includes itself or if the value is only
      derived from others.
    join_function: The function to use for joining, must accept a list of
      tensors. Use None for concat on the final dimension.
  Returns:
    self.
  """
  if include_self:
    list_of_tensors = [input_layer]
    list_of_tensors.extend(others)
  else:
    list_of_tensors = others
  return prettytensor.join_pretty_tensors(list_of_tensors, input_layer,
                                          join_function)


def _check_split_dims(num_splits, split_dim, shape):
  if split_dim >= len(shape):
    raise ValueError('split_dim out of bounds: %d  %s' % (split_dim, shape))
  if shape[split_dim] % num_splits != 0:
    raise ValueError(
        'Failure to split %s tensor at split_dim=%d\nMust divide the split '
        'dimension evenly: %d mod %d != 0' %
        (shape, split_dim, shape[split_dim], num_splits))


@prettytensor.Register
def unzip(input_layer, split_dim=0, num_splits=2):
  """Unzips this Tensor along the split_dim into num_splits Equal chunks.

  Examples:

  * `[1, 2, 3, 4] -> [1, 3], [2, 4]`
  * `[[1, 1], [2, 2], [3, 3], [4, 4]] -> [[1, 1], [3, 3]], [[2, 2], [4, 4]]`

  Args:
    input_layer: The chainable object, supplied.
    split_dim: The dimension to split along. Defaults to batch.
    num_splits: The number of splits.
  Returns:
    A list of PrettyTensors.
  Raises:
    ValueError: If split_dim is out of range or isn't divided evenly by
      num_splits.
  """
  shape = input_layer.shape
  _check_split_dims(num_splits, split_dim, shape)
  splits = functions.unzip(input_layer, split_dim, shape[split_dim], num_splits)
  return input_layer.with_sequence(splits)


@prettytensor.Register
def concat(input_layer, concat_dim, other_tensors=None):
  """Concatenates input PrettyTensor with other_tensors along the specified dim.

  This adds the Pretty Tensor passed via input_layer to the front of the list of
  tensors to concat.

  Args:
    input_layer: The input layer.
    concat_dim: The dimension along which to concat.
    other_tensors: The tensors to concatenate with as an iterable or None if
      this is called on a sequence.
  Returns:
    A new PrettyTensor.
  Raises:
    ValueError: If other_tensors is None and this is not a sequence.
  """
  if input_layer.is_sequence():
    all_tensors = input_layer.sequence
    all_tensors.extend(other_tensors or [])
  else:
    all_tensors = [input_layer]
    if other_tensors is None:
      raise ValueError('Other Tensors must be supplied.')
    all_tensors.extend(other_tensors)
  # Edge cases really only apply when this is a sequence with 0 or 1 element.
  if not all_tensors:
    return prettytensor.wrap_sequence([])
  else:
    return tf.concat(concat_dim, all_tensors)


@prettytensor.Register(method_name='slice')
def slice_(input_layer, begin, size):
  """Extracts a slice from a tensor.

  This operation extracts a slice of size `size` from a tensor `input` starting
  at the location specified by `begin`. The slice `size` is represented as a
  tensor shape, where `size[i]` is the number of elements of the 'i'th dimension
  of 'input' that you want to slice. The starting location (`begin`) for the
  slice is represented as an offset in each dimension of `input`. In other
  words, `begin[i]` is the offset into the 'i'th dimension of 'input' that you
  want to slice from.

  `begin` is zero-based; 'size' is one-based. If `size[i]` is -1,
  all remaining elements in dimension i are included in the
  slice. In other words, this is equivalent to setting:

  `size[i] = input.dim_size(i) - begin[i]`

  This operation requires that:

  `0 <= begin[i] <= begin[i] + size[i] <= Di  for i in [0, n]`

  Examples:

      # 'input' is [[[1, 1, 1], [2, 2, 2]],
      #             [[3, 3, 3], [4, 4, 4]],
      #             [[5, 5, 5], [6, 6, 6]]]
      tf.slice(input, [1, 0, 0], [1, 1, 3]) ==> [[[3, 3, 3]]]
      tf.slice(input, [1, 0, 0], [1, 2, 3]) ==> [[[3, 3, 3],
                                                  [4, 4, 4]]]
      tf.slice(input, [1, 0, 0], [2, 1, 3]) ==> [[[3, 3, 3]],
                                                 [[5, 5, 5]]]

  Args:
    input_layer: A Tensor.
    begin: An int32 or int64 Tensor of length rank(input_layer)
    size: An int32 or int64 Tensor of length rank(input_layer)
  Returns:
    A tensor with the selected slice.
  """
  return tf.slice(input_layer, begin, size)


@prettytensor.Register
def split(input_layer, split_dim=0, num_splits=2):
  """Splits this Tensor along the split_dim into num_splits Equal chunks.

  Examples:

  * `[1, 2, 3, 4] -> [1, 2], [3, 4]`
  * `[[1, 1], [2, 2], [3, 3], [4, 4]] -> [[1, 1], [2, 2]], [[3, 3], [4, 4]]`

  Args:
    input_layer: The chainable object, supplied.
    split_dim: The dimension to split along. Defaults to batch.
    num_splits: The number of splits.
  Returns:
    A list of PrettyTensors.
  Raises:
    ValueError: If split_dim is out of range or isn't divided evenly by
      num_splits.
  """
  shape = input_layer.shape
  _check_split_dims(num_splits, split_dim, shape)
  splits = tf.split(split_dim, num_splits, input_layer)
  return input_layer.with_sequence(splits)


@prettytensor.Register
def squeeze(input_layer, squeeze_dims=None):
  """Removes dimensions of size 1 from the shape of a tensor.

  This operation returns a tensor of the same type with all singleton
  dimensions removed. If you don't want to remove all singleton dimensions, you
  can remove specific size 1 dimensions by specifying a list of squeeze_dims.

  Args:
    input_layer: A Tensor of any type to squeeze.
    squeeze_dims: An optional list of ints. Defaults to [].

  Returns:
    The sequeezed tensor.
  """

  return tf.squeeze(input_layer, squeeze_dims)


def _zip_with_scalars(args):
  """Zips across args in order and replaces non-iterables with repeats."""
  zipped = []
  for arg in args:
    if isinstance(arg, prettytensor.PrettyTensor):
      zipped.append(arg if arg.is_sequence() else itertools.repeat(arg))
    elif (isinstance(arg, collections.Sequence) and
          not isinstance(arg, tf.compat.bytes_or_text_types)):
      zipped.append(arg)
    else:
      zipped.append(itertools.repeat(arg))
  assert len(args) == len(zipped)
  return zip(*zipped)


@prettytensor.Register(method_name='map')
def map_(input_layer, fn):
  """Maps the given function across this sequence.

  To map an entire template across the sequence, use the `as_fn` method on the
  template.

  Args:
    input_layer: The input tensor.
    fn: A function of 1 argument that is applied to each item in the sequence.
  Returns:
    A new sequence Pretty Tensor.
  Raises:
    ValueError: If the input_layer does not hold a sequence.
  """
  if not input_layer.is_sequence():
    raise ValueError('Can only map a sequence.')
  return [fn(x) for x in input_layer]


# Note: This is a private method.
@prettytensor.Register
def _map_or_apply(input_layer, op, *args, **kwargs):
  """Map op across the input if it is a sequence; otherwise apply it.

  Note: This takes a keyword argument `right_` to right apply the op to this
  input. The name is chosen to limit conflicts with other keyword arguments.

  Args:
    input_layer: The input_layer (self when chaining).
    op: The op to apply:
    *args: Positional arguments for op; if input is a list then any iterable is
      treated as an argument to co-map (i.e. it zips across non-scalars).
    **kwargs: Keyword arguments for op; note that `right_` is used by this
      function.
  Returns:
    A new Pretty Tensor that is the result of applying the op to every internal
    Tensor.
  Raises:
    ValueError: If a sequence argument is not the same length as the
      input_layer.
  """
  # Name is special because it can also set the name scope.
  kwargs.pop('name')
  right = kwargs.pop('right_', False)
  if input_layer.is_sequence():
    if right:
      args += (input_layer,)
    else:
      args = ((input_layer,) + args)
    result = [op(*x, **kwargs) for x in _zip_with_scalars(args)]
    if len(result) != len(input_layer):
      raise ValueError('Not all arguments were the same length.')
    return result
  else:
    if right:
      my_op = lambda x: op(*(args + (x,)), **kwargs)
    else:
      my_op = lambda x: op(x, *args, **kwargs)
    return my_op(input_layer.tensor)


def _strip_see(m):
  start = m.__doc__.find('See [')
  if start >= 0:
    end = m.__doc__.find('Args:', start)
    if end > 0:
      m.__doc__ = m.__doc__[:start] + m.__doc__[end:]
      return
  tf.logging.info('Unable to fix doc: %s' % m.__doc__)


# Adds some common activation functions.

prettytensor.Register(tf.nn.relu)
prettytensor.Register(tf.nn.relu6)
prettytensor.Register(tf.sigmoid)
prettytensor.Register(tf.nn.softplus)
prettytensor.Register(tf.nn.softsign)
prettytensor.Register(tf.nn.tanh)

prettytensor.Register(functions.leaky_relu)
prettytensor.Register(functions.l1_normalize)
prettytensor.Register(tf.nn.l2_normalize)

# These should be expected because they match numpy.
_strip_see(prettytensor.Register(tf.abs))
prettytensor.Register(tf.complex_abs)
prettytensor.Register(tf.log)
prettytensor.Register(tf.sqrt)
prettytensor.Register(tf.square)

prettytensor.Register(tf.pack)
prettytensor.Register(tf.unpack)


# Not strictly matching numpy, but reductions are along the same vein.
prettytensor.Register(tf.reduce_all)
prettytensor.Register(tf.reduce_any)
prettytensor.Register(tf.reduce_join)
prettytensor.Register(tf.reduce_max)
prettytensor.Register(tf.reduce_min)
prettytensor.Register(tf.reduce_mean)
prettytensor.Register(tf.reduce_prod)
prettytensor.Register(tf.reduce_sum)

# Casting
prettytensor.Register(tf.to_double)
prettytensor.Register(tf.to_float)
prettytensor.Register(tf.to_int32)
prettytensor.Register(tf.to_int64)


# This one is just plain useful now that indexing works in TF.
prettytensor.Register(method_name='tensor_shape')(tf.shape)
