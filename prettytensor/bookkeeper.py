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
"""Manages the bookkeeping of creating a model in TensorFlow.

This includes the variables, initializers, gradients, losses, summaries and
update operations.

Variable Names
===

Variables names use separate scoping to names.  This is because namescope is
designed for unique names and is somewhat dependent on creation order.  Saved
and loaded variables need to have consistent names across refactorings and model
tweaks. This means that the same variable will be reused if it has the same
name, even if it is in a disconnected subgraph.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from prettytensor import sequence_with_deltas


_BOOKKEEPER = ('__bookkeeper',)


class GraphKeys(object):
  """Graphs can store data in graph keys for constructing the graph."""
  LOSSES = 'losses'
  MARKED_LOSSES = 'marked_losses'
  RECURRENT_STATE_VARIABLES = 'recurrent_state_variables'
  REGULARIZATION_LOSSES = 'regularization_losses'
  TEST_VARIABLES = tf.GraphKeys.LOCAL_VARIABLES
  UPDATE_OPS = tf.GraphKeys.UPDATE_OPS

EPSILON = 0.00001

# Separator for name scope.
_NAME_SCOPE_SEP = '/'
BOOKKEEPER_FACTORY = None


def _strip_colon(name):
  if ':' in name:
    return name[:name.rfind(':')]
  else:
    return name


def _tag_for(name):
  index_of_end = name.rfind(_NAME_SCOPE_SEP)
  if index_of_end == -1:
    return _strip_colon(name)
  else:
    return _strip_colon(name[index_of_end + 1:])


def for_default_graph(*args, **kwargs):
  """Creates a bookkeeper for the default graph.

  Args:
    *args: Arguments to pass into Bookkeeper's constructor.
    **kwargs: Arguments to pass into Bookkeeper's constructor.
  Returns:
    A new Bookkeeper.
  Raises:
    ValueError: If args or kwargs are provided and the Bookkeeper already
      exists.
  """
  graph = tf.get_default_graph()
  collection = graph.get_collection(_BOOKKEEPER)
  if collection:
    if args or kwargs:
      raise ValueError('Requesting construction of a BookKeeper that already '
                       'exists: %s %s' % (args, kwargs))
    return collection[0]
  else:
    books = BOOKKEEPER_FACTORY(*args, g=graph, **kwargs)
    graph.add_to_collection(_BOOKKEEPER, books)
    return books


def for_new_graph(*args, **kwargs):
  """Creates a Bookkeeper for a new graph.

  You must use `m.g.as_default()` to put the graph in scope:

      m = Bookkeeper.for_new_graph()
      with m.g.as_default():
        ...

  Args:
    *args: Arguments to pass into Bookkeeper's constructor.
    **kwargs: Arguments to pass into Bookkeeper's constructor.
  Returns:
    A new Bookkeeper.
  """
  graph = tf.Graph()
  with graph.as_default():
    return for_default_graph(*args, **kwargs)


class Bookkeeper(object):
  """Small class to gather needed pieces from a Graph being built.

  This class is mostly an implementation detail of Pretty Tensor and almost
  never needs to be used when building a model. Most of the useful methods
  are exposed in the `pt` namespace. The most common usecase for directly
  calling a Bookkeeper methods are to create summaries in the same way as
  Pretty Tensor that are controlled by the `pt.defaults_scope`.
  """

  def __init__(self,
               g=None,
               default_device=None,
               global_step=None):  # pylint: disable=redefined-outer-name
    """Creates a Bookkeeper.

    Args:
      g: A graph, if not specified then the default graph is used.
      default_device: A default device or function.
      global_step: A variable to use as a global step.
    Raises:
      ValueError: If global_step is not an integer variable.
    """
    if g is None:
      self._g = tf.get_default_graph()
    else:
      self._g = g
    self._train_op = None
    # List of summaries to collect.
    self._summary_tags = set()
    if global_step and global_step.dtype.base_dtype not in (tf.int32, tf.int64):
      raise ValueError('Global step must be an int32 or int64 variable: %s' %
                       global_step.dtype)
    self._global_step = global_step

    if default_device:
      # pylint: disable=protected-access
      self.g._device_function_stack.append(default_device)

    self._recurrent_state = None
    self.reset_summary_collections()

  # Exposed properties without setters.

  def reset_summary_collections(self):
    """Sets the summary collections to the default."""
    self.summary_collections = [tf.GraphKeys.SUMMARIES]

  @property
  def update_ops(self):
    """Operations that update variables on each training step."""
    return tuple(self._g.get_collection(GraphKeys.UPDATE_OPS))

  @property
  def summaries(self):
    """A list of summaries to display."""
    return tuple(self._g.get_collection(tf.GraphKeys.SUMMARIES))

  @property
  def regularization_losses(self):
    """Returns a tuple of regularization losses."""
    return tuple(self._g.get_collection(GraphKeys.REGULARIZATION_LOSSES))

  @property
  def marked_losses(self):
    """Returns a tuple of non-regularization losses."""
    return tuple(self._g.get_collection(GraphKeys.MARKED_LOSSES))

  @property
  def g(self):
    """The graph that this Bookkeeper is building."""
    return self._g

  @property
  def global_step(self):
    """Returns global step counter."""
    if self._global_step is None:
      self._add_global_counter()
    return self._global_step

  @property
  def recurrent_state(self):
    """Returns the current recurrent state manager/state saver or creates one.

    Returns:
      Either the current recurrent state manager or a new one that tracks
      recurrent variables and does no state saving during training, but allows
      full control during inference.
    """
    if self._recurrent_state is None:
      self._recurrent_state = SimpleStateSaver()
    return self._recurrent_state

  @recurrent_state.setter
  def recurrent_state(self, state_saver):
    """Sets recurrent state to the given state saver.

    Args:
      state_saver: An object with state(name) and save_state(name) methods.
    """
    self._recurrent_state = state_saver

  def with_update_ops(self, train_op):
    if train_op is None:
      raise ValueError('train_op cannot be None.')
    update_ops = self.update_ops
    if not update_ops:
      return train_op
    if isinstance(train_op, tf.Operation):
      return tf.group(train_op, *update_ops)
    else:
      # Ensure that the result of evaluating train_op is the same.
      with self.g.as_default(), tf.control_dependencies(self.update_ops):
        return tf.identity(train_op)

  def _add_global_counter(self):
    """Adds a global counter, called once for setup by @property global_step."""
    assert self._global_step is None

    # Force this into the top-level namescope. Instead of forcing top-level
    # here, we could always call this in __init__() and then keep whatever
    # namescopes are around then.
    with self.g.as_default(), self.g.name_scope(None):
      try:
        self._global_step = self.g.get_tensor_by_name('global_step:0')
      except KeyError:
        self._global_step = tf.Variable(0, name='global_step', trainable=False)

  def add_scalar_summary(self, x, tag=None):
    """Adds a scalar summary for x."""
    if not self.summary_collections:
      return
    with self.g.as_default():
      tag = tag or _tag_for(x.name)
      summary = (tf.summary.scalar(
          tag, x, collections=self.summary_collections))
      return summary

  def add_histogram_summary(self, x, tag=None):
    """Add a summary operation to visualize the histogram of x's values."""
    if not self.summary_collections:
      return
    with self.g.as_default():
      tag = tag or _tag_for(x.name)
      summary = tf.summary.histogram(
          tag, x, collections=self.summary_collections)
      return summary

  def exponential_moving_average(self,
                                 var,
                                 avg_var=None,
                                 decay=0.999,
                                 ignore_nan=False):
    """Calculates the exponential moving average.

    TODO(): check if this implementation of moving average can now
    be replaced by tensorflows implementation.

    Adds a variable to keep track of the exponential moving average and adds an
    update operation to the bookkeeper. The name of the variable is
    '%s_average' % name prefixed with the current variable scope.

    Args:
       var: The variable for which a moving average should be computed.
       avg_var: The variable to set the average into, if None create a zero
         initialized one.
       decay: How much history to use in the moving average.
         Higher, means more history values [0, 1) accepted.
       ignore_nan: If the value is NaN or Inf, skip it.
    Returns:
       The averaged variable.
    Raises:
      ValueError: if decay is not in [0, 1).
    """
    with self._g.as_default():
      if decay < 0 or decay >= 1.0:
        raise ValueError('Decay is %5.2f, but has to be in [0, 1).' % decay)
      if avg_var is None:
        avg_name = '%s_average' % _bare_var_name(var)
        with tf.control_dependencies(None):
          with tf.name_scope(avg_name + '/Initializer/'):
            if isinstance(var, tf.Variable):
              init_val = var.initialized_value()
            elif var.get_shape().is_fully_defined():
              init_val = tf.constant(0,
                                     shape=var.get_shape(),
                                     dtype=var.dtype.base_dtype)
            else:
              init_val = tf.constant(0, dtype=var.dtype.base_dtype)
          avg_var = tf.Variable(init_val, name=avg_name, trainable=False)

      num_updates = tf.cast(self.global_step, tf.float32)
      decay = tf.minimum(decay, tf.maximum(0.9, (1.0 + num_updates) /
                                           (10.0 + num_updates)))
      with tf.device(avg_var.device):
        if ignore_nan:
          var = tf.where(tf.is_finite(var), var, avg_var)
        if var.get_shape().is_fully_defined():
          avg_update = tf.assign_sub(avg_var, (1 - decay) * (avg_var - var))
        else:
          avg_update = tf.assign(avg_var,
                                 avg_var - (1 - decay) * (avg_var - var),
                                 validate_shape=False)
      self._g.add_to_collection(GraphKeys.UPDATE_OPS, avg_update)
      return avg_update

  def add_average_summary(self, var, tag=None, decay=0.999, ignore_nan=True):
    """Add a summary with the moving average of var.

    Adds a variable to keep track of the exponential moving average and adds an
    update operation to the bookkeeper. The name of the variable is
    '%s_average' % name prefixed with the current variable scope.

    Args:
       var: The variable for which a moving average should be computed.
       tag: The tag of the summary. If None var.name[:-2] is used to strip off
         the ':0' that is added by TF.
       decay: How much history to use in the moving average.
         Higher, means more history values [0.9, 1) accepted.
       ignore_nan: If the value is NaN or Inf, skip it. Note that this default
         is different than the exponential_moving_average one.
    Returns:
       The averaged variable.
    Raises:
      ValueError: if decay is not in [0.9, 1).
    """
    if not self.summary_collections:
      return
    with self.g.as_default():
      if decay < 0.9 or decay >= 1.0:
        raise ValueError('Decay is %5.2f, but has to be in [0, 1).' % decay)
      avg_var = self.exponential_moving_average(var,
                                                decay=decay,
                                                ignore_nan=ignore_nan)
      if tag is None:
        tag = _bare_var_name(avg_var)
        tag = self.g.unique_name(tag)
      self.add_scalar_summary(avg_var, tag)
      return avg_var

  def add_losses(self, losses, regularization=False):
    for loss in losses:
      self.add_loss(loss, regularization=regularization)

  def add_loss(self, loss, name=None, regularization=False, add_summaries=True):
    """Append a loss to the total loss for the network.

    Args:
      loss: append this loss operation
      name: The name for this loss, defaults to loss.op.name
      regularization: Set to True if this is a regularization loss.
      add_summaries: Set to True if you want to see scalar and average summary.
    """
    # TODO(eiderman): Strip name out and just rely on the name scope.
    _ = name  # Eliminates pylint warning.
    if regularization:
      self._g.add_to_collection(GraphKeys.REGULARIZATION_LOSSES, loss)

    tf.add_to_collection(GraphKeys.LOSSES, loss)
    if add_summaries:
      self.add_scalar_summary(loss, 'loss')
      self.add_average_summary(loss, 'loss_average')

  def create_composite_loss(self,
                            losses,
                            regularize=True,
                            include_marked=True,
                            name='cost'):
    """Creates a loss that is the sum of all specified losses.

    Args:
      losses: A sequence of losses to include.
      regularize: Whether or not to include regularization losses.
      include_marked: Whether or not to use the marked losses.
      name: The name for this variable.
    Returns:
      A single tensor that is the sum of all losses.
    Raises:
      ValueError: if there are no losses.
    """
    all_losses = []
    if losses:
      all_losses.extend(losses)
    if include_marked:
      all_losses.extend(self.marked_losses)
    if not all_losses:
      raise ValueError('No losses specified!')
    if regularize:
      all_losses.extend(self.regularization_losses)
    with self._g.as_default():
      result = tf.add_n(all_losses, name=name)
      self.add_scalar_summary(result)
      return result


# This implements an interface with GoogleNames so it is inconsistent with the
# rest of the file.
class SimpleStateSaver(object):
  """A minimal implementation of a recurrent neural net state saver.

  There are effectively two ways to run training or inference on a recurrent
  neural net. One is to run inference or training one sample at a time. In this
  case, the "recurrent state" - in the simplest case, the recurrent layer
  activations - needs to be stored across time steps; i.e.:
    while (t < N) { output[t], state[t] = network(input[t], state[t-1]); }

  The alternative approach is to unroll the network in time, effectively
  creating an N-layer non-recurrent neural net for N time steps. This approach
  is stateless but requires a decision on the size of N.

  At runtime, the first approach is commonly used - however, it requires
  persisting the state across time steps somehow. This is what "state savers"
  do - e.g. this class - by providing two calls, state() and save_state().
  Under the hood, this could lead to storing the state in an actual Tensorflow
  variable, but in this minimal state saver, no saving/restoring is actually
  performed - instead, placeholder or constant nodes for state restoring,
  and no_op nodes for state saving, are added to the graph. This allows
  state saving to be performed outside of the Tensorflow runtime, by using
  feeds and fetches on these added nodes (see Tensorflow's Session.run()
  for details on feeds/fetches). Using feeds/fetches instead of Tensorflow
  variables adds a lot of flexibility but also imposes the burden on the caller
  to take care of feeding and fetching states - to hide this complexity from
  Python users, we recommend using the RecurrentRunner class to run inference
  on recurrent networks.

  Feeds and fetches are not commonly supported for training in Tensorflow so an
  alternative approach is required. As described above, recurrent networks are
  usually unrolled to a fixed number of time steps, N, which may be larger, or
  shorter, than the longest training sequence. When longer (i.e. the network is
  large enough to train all sequences in their entirety), this class can be used
  (it doesn't do anything except for providing the initial state value - all
  Zeros). When the unrolled network is shorter than the longest sequence, each
  training example S is split into chunks of size N - s1, s2, ... - and the
  network is trained on these chunks, with one important added detail: The
  recurrent state of the network at the end of s1 is used as the initial value
  for the state when training on s2. This scenario requires more elaborate
  state savers that know when an input sequence is over to reset the state -
  e.g. StateSavingLegacyInput - and is not supported by SimpleStateSaver.
  """
  # pylint: disable=invalid-name

  def __init__(self):
    self._states = {}

  def _as_shape_proto(self, shape):
    return tf.TensorShape(shape).as_proto()

  def add_state(self, state_name, initial_state, batch_size=None):
    """Adds a state to the state saver.

    Args:
      state_name: The name of this state.
      initial_state: The initial state vector. Only zeros are supported.
      batch_size: The batch_size or None for unknown.
    """
    state_shape = initial_state.get_shape().as_list()
    full_shape = [batch_size] + state_shape
    if not batch_size:
      # TODO(): -1 is now reserved for unknown, so this should be
      # updated, but that requires coordination with the binary and is
      # checkpoint incompatible.
      # TODO(eiderman): When we make the above breaking change, we should make
      # the C++ client use the initial state instead of passing in zeros.
      shape_proto = self._as_shape_proto([0] + state_shape)
      batch_size = 1
    else:
      shape_proto = self._as_shape_proto([batch_size] + state_shape)

    # Add a constant tensor of zeros. At training time, this will initialize
    # the state with the initial_state - at inference time,
    # this node is replaced by a feed.
    tiles = [batch_size] + ([1] * len(initial_state.get_shape()))
    feed_op = tf.placeholder_with_default(
        tf.tile(
            tf.expand_dims(initial_state, [0]), tiles),
        shape=full_shape,
        name='%s_feed' % state_name)
    s = {'feed_op': feed_op,
         'feed_type': initial_state.dtype,
         'feed_shape': shape_proto}
    self._states[state_name] = s

  def state(self, state_name):
    if not self._states[state_name]:
      raise ValueError('state %s not found - please call add_state() first.' %
                       state_name)
    return self._states[state_name]['feed_op']

  def save_state(self, state_name, tensor):
    if not self._states[state_name]:
      raise ValueError('state %s not found - please call add_state() first.' %
                       state_name)
    elif 'fetch_name' in self._states[state_name]:
      raise ValueError('save_state has already been called for state %s' %
                       state_name)
    self._states[state_name]['fetch_name'] = tensor.name
    return tf.no_op(name='%s_fetch' % state_name)

  def GetStateDescriptors(self):
    return self._states


def _bare_var_name(var):
  result = var.name[:-2]
  # Remove prefixes.
  result = result.split(_NAME_SCOPE_SEP)[-1]
  return result


def regroup_if_changed(group, op_list, name=None):
  """Creates a new group for op_list if it has changed.

  Args:
    group: The current group. It is returned if op_list is unchanged.
    op_list: The list of operations to check.
    name: The name to use if a new group is created.
  Returns:
    Either group or a new group (or if op_list is empty then no_op).
  """
  has_deltas = isinstance(op_list, sequence_with_deltas.SequenceWithDeltas)
  if (group is None or len(group.control_inputs) != len(op_list) or
      (has_deltas and op_list.has_changed())):
    if has_deltas:
      op_list.mark()
    if op_list:
      return tf.group(*op_list, name=name)
    else:
      return tf.no_op(name=name)
  else:
    return group


def with_update_ops(train_op):
  """Creates a new op that runs all of the required updates when train_op runs.

  Args:
    train_op: An operation that will run every step, usually the result of an
      optimizer.
  Returns:
    A new op that returns the same value as train_op, but also runs the
    updaters.
  """
  books = for_default_graph()
  return books.with_update_ops(train_op)


def global_step():
  """Returns the global step variable."""
  books = for_default_graph()
  return books.global_step


def create_composite_loss(losses=None,
                          regularize=True,
                          include_marked=True,
                          name='cost'):
  """Creates a loss that is the sum of all specified losses.

  Args:
    losses: A sequence of losses to include.
    regularize: Whether or not to include regularization losses.
    include_marked: Whether or not to use the marked losses.
    name: The name for this variable.
  Returns:
    A single tensor that is the sum of all losses.
  """
  books = for_default_graph()
  return books.create_composite_loss(losses,
                                     regularize,
                                     include_marked=include_marked,
                                     name=name)


def apply_optimizer(optimizer,
                    losses,
                    regularize=True,
                    include_marked=True,
                    clip_gradients_by_norm=None,
                    **kwargs):
  """Apply an optimizer to the graph and returns a train_op.

  The resulting operation will minimize the specified losses, plus the
  regularization losses that have been collected during graph construction and
  the losses that were marked by calling `mark_as_required`.

  It will also apply any updates that have been collected (e.g. for moving
  average summaries).

  This is equivalent to:

      total_loss = prettytensor.create_composite_loss(
          losses=losses, regularize=regularize, include_marked=include_marked)
      train_op_without_updates = optimizer.minimize(total_loss)
      train_op = prettytensor.with_update_ops(train_op_without_updates)

  N.B. Pay special attention to the `gate_gradients` argument to the optimizer.
  If your graph is large, it will likely train unacceptably slow if you don't
  specify it as GATE_NONE.

  Args:
    optimizer: The optimizer the minimize.
    losses: A list of losses to apply.
    regularize: Whether or not to include the regularization losses.
    include_marked: Whether or not to use the marked losses.
    clip_gradients_by_norm: If not None, clip gradients by the norm using
      `tf.clip_by_norm`.
    **kwargs: Additional arguments to pass into the optimizer.
  Returns:
    An operation to use for training that also updates any required ops such as
      moving averages.
  """
  books = for_default_graph()

  g_step = kwargs.pop('global_step', books.global_step)
  total_loss = books.create_composite_loss(losses=losses,
                                           regularize=regularize,
                                           include_marked=include_marked)

  grads_and_vars = optimizer.compute_gradients(total_loss, **kwargs)
  if clip_gradients_by_norm is not None:
    clipped_grads_and_vars = []
    for g, v in grads_and_vars:
      if isinstance(g, tf.SparseTensor):
        cg = tf.SparseTensor(
            tf.clip_by_norm(g.values, clip_gradients_by_norm),
            g.indices,
            g.shape)
      elif isinstance(g, tf.IndexedSlices):
        cg = tf.IndexedSlices(
            tf.clip_by_norm(g.values, clip_gradients_by_norm),
            g.indices)
      else:
        cg = tf.clip_by_norm(g, clip_gradients_by_norm)
      clipped_grads_and_vars.append((cg, v))
    grads_and_vars = clipped_grads_and_vars
  train_op = optimizer.apply_gradients(grads_and_vars, global_step=g_step)
  return books.with_update_ops(train_op)


def set_recurrent_state_saver(state_saver):
  """Sets the state saver used for recurrent sequences."""
  books = for_default_graph()
  books.recurrent_state = state_saver


def recurrent_state():
  books = for_default_graph()
  return books.recurrent_state

# Set the factory.
BOOKKEEPER_FACTORY = Bookkeeper  # pylint: disable=invalid-name
