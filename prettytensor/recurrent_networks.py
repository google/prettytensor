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
"""Adds methods related to recurrent networks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

import six
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.ops import control_flow_ops

from prettytensor import bookkeeper
from prettytensor import pretty_tensor_class as prettytensor
from prettytensor.pretty_tensor_class import PROVIDED

# TODO(eiderman): Figure out the best dimensionality for this.
#
# List of Tensors, 1 per timestep: Logically makes sense, but doesn't let you
#   run in preprocessing.
# First dim time, second batch, rest the data: Requires splitting to unroll
#   LSTM and merging time and batch to do other operations that assume batch
#   first.
# First dim batch * time: Requires splitting on batch to run LSTM

STATE_NAME = '%s_prev'


class RecurrentResult(
    collections.namedtuple('RecurrentResult', ['output', 'state']),
    prettytensor.PrettyTensorTupleMixin):
  """Holds the result of a recurrent node which contains the output and a state.

  This also provides binding and construction if the result contains a template.
  """

  def flatten(self):
    """Create a flattened version by putting output first and then states."""
    ls = [self.output]
    ls.extend(self.state)
    return ls

  def build_from_flattened(self, flattened):
    return self.__class__(flattened[0], tuple(flattened[1:]))


# LSTM and GRU cells are implemented as compound ops because they return
# a tuple as a result.
#
# As compound ops, they have more responsibilities:
# 1. input_layer may be sequential, so they need to set the proper head
#    (there can be only one).
# 2. input_layer may be deferred, so they need to only use pretty tensor
#    methods.
@prettytensor.RegisterCompoundOp(assign_defaults='stddev')
def lstm_cell(input_layer,
              states,
              num_units,
              bias=True,
              peephole=True,
              stddev=None,
              init=None):
  """Long short-term memory cell (LSTM).

  Args:
    input_layer: The input layer.
    states: The current state of the network, as
      [[batch, num_units], [batch, num_units]] (c, h).
    num_units: How big is the hidden state.
    bias: Whether or not to use a bias.
    peephole: Whether to use peephole connections as described in
        http://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf
    stddev: Standard deviation for Gaussian initialization of parameters.
    init: A tf.*Initializer that is used to initialize the variables.
  Returns:
    A RecurrentResult.
  """
  # As a compound op, it needs to respect whether or not this is a sequential
  # builder.
  if input_layer.is_sequential_builder():
    layer = input_layer.as_layer()
  else:
    layer = input_layer
  c, h = [prettytensor.wrap(state, layer.bookkeeper) for state in states]
  activation_input = layer.fully_connected(4 * num_units,
                                           bias=bias,
                                           activation_fn=None,
                                           stddev=stddev,
                                           init=init)
  activation_h = h.fully_connected(4 * num_units,
                                   bias=False,
                                   activation_fn=None,
                                   stddev=stddev,
                                   init=init)

  activation = activation_input + activation_h

  # i = input_gate, j = new_input, f = forget_gate, o = output_gate
  split = activation.split(1, 4)
  i = split[0]
  j = split[1]
  f = split[2]
  if bias:
    # Biases of the forget gate are initialized to 1 in order to reduce the
    # scale of forgetting in the beginning of the training.
    f += 1.
  o = split[3]
  if peephole:
    # TODO(eiderman): It would be worthwhile to determine the best initialization.
    i += c.diagonal_matrix_mul(stddev=stddev, init=init)
    f += c.diagonal_matrix_mul(stddev=stddev, init=init)

  f_gate = f.apply(tf.sigmoid, name='f_gate')
  new_c = (
      c * f_gate + i.apply(tf.sigmoid, name='i_gate') * j.apply(tf.tanh))
  if peephole:
    o += new_c.diagonal_matrix_mul(stddev=stddev, init=init)

  new_h = new_c.apply(tf.tanh) * o.apply(tf.sigmoid, name='o_gate')

  if input_layer.is_sequential_builder():
    new_h = input_layer.set_head(input_layer)
  return RecurrentResult(new_h, [new_c, new_h])


@prettytensor.RegisterCompoundOp(assign_defaults='stddev')
def gru_cell(input_layer, state, num_units, bias=True, stddev=None, init=None):
  """Gated recurrent unit memory cell (GRU).

  Args:
    input_layer: The input layer.
    state: The current state of the network. For GRUs, this is a list with
      one element (tensor) of shape [batch, num_units].
    num_units: How big is the hidden state.
    bias: Whether or not to use a bias.
    stddev: Standard deviation for Gaussian initialization of parameters.
    init: A tf.*Initializer that is used to initialize the variables.
  Returns:
    A RecurrentResult.
  """
  # As a compound op, it needs to respect whether or not this is a sequential
  # builder.
  if input_layer.is_sequential_builder():
    layer = input_layer.as_layer()
  else:
    layer = input_layer
  # We start with bias of 1.0 to not reset and not udpate.
  concat = layer.concat(1, [state]).fully_connected(2 * num_units,
                                                    bias=bias,
                                                    bias_init=1.0,
                                                    activation_fn=None,
                                                    stddev=stddev,
                                                    init=init).apply(tf.sigmoid)

  split = concat.split(1, 2)
  r = split[0]
  u = split[1]

  c = layer.concat(1, [r * state]).fully_connected(num_units,
                                                   bias=bias,
                                                   activation_fn=None,
                                                   stddev=stddev,
                                                   init=init).apply(tf.tanh)
  new_h = u * state + (1 - u) * c
  if input_layer.is_sequential_builder():
    new_h = input_layer.set_head(input_layer)
  return RecurrentResult(new_h, [new_h])


def unwrap_all(*args):
  """Unwraps all of the tensors and returns a list."""
  result = [prettytensor.unwrap(x) for x in args]
  return result


def unroll_state_saver(input_layer, name, state_shapes, template, lengths=None):
  """Unrolls the given function with state taken from the state saver.

  Args:
    input_layer: The input sequence.
    name: The name of this layer.
    state_shapes: A list of shapes, one for each state variable.
    template: A template with unbound variables for input and states that
      returns a RecurrentResult.
    lengths: The length of each item in the batch.  If provided, use this to
      truncate computation.
  Returns:
    A sequence from applying the given template to each item in the input
    sequence.
  """
  state_saver = input_layer.bookkeeper.recurrent_state
  state_names = [STATE_NAME % name + '_%d' % i
                 for i in xrange(len(state_shapes))]
  if hasattr(state_saver, 'add_state'):
    for state_name, state_shape in zip(state_names, state_shapes):
      initial_state = tf.zeros(state_shape[1:], dtype=input_layer.dtype)
      state_saver.add_state(state_name,
                            initial_state=initial_state,
                            batch_size=state_shape[0])
  if lengths is not None:
    max_length = tf.reduce_max(lengths)
  else:
    max_length = None

  results = []
  prev_states = []
  for state_name, state_shape in zip(state_names, state_shapes):
    my_shape = list(state_shape)
    my_shape[0] = -1
    prev_states.append(
        tf.reshape(state_saver.state(state_name), my_shape))

  parameters = None
  for i, layer in enumerate(input_layer.sequence):
    with input_layer.g.name_scope('unroll_%00d' % i):
      if i > 0 and max_length is not None:
        # TODO(eiderman): Right now the everything after length is undefined.
        # If we can efficiently propagate the last result to the end, then
        # models with only a final output would require a single softmax
        # computation.
        # pylint: disable=cell-var-from-loop
        result = control_flow_ops.cond(
            i < max_length,
            lambda: unwrap_all(*template(layer, *prev_states).flatten()),
            lambda: unwrap_all(out, *prev_states))
        out = result[0]
        prev_states = result[1:]
      else:
        out, prev_states = template(layer, *prev_states)
    if parameters is None:
      parameters = out.layer_parameters
    results.append(prettytensor.unwrap(out))

  updates = [state_saver.save_state(state_name, prettytensor.unwrap(prev_state))
             for state_name, prev_state in zip(state_names, prev_states)]

  # Set it up so that update is evaluated when the result of this method is
  # evaluated by injecting a dependency on an arbitrary result.
  with tf.control_dependencies(updates):
    results[0] = tf.identity(results[0])
  return input_layer.with_sequence(results, parameters=parameters)


# pylint: disable=invalid-name
@prettytensor.Register
class sequence_lstm(prettytensor.VarStoreMethod):
  """The Sequence Lstm.

  This holds an internal template to insure that the same parameters are used
  each time it is called.
  """

  def __init__(self):
    super(self.__class__, self).__init__()
    self.template = None

  def __call__(self,
               input_layer,
               num_units,
               bias=True,
               peephole=True,
               name=PROVIDED,
               stddev=None,
               init=None,
               lengths=None):
    """Creates an unrolled LSTM to process sequence data.

    The initial state is drawn from the bookkeeper's recurrent state and if it
    supports state saving, then it is saved.

    Args:
      input_layer: PrettyTensor (provided).
      num_units: Number of nodes in the hidden states and the output size.
      bias: Whether or not to use a bias.
      peephole: Whether to use peephole connections.
      name: The name of this layer.
      stddev: Standard deviation for Gaussian initialization of parameters.
      init: A tf.*Initializer that is used to initialize the variables.
      lengths: An optional Tensor that encodes a length for each item in the
        minibatch. This is used to truncate computation.
    Returns:
      A sequence with the result at each timestep.
    Raises:
      ValueError: if head is not a sequence, the shape is not rank 2 or the
        number of nodes (second dim) is not known.
    """
    if not self.template:
      lstm_template = prettytensor.template('input', input_layer.bookkeeper)
      names = ['c', 'h']
      states = [prettytensor.UnboundVariable(n) for n in names]
      # TODO(eiderman): Move this to pt.make_template() when it is ready.
      self.template = lstm_template.lstm_cell(states=states,
                                              num_units=num_units,
                                              bias=bias,
                                              peephole=peephole,
                                              stddev=stddev,
                                              init=init).as_fn('input', *names)

    batch_size = input_layer.shape[0]
    state_shapes = [[batch_size, num_units],
                    [batch_size, num_units]]
    return unroll_state_saver(input_layer, name, state_shapes, self.template,
                              lengths)


# pylint: disable=invalid-name
@prettytensor.Register
class sequence_gru(prettytensor.VarStoreMethod):
  """The Sequence Gru.

  This holds an internal template to insure that the same parameters are used
  each time it is called.
  """

  def __init__(self):
    super(self.__class__, self).__init__()
    self.template = None

  def __call__(self,
               input_layer,
               num_units,
               bias=True,
               name=PROVIDED,
               stddev=None,
               init=None,
               lengths=None):
    """Creates an unrolled GRU to process sequence data.

    The initial state is drawn from the bookkeeper's recurrent state and if it
    supports state saving, then it is saved.

    Args:
      input_layer: PrettyTensor (provided).
      num_units: Number of units in the hidden states.
      bias: Whether or not to use a bias.
      name: The name of this layer.
      stddev: Standard deviation for Gaussian initialization of parameters.
      init: A tf.*Initializer that is used to initialize the variables.
      lengths: An optional Tensor that encodes a length for each item in the
        minibatch. This is used to truncate computation.
    Returns:
      A sequence with the result at each timestep.
    Raises:
      ValueError: if head is not a sequence, the shape is not rank 2 or the
        number of nodes (second dim) is not known.
    """
    if not self.template:
      gru_template = prettytensor.template('input', input_layer.bookkeeper)
      self.template = gru_template.gru_cell(
          prettytensor.UnboundVariable('state'),
          num_units,
          bias=bias,
          stddev=stddev,
          init=init).as_fn('input', 'state')

    batch_size = input_layer.shape[0]
    return unroll_state_saver(input_layer, name, [(batch_size, num_units)],
                              self.template, lengths)


@prettytensor.Register
def squash_sequence(input_layer):
  """"Squashes a sequence into a single Tensor with dim 1 being time*batch.

  A sequence is an array of Tensors, which is not appropriate for most
  operations, this squashes them together into Tensor.

  Defaults are assigned such that cleave_sequence requires no args.

  Args:
    input_layer: The input layer.
  Returns:
    A PrettyTensor containing a single tensor with the first dim containing
    both time and batch.
  Raises:
    ValueError: If the sequence is empty.
  """
  timesteps = len(input_layer.sequence)
  if not timesteps:
    raise ValueError('Empty tensor sequence.')
  elif timesteps == 1:
    result = input_layer.sequence[0]
  else:
    result = tf.concat(0, input_layer.sequence)
  return input_layer.with_tensor(result).with_defaults(unroll=timesteps)


@prettytensor.Register(assign_defaults='unroll')
def cleave_sequence(input_layer, unroll=None):
  """Cleaves a tensor into a sequence, this is the inverse of squash.

  Recurrent methods unroll across an array of Tensors with each one being a
  timestep.  This cleaves the first dim so that each it is an array of Tensors.
  It is the inverse of squash_sequence.

  Args:
    input_layer: The input layer.
    unroll: The number of time steps.
  Returns:
    A PrettyTensor containing an array of tensors.
  Raises:
    ValueError: If unroll is not specified and it has no default or it is <= 0.
  """
  if unroll is None:
    raise ValueError('You must set unroll either here or in the defaults.')

  shape = input_layer.shape
  if shape[0] is not None and shape[0] % unroll != 0:
    raise ValueError('Must divide the split dimension evenly: %d mod %d != 0' %
                     (shape[0], unroll))

  if unroll <= 0:
    raise ValueError('Unroll must be > 0: %s' % unroll)
  elif unroll == 1:
    splits = [input_layer.tensor]
  else:
    splits = tf.split(0, unroll, input_layer.tensor)
  result = input_layer.with_sequence(splits)

  # This is an abuse of the defaults system, but it is safe because we are only
  # modifying result.
  defaults = result.defaults
  if 'unroll' in defaults:
    del defaults['unroll']
  return result


# pylint: disable=invalid-name
@prettytensor.Register
class embedding_lookup(prettytensor.VarStoreMethod):

  def __call__(self,
               input_layer,
               embedding_count,
               embedding_shape,
               name=PROVIDED,
               init=None):
    """Looks up values in a learned embedding lookup.

    `embedding_count` embedding tensors are created each with shape
    `embedding_shape`. The values are by defaulted initialized with a standard
    deviation of 1, but in some cases zero is a more appropropriate initial
    value.  The embeddings themselves are learned through normal
    backpropagation.

    You can initialize these to a fixed embedding and follow with
    stop_gradients() to use a previously learned embedding.

    N.B. This uses  tf.nn.embedding_lookup under the hood, so by default the
    lookup is id % embedding_count

    Args:
      input_layer: PrettyTensor (provided).
      embedding_count: Number of items in the embedding.
      embedding_shape: Shape of each embedding.
      name: The name of this layer.
      init: tf.*Initializer to use for initializing the input or a Tensor.
        Defaults to a truncated normal.
    Returns:
      input_layer
    Raises:
      ValueError: If head is not a rank 2 Tensor with second dim of 1.
    """
    if not hasattr(embedding_shape, '__iter__'):
      raise ValueError('Embedding shape must be a tuple or list.')
    head = input_layer.tensor
    if len(input_layer.shape) == 2:
      if input_layer.shape[1] == 1:
        head = tf.reshape(head, [-1])
      else:
        raise ValueError('Last dim of shape must be 1: %s' % input_layer.shape)
    else:
      raise ValueError('head must be a rank 2 Tensor: %s' % input_layer.shape)
    full_shape = [embedding_count]
    full_shape.extend(embedding_shape)
    if init is None:
      size = 1
      for dim in embedding_shape:
        size *= dim
      init = tf.truncated_normal_initializer(stddev=1. / math.sqrt(size))

    embeddings = self.variable('params', full_shape, init=init)

    name = 'params_1' if name == 'params' else name
    return input_layer.with_tensor(
        tf.nn.embedding_lookup(embeddings, head, name=name),
        parameters=self.vars)


# TODO(eiderman): It would be nice to have a mechanism where a network could
# list the required states it will want to save so that they could be supplied
# to the input.
def lstm_state_tuples(num_nodes, name):
  """Convenience so that the names of the vars are defined in the same file."""
  if not isinstance(num_nodes, tf.compat.integral_types):
    raise ValueError('num_nodes must be an integer: %s' % num_nodes)
  return [(STATE_NAME % name + '_0', tf.float32, num_nodes),
          (STATE_NAME % name + '_1', tf.float32, num_nodes)]


def gru_state_tuples(num_nodes, name):
  """Convenience so that the names of the vars are defined in the same file."""
  if not isinstance(num_nodes, tf.compat.integral_types):
    raise ValueError('num_nodes must be an integer: %s' % num_nodes)
  return [(STATE_NAME % name + '_0', tf.float32, num_nodes)]


def create_sequence_pretty_tensor(sequence_input, shape=None, save_state=True):
  """Creates a PrettyTensor object for the given sequence.

  The first dimension is treated as a time-dimension * batch and a default is
  set for `unroll` and `state_saver`.

  TODO(eiderman): Remove shape.

  Args:
    sequence_input: A SequenceInput or StateSavingSequenceInput
    shape: The shape of each item in the sequence (including batch).
    save_state: If true, use the sequence_input's state and save_state methods.
  Returns:
    2 Layers: inputs, targets
  """
  inputs = prettytensor.wrap_sequence(sequence_input.inputs, tensor_shape=shape)
  targets = prettytensor.wrap_sequence(sequence_input.targets)
  if save_state:
    bookkeeper.set_recurrent_state_saver(sequence_input)
  return inputs, targets


class RecurrentRunner(object):
  """A helper class for managing states for recurrent neural net inference."""

  def __init__(self, batch_size=1):
    self._state_feeds = {}
    self._state_fetches = []
    self._state_feed_names = []
    self._batch_size = batch_size
    self._graph = tf.get_default_graph()

    # Store the feeds and fetches for recurrent states.
    statesaver = bookkeeper.recurrent_state()
    for state in six.itervalues(statesaver.GetStateDescriptors()):
      shape = [d.size for d in state['feed_shape'].dim]
      if shape[0] == 0:
        shape[0] = batch_size
      feed_name = state['feed_op'].name
      self._state_feed_names.append(feed_name)
      self._state_fetches.append(state['fetch_name'])

  def reset(self):
    self._state_feeds = {}

  def run(self, fetch_list, feed_dict=None, sess=None):
    """Runs the graph with the provided feeds and fetches.

    This function wraps sess.Run(), but takes care of state saving and
    restoring by feeding in states and storing the new state values.
    Args:
      fetch_list: A list of requested output tensors.
      feed_dict: A dictionary of feeds - see Session.Run(). Optional.
      sess: The Tensorflow session to run. Can be None.
    Returns:
      The requested tensors as numpy arrays.
    Raises:
      ValueError: If the default graph during object construction was
      different from the current default graph.
    """
    if tf.get_default_graph() != self._graph:
      raise ValueError('The current default graph is different from the graph'
                       ' used at construction time of RecurrentRunner.')
    if feed_dict is None:
      all_feeds_dict = {}
    else:
      all_feeds_dict = dict(feed_dict)
    all_feeds_dict.update(self._state_feeds)
    all_fetches_list = list(fetch_list)
    all_fetches_list += self._state_fetches

    sess = sess or tf.get_default_session()

    # Run the compute graph.
    fetches = sess.run(all_fetches_list, all_feeds_dict)
    # Update the feeds for the next time step.
    states = fetches[len(fetch_list):]
    for i, s in enumerate(states):
      self._state_feeds[self._state_feed_names[i]] = s

    return fetches[:len(fetch_list)]
