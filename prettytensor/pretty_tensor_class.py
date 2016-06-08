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
"""This provides a nice syntax layer on top of TensorFlow.

see [README.md](https://github.com/google/prettytensor) for documentation.
see pretty_tensor_samples/ for usage examples.

TODO(eiderman): This class should be broken apart into several smaller classes.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import functools
import inspect
import itertools
import operator
import traceback

import enum
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin
import tensorflow as tf

from prettytensor import chain_dict
from prettytensor import bookkeeper
from prettytensor import scopes

# Constants for the shape description.
DIM_SAME = '_'
DIM_REST = '*'

# Constants describing padding.
PAD_SAME = 'SAME'
PAD_VALID = 'VALID'

# A constant that can be used in code to signal that PT is providing the value.
# This is just documentation.
PROVIDED = None

# Maintain a list of valid defaults so they can be validated.
_valid_defaults = {'summary_collections',
                   'trainable_variables',
                   'variable_collections'}
_defaults = {}

# A constant used to disambiguate None from unspecified in a couple of optional
# keyword arguments.
_unspecified = ('unspecified',)


class Phase(enum.Enum):
  """Some nodes are different depending on the phase of the graph construction.

  The standard phases are train, test and infer.
  """
  train = 1
  test = 2
  infer = 3


def _set_shape_on_tensor(tensor, shape):
  """Convenience to set a shape or check it."""
  if shape is not None:
    try:
      tensor.set_shape(shape)
    except ValueError:
      raise ValueError("Requested shape does not match tensor's shape: %s vs %s"
                       % (shape, tensor.get_shape()))
  elif tensor.get_shape().ndims is None:
    raise ValueError('Unknown shape on tensor: %s' % tensor)


def unwrap(tensor):
  """Returns the underlying tensor if tensor is wrapped or tensor.

  Args:
    tensor: The tensor to unwrap.
  Returns:
    Tensor or if it is a pretty tensor, the unwrapped version.
  Raises:
    ValueError: if tensor holds a sequence.
  """
  while isinstance(tensor, (PrettyTensor, Loss)):
    tensor = tensor.tensor
  return tensor


def wrap(tensor, books=None, tensor_shape=None):
  """Creates an input layer representing the given tensor.

  Args:
    tensor: The tensor.
    books: The bookkeeper.
    tensor_shape: An optional shape that will be set on the Tensor or verified
      to match the tensor.
  Returns:
    A layer.
  """
  if books is None:
    books = bookkeeper.for_default_graph()
  if isinstance(tensor, PrettyTensor):
    return tensor.as_layer()
  elif isinstance(tensor, UnboundVariable):

    def set_input_from_unbound_var(data):
      """Sets the input from the given unbound_var."""
      if data is not None:
        return wrap(data, books)
      else:
        return None

    return _DeferredLayer(books, set_input_from_unbound_var, [tensor], {})
  else:
    tensor = tf.convert_to_tensor(tensor, name='input')
    if tensor_shape:
      _set_shape_on_tensor(tensor, tensor_shape)
    return Layer(books, tensor=tensor, name=tensor.name)


def template(key, books=None, optional=False):
  """Starts a Pretty Tensor graph template.

  A Layer in the resulting graph can be realized by calling
  `bind(key=value)` and then `construct`.

  Args:
    key: A key for this template, used for assigning the correct substitution.
    books: The bookkeeper.
    optional: If this template is an optional value.
  Returns:
    A template that can be constructed or attached to other layers and that
    guarantees parameter reuse when constructed/attached multiple times.
  """
  if books is None:
    books = bookkeeper.for_default_graph()

  def set_input_from_unbound_var(data):
    """Sets the input from the given unbound_var."""
    if data is not None:
      return wrap(data, books)
    else:
      return None

  if optional:
    data = UnboundVariable(key=key, default=None)
  else:
    data = UnboundVariable(key=key)
  return _DeferredLayer(books, set_input_from_unbound_var, [data], {})


def wrap_sequence(sequence, books=None, tensor_shape=None):
  """Creates an input layer representing the given sequence of tensors.

  Args:
    sequence: A sequence of tensors.
    books: The bookkeeper.
    tensor_shape: An optional shape that will be set on the Tensor or verified
      to match the tensor.
  Returns:
    A layer.
  """
  if books is None:
    books = bookkeeper.for_default_graph()
  for t in sequence:
    _set_shape_on_tensor(t, tensor_shape)
  return Layer(books, sequence=sequence, name=sequence[0].name)


def _assert_value_not_string(name, kwargs):
  if isinstance(kwargs.get(name, None), six.string_types):
    raise ValueError('%s cannot be a string, must be a tuple or list.' % name)


@contextlib.contextmanager
def defaults_scope(**kwargs):
  """Creates a scope for the defaults that are used in a `with` block.

  In addition to setting defaults for some methods, this also can control:

  * `summary_collections`: Choose which collection to place summaries in or
      disable with `None`.
  * `trainable_variables`: Boolean indicating if variables are trainable.
  * `variable_collections`: Default collections in which to place variables;
      `tf.GraphKeys.VARIABLES` is always included.

  Args:
    **kwargs: The defaults.Cloned from CL 117475959 by 'g4 patch'.
  Yields:
    Doesn't really yield, instead this creates a Context Manager for use in a
    `with` statement.
  Raises:
    ValueError: if a collection type is accidently supplied a string.
  """
  _assert_value_not_string('summary_collections', kwargs)
  _assert_value_not_string('variable_collections', kwargs)

  _check_defaults(kwargs)
  global _defaults
  old_defaults = _defaults
  _defaults = chain_dict.ChainDict(_defaults)
  _defaults.update(kwargs)

  # Special logic to support summary_collections.
  # This is added here because introducing more scopes would add more confusion
  # than overloading this one a bit.
  books = bookkeeper.for_default_graph()
  if 'summary_collections' in _defaults:
    books.summary_collections = _defaults['summary_collections']
  else:
    books.reset_summary_collections()
  try:
    yield _defaults
  finally:
    _defaults = old_defaults


def supported_defaults():
  """Returns a set of supported defaults."""
  return frozenset(_valid_defaults)


def _check_defaults(defaults):
  unused_defaults = set(defaults.keys()) - _valid_defaults
  if unused_defaults:
    raise ValueError('Unused arguments: %s' % unused_defaults)


@contextlib.contextmanager
def _subdivide_context(sequential, branch_count, join_function, name):
  """Creates a context so that branches/joins can be done using 'with'."""
  with sequential.g.as_default(), scopes.var_and_name_scope((name, None)):
    branches = []
    for i in xrange(branch_count):
      branches.append(SequentialLayerBuilder(copy=sequential)
                      .with_name('l%d' % i))
    yield tuple(branches)
    layers = [b.as_layer() for b in branches]
    sequential.join(layers, include_self=False, join_function=join_function)


def join_pretty_tensors(tensors, output, join_function=None, name='join'):
  """Joins the list of pretty_tensors and sets head of output_pretty_tensor.

  Args:
    tensors: A sequence of Layers or SequentialLayerBuilders to join.
    output: A pretty_tensor to set the head with the result.
    join_function: A function to join the tensors, defaults to concat on the
      last dimension.
    name: A name that is used for the name_scope
  Returns:
    The result of calling with_tensor on output
  Raises:
    ValueError: if pretty_tensors is None or empty.
  """
  if not tensors:
    raise ValueError('pretty_tensors must be a non-empty sequence.')
  with output.g.name_scope(name):
    if join_function is None:
      # Use depth concat
      last_dim = len(tensors[0].shape) - 1
      return output.with_tensor(tf.concat(last_dim, tensors))
    else:
      return output.with_tensor(join_function(tensors))


def _merge_unbound_var_dicts(src, dst):
  """Merges src into dst and throws an exception if a value is incompatible."""
  for k, v in six.iteritems(src):
    if dst.get(k, v) != v:
      trace1 = ''.join(scopes.skip_common_stack_elements(v.stacktrace, dst[
          k].stacktrace))
      trace2 = ''.join(
          scopes.skip_common_stack_elements(dst[k].stacktrace, v.stacktrace))
      raise ValueError('Key conflict: %s\nDefined At:\n%s\nand\n%s' %
                       (k, trace1, trace2))
    else:
      dst[k] = v


def _assign_values_to_unbound_vars(unbound_vars, unbound_var_values):
  """Assigns values to the vars and raises ValueError if one is missing."""
  context = {}
  for key, value in six.iteritems(unbound_var_values):
    if key not in unbound_vars:
      raise ValueError('unexpected key: %s. Legal values are: %s' %
                       (key, list(six.iterkeys(unbound_vars))))
    context[unbound_vars[key]] = value
  unspecified = []
  for unbound_var in six.itervalues(unbound_vars):
    if unbound_var not in context:
      if unbound_var.has_default():
        context[unbound_var] = unbound_var.default
      else:
        unspecified.append(unbound_var.key)
  if unspecified:
    raise ValueError('Unspecified keys: %s' % unspecified)
  return context


def construct_all(templates, **unbound_var_values):
  """Constructs all the given templates in a single pass without redundancy.

  This is useful when the templates have a common substructure and you want the
  smallest possible graph.

  Args:
    templates: A sequence of templates.
    **unbound_var_values: The unbound_var values to replace.
  Returns:
    A list of results corresponding to templates.
  Raises:
    TypeError: If any value in templates is unsupported.
    ValueError: If the unbound_var values specified are not complete or contain
      unknown values.
  """

  def _merge_dicts(src, dst):
    for k, v in six.iteritems(src):
      if dst.get(k, v) != v:
        raise ValueError('Conflicting values bound for %s: %s and %s' %
                         (k, v, dst[k]))
      else:
        dst[k] = v
  # pylint: disable=protected-access
  all_unbound_vars = {}
  context = {}
  for x in templates:
    if isinstance(x, _DeferredLayer):
      _merge_unbound_var_dicts(x.unbound_vars, all_unbound_vars)
      _merge_dicts(x._partial_context, context)
    else:
      raise TypeError('Unexpected type: %s' % type(x))

  _merge_dicts(
      _assign_values_to_unbound_vars(all_unbound_vars, unbound_var_values),
      context)
  # We need to create a result of known size to avoid client pylint errors.
  result = list(templates)
  for i, x in enumerate(result):
    if isinstance(x, _DeferredLayer):
      result[i] = x._construct(context)
  return result


class PrettyTensorTupleMixin(object):
  """Adds methods to any sequence type so that it can be used with binding.

  Generally this can be used with namedtuples to provide clean multi-value
  returns:

  class MyType(namedtuple(...), PrettyTensorTupleMixin):
    pass

  Subclasses with nested structure should note that this does not unpack
  nested structure by default.  You must implement flatten and
  build_from_flattened.
  """

  # NOT_TODO(eiderman): It may be a very bad idea, but this could support applying
  # the standard operations to many PrettyTensors in parallel.

  def has_unbound_vars(self):
    """Returns whether there are any unbound vars in this tuple."""
    return self.unbound_vars

  def flatten(self):
    """Subclasses with nested structure should implement this method.

    Returns:
      A list of data that should be bound and constructed, by default just self.
    """
    return self

  def build_from_flattened(self, flattened):
    """Given a flattened structure from flatten, make a new version of this."""
    return self.__class__(*flattened)

  def bind(self, **bindings):
    """Makes the bindings to each item in this and returns a new tuple."""
    found_vars = set()
    result = []
    for layer in self.flatten():
      if isinstance(layer, _DeferredLayer):
        var_keys = {var.key for var in six.itervalues(layer.unbound_vars)}
        layers_bindings = {
            k: v
            for k, v in six.iteritems(bindings) if k in var_keys
        }
        result.append(layer.bind(**layers_bindings))
        found_vars.update(six.iterkeys(layers_bindings))
      else:
        result.append(layer)
    missing_vars = set(six.iterkeys(bindings)) - found_vars
    if missing_vars:
      raise ValueError('Unused bindings: %s' % missing_vars)
    return self.__class__(*result)

  def construct(self, **bindings):
    return self.build_from_flattened(construct_all(self.flatten(), **bindings))

  @property
  def unbound_vars(self):
    all_unbound_vars = {}
    for layer in self.flatten():
      if isinstance(layer, _DeferredLayer):
        _merge_unbound_var_dicts(layer.unbound_vars, all_unbound_vars)
    return all_unbound_vars

  def as_fn(self, *binding_order):
    """Creates a function by binding the arguments in the given order.

    Args:
      *binding_order: The unbound variables. This must include all values.
    Returns:
      A function that takes the arguments of binding_order.
    Raises:
      ValueError: If the bindings are missing values or include unknown values.
    """
    if len(binding_order) != len(self.unbound_vars):
      raise ValueError('All vars must be specified.')
    for arg in binding_order:
      if arg not in self.unbound_vars:
        raise ValueError('Unknown binding: %s' % arg)

    def func(*args, **kwargs):
      """Constructs a template."""
      if len(binding_order) != len(args):
        raise ValueError('Missing values, expects: %s' % binding_order)
      values = dict(zip(binding_order, args))
      values.update(kwargs)
      return self.construct(**values)

    func.__doc__ = _gen_ipython_string(func, binding_order, [], func.__doc__)
    return func


class Loss(object):
  """Wraps a layer to provide a handle to the tensor and disallows chaining.

  A loss can be used as a regular Tensor.  You can also call `mark_as_required`
  in order to put the loss into a collection. This is useful for auxilary heads
  and other multi-loss structures.
  """

  def __init__(self, loss, name):
    self._loss = loss
    if name is None:
      self._name = loss.op.name
    else:
      self._name = name
    self._marked = False

  @property
  def tensor(self):
    """Returns the tensor for this layer."""
    return self._loss

  @property
  def name(self):
    return self._loss.name

  @property
  def shape(self):
    return self._loss.get_shape().as_list()

  def get_shape(self):
    return self._loss.get_shape()

  @property
  def dtype(self):
    return self._loss.dtype

  def _as_graph_element(self):
    """Returns the underlying graph element if possible."""
    # Self might be holding something else that isn't a true tensor, so
    # if the 'tensor' can behave like a graph element, look for its
    # _AsGraphElement method and call it. Graph elements themselves may not
    # have or need this method, so just return other items directly.
    obj = self.tensor
    conv_fn = getattr(obj, '_as_graph_element', None)
    if conv_fn and isinstance(conv_fn, collections.Callable):
      obj = conv_fn()
    return obj

  def mark_as_required(self):
    """Adds this loss to the MARKED_LOSSES collection."""
    if not self._marked:
      self._loss.graph.add_to_collection(bookkeeper.GraphKeys.MARKED_LOSSES,
                                         self._loss)
      self._marked = True

  def is_sequence(self):
    """Losses are never sequences."""
    return False

  def __str__(self):
    return self._name


class PrettyTensor(object):
  """A PrettyTensor is a Tensor with a builder interface facade.

  A PrettyTensor behaves like a Tensor, but also
  supports a chainable object syntax to quickly define neural networks
  and other layered architectures in TensorFlow.

      result = (pretty_tensor.wrap(input_data)
                .flatten()
                .fully_connected(200, activation_fn=tf.nn.relu)
                .fully_connected(10, activation_fn=None)
                .softmax(labels, name=softmax_name))


  PrettyTensor has 3 modes of operation that share the ability to chain
  methods.

  ## Normal mode

  In the normal mode, everytime a method is called a new PrettyTensor is
  created. This allows for easy chaining and yet you can still use any
  particular object multiple times. This makes it easy to branch your network.

  ## Sequential mode

  In sequential mode, an internal variable - the head - keeps track of the most
  recent output tensor, thus allowing for breaking call chains into multiple
  statements:

      seq = pretty_tensor.wrap(input_data).sequential()
      seq.flatten()
      seq.fully_connected(200, activation_fn=tf.nn.relu)
      seq.fully_connected(10, activation_fn=None)
      result = seq.softmax(labels, name=softmax_name))

  To return to the normal mode, just use `as_layer()`.

  It is important to note that in sequential mode, self is always returned! This
  means that the following 2 definitions are equivalent:

      def network1(input_data):
        seq = pretty_tensor.wrap(input_data).sequential()
        seq.flatten()
        seq.fully_connected(200, activation_fn=(tf.nn.relu,))
        seq.fully_connected(10, activation_fn=None)

      def network2(input_data):
        seq = pretty_tensor.wrap(input_data).sequential()
        x = seq.flatten()
        y = x.fully_connected(200, activation_fn=(tf.nn.relu,))

        # x refers to the sequential here, whose head points at y!
        z = x.fully_connected(10, activation_fn=None)

  ### Branch and Join

  More complex networks can be built using the the first class methods of branch
  and join. `branch` creates a separate PrettyTensor object that points to the
  current head when it is called and this allows the user to define a separate
  tower that either ends in a regression target, output or rejoins the network.
  Rejoining allows the user define composite layers like inception.  `join` on
  the other hand can be used to join multiple inputs or to rejoin a composite
  layer. The default join operation is to concat on the last dimension
  (depth-concat), but custom joins such as Add are also supported.

  In addition to the atoms of branch and join, PrettyTensor provides a clean
  syntax called `subdivide` when the user needs to branch and rejoin for a
  composite layer. `subdivide` breaks the input into the requested number of
  towers and then automatically rejoins the towers after the block completes.
  This makes it so that the indentation matches the logical structure of the
  network.


      seq = pretty_tensor.wrap(input_data).sequential()
      with seq.subdivide(2) as [a, b]:
        a.conv2d([1, 1], 64)
        b.conv2d([1, 1], 64).conv2d([3, 3], 64)
      seq.flatten()
      seq.fully_connected(200, activation_fn=(tf.nn.relu,))
      seq.fully_connected(10, activation_fn=None)
      result = seq.softmax(labels, name=softmax_name)

  ## Template Mode

  Templates allow you to define a (potentially large) graph with some unknown
  values. The most common use case is to leave the input undefined and then
  define a graph normally. The variables are only defined once every time the
  graph is constructed.  For example:

      template = (pretty_tensor.template('input')
                  .fully_connected(200, name='l1')
                  .fully_connected(200, name='l2'))
      train_output = template.construct(input=train_data)

      # All parameters are reused when the same template object is called again.
      test_output = template.construct(input=test_data)

  Any argument to a pretty tensor method can be substituted by using an
  `UnboundVariable`.
  This allows you to parameterize a graph in arbitrary ways. The most cannonical
  usage would be to substitute a phase variable.

      with pretty_tensor.defaults_scope(phase=UnboundVariable('train')):
        # dropout uses train to optionaly disable itself.

        template = (pretty_tensor.template('input')
                    .fully_connected(200, name='l1')
                    .fully_connected(200, name='l2')
                    .dropout(.8))
      train_output = template.construct(input=train_data, train=True)
      test_output = template.construct(input=test_data, train=False)


  You should use caution because if a template is called with incompatible
  values (e.g. train and test using different widths), then it will break.

      template = (pretty_tensor.template('input')
                  .fully_connected(200, name='l1')
                  .fully_connected(
                      pretty_tensor.UnboundVariable('width'), name='l2'))
      train_output = template.construct(input=train_data, width=200)

      # The following line will die because the shared parameter is the wrong
      # size.
      test_output = template.construct(input=test_data, width=100)
  """

  def __init__(self, books):
    self._bookkeeper = books

  @property
  def layer_parameters(self):
    """Returns a dict of short-parameter name to model parameter.

    This only tracks direct dependencies (i.e. `Variable`s used to generate this
    layer.

    Returns:
      A dict of name to parameters.
    """
    raise NotImplementedError('Not implemented')

  @property
  def shape(self):
    return self.get_shape().as_list()

  def get_shape(self):
    if self.is_sequence():
      return self.sequence[0].get_shape()
    else:
      return self.tensor.get_shape()

  @property
  def dtype(self):
    if self.is_sequence():
      return self.sequence[0].dtype
    else:
      return self.tensor.dtype

  @property
  def name(self):
    if self.is_sequence():
      raise TypeError('Sequences do not have unique names.')
    else:
      return self.tensor.name

  @property
  def tensor(self):
    """Returns the tensor for this layer."""
    raise NotImplementedError('Not implemented')

  @property
  def sequence(self):
    """Returns the sequence for this layer."""
    raise NotImplementedError('Not implemented')

  @property
  def defaults(self):
    """Returns the defaults for this layer."""
    raise NotImplementedError('Not implemented')

  def is_sequential_builder(self):
    """Returns true if this is a sequential builder.

    NB: A sequential builder is a mode of construction and is different from
    whether or not this holds a sequence of tensors.

    Returns:
      Whether this is a sequential builder.
    """
    return False

  def is_sequence(self):
    """Returns True if this holds a sequence and False if it holds a Tensor."""
    raise NotImplementedError('Not implemented')

  def with_name(self, name):
    """Sets the name scope for future operations."""
    raise NotImplementedError('Not implemented')

  def with_defaults(self, **kwargs):
    """Sets defaults that will be passed to all derived PrettyTensors.

    These take precedence over the defaults defined by the scope.

    Args:
      **kwargs: The defaults.
    """
    raise NotImplementedError('Not implemented')

  def with_tensor(self, tensor, parameters=None):
    """Returns a PrettyTensor that points to tensor."""
    raise NotImplementedError('Not implemented')

  def with_sequence(self, sequence, parameters=None):
    """Returns a PrettyTensor that points to sequence."""
    raise NotImplementedError('Not implemented')

  def as_layer(self):
    """Returns a PrettyTensor snapshotted to the current tensor or sequence.

    The primary use case of this is to break out of a sequential.

    Returns:
      An immutable PrettyTensor.
    """
    return self

  def _method_complete(self, result):
    """Called after a registered method with the result."""
    if isinstance(result, (PrettyTensor, Loss, PrettyTensorTupleMixin)):
      return result
    elif (isinstance(result, collections.Sequence) and
          not isinstance(result, six.string_types)):
      return self.with_sequence(result)
    else:
      return self.with_tensor(result)

  @property
  def bookkeeper(self):
    return self._bookkeeper

  @property
  def g(self):
    return self._bookkeeper.g

  # So that this behaves like other graph elements.
  graph = g

  def add_loss(self, loss, name=None):
    """Adds a loss and returns a wrapper for that loss."""
    self.bookkeeper.add_loss(loss, name=name)
    return Loss(loss, name)

  # pylint: disable=invalid-name
  def _replace_args_with_defaults(self, _args=None, **kwargs):
    """Internal method to fill absent values in the kwargs with the defaults.

    Args:
      _args: A list of arguments to replace if a subset is required.  Name
        chosen to prevent conflicts with kwargs.
      **kwargs: The arguments to replace with defaults.
    Returns:
      A map with the same fields as kwargs, but absent values are filled with
      defaults.
    """
    if _args is None:
      _args = six.iterkeys(kwargs)
    my_defaults = self.defaults
    for k in _args:
      if k not in kwargs:
        if k in my_defaults:
          kwargs[k] = my_defaults[k]
        elif k in _defaults:
          kwargs[k] = _defaults[k]
    return kwargs

  def attach_template(self, _template, _key, **unbound_var_values):
    """Attaches the template to this such that _key=this layer.

    Note: names were chosen to avoid conflicts with any likely unbound_var keys.

    Args:
      _template: The template to construct.
      _key: The key that this layer should replace.
      **unbound_var_values: The values for the unbound_vars.
    Returns:
      A new layer with operation applied.
    Raises:
      ValueError: If _key is specified twice or there is a problem computing the
        template.
    """
    if _key in unbound_var_values:
      raise ValueError('%s specified twice.' % _key)
    unbound_var_values[_key] = self
    return _template.as_layer().construct(**unbound_var_values)

  @functools.wraps(tf.Tensor.eval)
  def eval(self, feed_dict=None, session=None):
    if self.is_sequence():
      if session is None:
        session = tf.get_default_session()
      return session.run(self.sequence, feed_dict=feed_dict)
    else:
      return self.tensor.eval(feed_dict=feed_dict, session=session)
  # pylint: enable=invalid-name

  def _as_graph_element(self):
    """Returns the underlying graph element if possible."""
    if self.is_sequence():
      raise TypeError('A Pretty Tensor that holds a sequence cannot be '
                      'represented as a graph element.')
    else:
      # Self might be holding something else that isn't a true tensor, so
      # if the 'tensor' can behave like a graph element, look for its
      # _AsGraphElement method and call it. Graph elements themselves may not
      # have or need this method, so just return other items directly.
      obj = self.tensor
      conv_fn = getattr(obj, '_as_graph_element', None)
      if conv_fn and isinstance(conv_fn, collections.Callable):
        obj = conv_fn()
      return obj

  # Overloaded operators.
  # Give this higher priority in numpy computations.
  __array_priority__ = 100

  def __add__(self, other):
    return self._map_or_apply(operator.add, other, name='apply_op')

  def __radd__(self, other):
    return self._map_or_apply(operator.add, other, right_=True, name='apply_op')

  def __sub__(self, other):
    return self._map_or_apply(operator.sub, other, name='apply_op')

  def __rsub__(self, other):
    return self._map_or_apply(operator.sub, other, right_=True, name='apply_op')

  def __mul__(self, other):
    return self._map_or_apply(operator.mul, other, name='apply_op')

  def __rmul__(self, other):
    return self._map_or_apply(operator.mul, other, right_=True, name='apply_op')

  def __div__(self, other):
    return self._map_or_apply(operator.div, other, name='apply_op')

  def __rdiv__(self, other):
    return self._map_or_apply(operator.div, other, right_=True, name='apply_op')

  def __truediv__(self, other):
    return self._map_or_apply(operator.truediv, other, name='apply_op')

  def __rtruediv__(self, other):
    return self._map_or_apply(
        operator.truediv, other, right_=True, name='apply_op')

  def __mod__(self, other):
    return self._map_or_apply(operator.mod, other, name='apply_op')

  def __rmod__(self, other):
    return self._map_or_apply(operator.mod, other, right_=True, name='apply_op')

  def __lt__(self, other):
    return self._map_or_apply(operator.lt, other, name='apply_op')

  def __le__(self, other):
    return self._map_or_apply(operator.le, other, name='apply_op')

  def __gt__(self, other):
    return self._map_or_apply(operator.gt, other, name='apply_op')

  def __ge__(self, other):
    return self._map_or_apply(operator.and_, other, name='apply_op')

  def __and__(self, other):
    return self._map_or_apply(operator.and_, other, name='apply_op')

  def __rand__(self, other):
    return self._map_or_apply(
        operator.and_, other, right_=True, name='apply_op')

  def __or__(self, other):
    return self._map_or_apply(operator.or_, other, name='apply_op')

  def __ror__(self, other):
    return self._map_or_apply(
        operator.ror_, other, right_=True, name='apply_op')

  def __xor__(self, other):
    return self._map_or_apply(operator.xor, other, name='apply_op')

  def __rxor__(self, other):
    return self._map_or_apply(operator.xor, other, right_=True, name='apply_op')

  def __invert__(self):
    return self._map_or_apply(operator.invert, name='apply_op')

  def __neg__(self):
    return self._map_or_apply(operator.neg, name='apply_op')

  def __abs__(self):
    return self._map_or_apply(operator.abs, name='apply_op')

  def __len__(self):
    if self.is_sequence():
      return len(self.sequence)
    else:
      return len(self.tensor)

  def __nonzero__(self):
    return True

  # __nonzero__ is now called __bool__ in Python 3, so just do an alias.
  __bool__ = __nonzero__

  def __iter__(self):
    if self.is_sequence():
      for i in xrange(len(self.sequence)):
        yield self[i]
    else:
      raise ValueError('Can only iterate on a sequence and not a Tensor.')


class Layer(PrettyTensor):
  """PrettyTensor layer that references a tensor or a sequence.

  Everytime a method is called that creates a tensor, a new Layer is returned.
  """

  def __init__(self,
               books=None,
               copy=None,
               name=None,
               tensor=None,
               sequence=None,
               scope=_unspecified,
               defaults=None):
    """Creates a PrettyTensor object.

    Args:
      books: The bookkeeper that tracks the variables, or None to use the one
        from copy.
      copy: Another PrettyTensor to copy.
      name: The name of this layer.
      tensor: A new Tensor to set, if specified then sequence should be None.
      sequence: A new sequence to set, if specified then tensor should be None.
      scope: The scope, if not specified then the scope from copy is used.
        The default value is a specific tuple, which is never legal and so
        will not collide with any legal value.
      defaults: A dict of defaults.
    Raises:
      ValueError: if this doesn't end up with a bookkeeper, value and name.
    """
    # pylint: disable=protected-access
    if copy:
      super(self.__class__, self).__init__(books or copy.bookkeeper)
      if tensor is not None or sequence is not None:
        self._tensor = tensor
        self._sequence = sequence
      else:
        self._tensor = copy._tensor
        self._sequence = copy._sequence
      self._scope = copy._scope if scope is _unspecified else scope
      self._defaults = defaults or copy._defaults
    else:
      super(self.__class__, self).__init__(books)
      self._tensor = tensor
      self._sequence = sequence
      self._scope = None if scope is _unspecified else scope
      self._defaults = defaults or {}
    if name is None:
      if tensor is not None:
        self._name = self._tensor.op.name
      else:
        self._name = self._sequence[0].op.name
    else:
      self._name = name
    if (self.bookkeeper is None or
        (self._sequence is None == self._tensor is None)):
      raise ValueError('Not completely specified: %s %s %s' %
                       (self.bookkeeper, self._sequence, self._tensor))
    self._layer_parameters = {}

  @property
  def layer_parameters(self):
    return self._layer_parameters

  def __str__(self):
    return self._name

  @property
  def op(self):
    return self.tensor.op

  @property
  def tensor(self):
    """Returns the tensor for this layer."""
    if self._sequence:
      raise ValueError('This PrettyTensor is currently a sequence.')
    return self._tensor

  @property
  def sequence(self):
    """Returns the sequence for this layer."""
    if self._tensor is not None:
      raise ValueError('This PrettyTensor is currently a Tensor.')
    return self._sequence

  def is_sequence(self):
    """Returns True if this holds a sequence and False if it holds a Tensor."""
    return self._sequence is not None

  @functools.wraps(PrettyTensor.with_tensor)
  def with_tensor(self, tensor, parameters=None):
    # This is very forgiving since there are quite a few types that act like
    # tensors.
    if isinstance(tensor, collections.Sequence):
      raise ValueError('Attempting to use a sequence as a tensor %s.' %
                       (tensor,))
    layer = Layer(copy=self, tensor=unwrap(tensor), sequence=None)
    if parameters:
      layer.layer_parameters.update(parameters)
    return layer

  @functools.wraps(PrettyTensor.with_sequence)
  def with_sequence(self, sequence, parameters=None):
    if not isinstance(sequence, collections.Sequence):
      raise ValueError('Attempting to use a tensor as a sequence.')
    layer = Layer(copy=self, tensor=None, sequence=sequence)
    if parameters:
      layer.layer_parameters.update(parameters)
    return layer

  def sequential(self):
    """Creates a SequentialLayerBuilder that tracks the most recent tensor."""
    return SequentialLayerBuilder(head=self)

  @functools.wraps(PrettyTensor.with_name)
  def with_name(self, name):
    """Sets the name scope for future operations."""
    with self.g.as_default(), scopes.var_and_name_scope((name, None)) as (
        name_scope, var_scope):
      return Layer(copy=self, name=self._name, scope=(name_scope, var_scope))

  @functools.wraps(PrettyTensor.with_defaults)
  def with_defaults(self, **kwargs):
    _check_defaults(kwargs)
    return Layer(copy=self, name=self._name, defaults=kwargs)

  @property
  def defaults(self):
    return self._defaults


class UnboundVariable(object):
  """An UnboundVariable is a variable with a value that is supplied using bind.

  UnboundVariables are typically used so that input layers can be specified at a
  later time or for hyper parameters. Supplying a UnboundVariable as an input
  variable automatically forces the graph to be a template.
  """

  def __init__(self, key, default=_unspecified):
    """Inits this with the given key and if specified, a default."""
    self.key = key
    self.default = default
    self.stacktrace = traceback.format_stack()[:-1]

  def has_default(self):
    return self.default is not _unspecified


class _DeferredLayer(PrettyTensor):
  """Defines a template by encapsulating a deferred construction function."""

  def __init__(self,
               books,
               method,
               method_args,
               method_kwargs,
               defaults=None,
               scope=None,
               pass_through=None,
               partial_context=None):
    """Creates a _DeferredLayer.

    This searches all of method_args and method_kwargs (and any sublists or
    dicts) for UnboundVariables so that it knows the required variables for
    construction.

    During construction, any UnboundVariables or DeferredLayers are replaced
    with their proper values for that pass.

    Args:
      books: The bookkeeper for bookkeeping.
      method: A method to call that will create.
      method_args: A sequence of args for the method.
      method_kwargs: A dict of key word args for the method.
      defaults: Defaults to set on any layers made by calling a method on this.
      scope: A base scope for creating values.
      pass_through: Optionally instead of giving method a callable object,
        pass_through can be set in order to construct another layer instead.
      partial_context: A mapping of UnboundVariables to values.
    Raises:
      ValueError: if both method and pass_through are set or neither is set.
    """
    super(self.__class__, self).__init__(books)
    self._layer = None
    self._unbound_vars = {}
    if method is None:
      if pass_through is None:
        raise ValueError('method or pass_through must be set.')
      self._pass_through = pass_through
      self._unbound_vars = dict(pass_through.unbound_vars)
    else:
      if pass_through is not None:
        raise ValueError('Cannot set both method and pass_through')
      self._pass_through = None
      self._method = method
      self._method_args = method_args
      self._method_kwargs = method_kwargs
      self._merge_all_unbound_vars(method_args)
      self._merge_all_unbound_vars(method_kwargs)
    self._scope = scope
    self._defaults = defaults or {}
    self._partial_context = partial_context or {}
    for k in six.iterkeys(self._partial_context):
      if k.key in self._unbound_vars:
        del self._unbound_vars[k.key]

  def _merge_all_unbound_vars(self, arg):
    if isinstance(arg, UnboundVariable):
      _merge_unbound_var_dicts({arg.key: arg}, self._unbound_vars)
    elif isinstance(arg, _DeferredLayer):
      _merge_unbound_var_dicts(arg.unbound_vars, self._unbound_vars)
    elif (isinstance(arg, collections.Sequence) and
          not isinstance(arg, six.string_types)):
      for x in arg:
        self._merge_all_unbound_vars(x)
    elif isinstance(arg, collections.Mapping):
      for x in six.itervalues(arg):
        self._merge_all_unbound_vars(x)

  @property
  def unbound_vars(self):
    return self._unbound_vars

  @property
  def defaults(self):
    return self._defaults

  def _replace_deferred(self, arg, context):
    """This replaces all deferred nodes (UnboundVariables and _DeferredLayers).

    If arg is a sequence or a dict, then it's deferred values are also replaced.

    Args:
      arg: The argument to replace. If a list or a dict, then all items are also
        replaced.
      context: The context for this replacement.
    Returns:
      The replaced values or arg if it is not a deferred node.
    """
    if isinstance(arg, UnboundVariable):
      return context[arg]
    elif isinstance(arg, _DeferredLayer):
      # pylint: disable=protected-access
      return arg._construct(context)
    elif isinstance(arg, tuple):
      return tuple((self._replace_deferred(x, context) for x in arg))
    elif (isinstance(arg, collections.Sequence) and
          not isinstance(arg, six.string_types)):
      return [self._replace_deferred(x, context) for x in arg]
    elif isinstance(arg, collections.Mapping):
      return {k: self._replace_deferred(v, context)
              for k, v in six.iteritems(arg)}
    else:
      return arg

  def _construct(self, context):
    """Constructs this by calling the deferred method.

    This assumes that all unbound_vars have been specified in context and if
    this layer has already been computed in this context, then the previously
    constructed value will be returned.

    Args:
      context: A dict of UnboundVariables/_DeferredLayers to their values.
    Returns:
      The result of calling the given method on this layer.
    """
    with self.g.as_default():
      if self._pass_through:
        # pylint: disable=protected-access
        return self._pass_through._construct(context)
      current_value = context.get(self, None)
      assert current_value is not _unspecified, 'Circular dependency'
      if current_value is not None:
        return current_value
      context[self] = _unspecified
      method_args = self._replace_deferred(self._method_args, context)
      method_kwargs = self._replace_deferred(self._method_kwargs, context)
      result = self._method(*method_args, **method_kwargs)
      _strip_unnecessary_contents_from_stack(result, set())

      context[self] = result
      return result

  def bind(self, **bindings):
    """Creates a new template with the given unbound variables bound.

    Args:
      **bindings: Arguments for every deferred parameter.
    Returns:
      A new template with the given bindings.
    Raises:
      ValueError: If any of the bindings do not correspond to unbound variables.
    """
    new_context = dict(self._partial_context)
    unknown_keys = []
    for k, v in six.iteritems(bindings):
      if k not in self._unbound_vars:
        unknown_keys.append(k)
      new_context[self._unbound_vars[k]] = v
    if unknown_keys:
      raise ValueError(
          'The following keys are not associated with any unbound vars: %s, '
          'legal values are %s' %
          (unknown_keys, list(self._unbound_vars.keys())))
    return _DeferredLayer(self.bookkeeper,
                          None,
                          (),
                          {},
                          scope=self._scope,
                          defaults=self._defaults,
                          pass_through=self,
                          partial_context=new_context)

  def as_fn(self, *binding_order):
    """Creates a function by binding the arguments in the given order.

    Args:
      *binding_order: The unbound variables. This must include all values.
    Returns:
      A function that takes the arguments of binding_order.
    Raises:
      ValueError: If the bindings are missing values or include unknown values.
    """
    if len(binding_order) != len(self._unbound_vars):
      raise ValueError('All vars must be specified.')
    for arg in binding_order:
      if arg not in self._unbound_vars:
        raise ValueError('Unknown binding: %s' % arg)

    def func(*args, **kwargs):
      """Constructs a template."""
      if len(binding_order) != len(args):
        raise ValueError('Missing values, expects: %s' % binding_order)
      values = dict(zip(binding_order, args))
      values.update(kwargs)
      return self.construct(**values)

    func.__doc__ = _gen_ipython_string(func, binding_order, [], func.__doc__)
    return func

  def construct(self, **bindings):
    """Constructs the graph and returns either a tensor or a sequence.

    Args:
      **bindings: Arguments for every deferred parameter.
    Returns:
      The value that is placed into this.
    """
    context = _assign_values_to_unbound_vars(self._unbound_vars, bindings)
    context.update(self._partial_context)
    return self._construct(context)

  def sequential(self):
    """Creates a SequentialLayerBuilder that tracks the most recent tensor."""
    return SequentialLayerBuilder(head=self)

  def with_name(self, name):
    """Sets the name scope for future operations."""
    with self.g.as_default(), tf.variable_scope(name) as var_scope:
      name_scope = scopes.get_current_name_scope()
      return _DeferredLayer(self.bookkeeper,
                            None,
                            (),
                            {},
                            scope=(name_scope, var_scope),
                            defaults=self._defaults,
                            pass_through=self,
                            partial_context=self._partial_context)

  @functools.wraps(PrettyTensor.with_defaults)
  def with_defaults(self, **kwargs):
    _check_defaults(kwargs)
    return _DeferredLayer(self.bookkeeper,
                          None,
                          (),
                          {},
                          scope=self._scope,
                          defaults=kwargs,
                          pass_through=self,
                          partial_context=self._partial_context)

  def get_value(self):
    return self._layer

  def attach_template(self, _template, _key, **unbound_var_values):
    """Attaches the template to this with the _key is supplied with this layer.

    Note: names were chosen to avoid conflicts.

    Args:
      _template: The template to construct.
      _key: The key that this layer should replace.
      **unbound_var_values: The values for the unbound_vars.
    Returns:
      A new layer with operation applied.
    Raises:
      ValueError: If _key is specified twice or there is a problem computing the
        template.
    """
    if _key in unbound_var_values:
      raise ValueError('%s specified twice.' % _key)
    unbound_var_values[_key] = self
    return _DeferredLayer(self.bookkeeper,
                          _template.as_layer().construct,
                          [],
                          unbound_var_values,
                          scope=self._scope,
                          defaults=self._defaults,
                          partial_context=self._partial_context)


def _strip_unnecessary_contents_from_stack(result, processed):
  """Remove the distracting lines from the stored tracebacks.

  This also reduces memory overhead by removing the frame contents. This is very
  important when doing long unrolls.

  Args:
    result: The result to process.
    processed: A set of already processed nodes, used to stop early.
  """
  # pylint: disable=protected-access
  if isinstance(result, (PrettyTensor, Loss)):
    if result.is_sequence():
      for tensor in result.sequence:
        _strip_unnecessary_contents_from_stack(tensor, processed)
        return
    else:
      result = result.tensor
  if hasattr(result, 'op'):
    result = result.op
  if result in processed:
    return
  else:
    processed.add(result)
  trace = []
  found = False
  for f, line_no, method, _ in result._traceback:
    if (method in ('_replace_deferred', '_construct') and
        f.endswith('pretty_tensor_class.py')):
      found = True
      continue
    trace.append((f, line_no, method, {}))
  result._traceback = trace

  # Assume that if we didn't find any PT deferred lines, then this node is
  # not part of the deferred construction.
  if not found:
    return
  for inp in result.inputs:
    _strip_unnecessary_contents_from_stack(inp, processed)


class SequentialLayerBuilder(PrettyTensor):
  """PrettyTensor that builds a network by modifying an internal head."""

  def __init__(self, head=None, copy=None):
    """Creates a SequentialLayerBuilder either with the head or copied."""
    # pylint: disable=protected-access
    if copy is None == head is None:
      raise ValueError('Must set head (%s) or copy (%s).' % (head, copy))
    if copy:
      head = copy._head
    super(self.__class__, self).__init__(head.bookkeeper)
    self._head = head

  @property
  def layer_parameters(self):
    return self._head.layer_parameters

  def __str__(self):
    return 'Sequential (head=%s)' % self._head

  @property
  def tensor(self):
    """Returns the tensor for this layer."""
    return self._head.tensor

  @property
  def sequence(self):
    """Returns the sequence for this layer."""
    return self._head.sequence

  @property
  def _scope(self):
    # pylint: disable=protected-access
    return self._head._scope

  def is_sequential_builder(self):
    return True

  def is_sequence(self):
    return self._head.is_sequence()

  def with_tensor(self, tensor, parameters=None):
    self._head = self._head.with_tensor(tensor, parameters=parameters)
    return self

  def with_sequence(self, sequence, parameters=None):
    self._head = self._head.with_sequence(sequence, parameters=parameters)
    return self

  def set_head(self, new_head):
    """Sets the head an returns self."""
    self._head = new_head
    return self

  def _method_complete(self, result):
    """Called after an extention method with the result."""
    if isinstance(result, PrettyTensor):
      self._head = result
      return self
    elif isinstance(result, Loss):
      return result
    elif isinstance(result, PrettyTensorTupleMixin):
      self._head = result[0]
      return result
    else:
      self._head = self._head.with_tensor(result)
      return self

  def subdivide(self, branches, name='mixed'):
    """Branches this pretty tensor with the default join function of concat.

    This should be used in a with statement:

        with pt.subdivide(2) as [a, b]:
          a...
          b...

    Args:
      branches: The number of branches.
      name: A base name for this branch.
    Returns:
      A python context manager to use in a with statement that supplies a
      sequence of tensors with one per branch.
    """
    return self.subdivide_with(branches, None, name=name)

  def subdivide_with(self, branches, join_function, name='mixed'):
    """Branches this pretty tensor and uses an explicit join function.

    This should be used in a with statement, for example to fork and join with
    a sum:

        with pt.subdivide_with(2, tf.add_n) as [a, b]:
          a...
          b...

    Args:
      branches: The number of branches.
      join_function: A function to use when rejoining.
      name: A base name for this branch.
    Returns:
      A python context manager to use in a with statement that supplies a
      sequence of tensors with one per branch.
    Raises:
      ValueError: if join_function is None.
    """
    return _subdivide_context(self, branches, join_function, name)

  @functools.wraps(PrettyTensor.with_name)
  def with_name(self, name):
    """Sets the name scope for future operations."""
    self._head = self._head.with_name(name)
    return self

  @functools.wraps(PrettyTensor.with_defaults)
  def with_defaults(self, **kwargs):
    _check_defaults(kwargs)
    self._head = self._head.with_defaults(**kwargs)
    return self

  @property
  def defaults(self):
    return self._head.defaults

  def as_layer(self):
    """Creates a Layer snapshotted to the current head.

    This is used to create break out of a sequence at a specific point.

    Returns:
      A Layer object at this point.
    """
    return self._head

  # Make ops side effect free; as_layer takes a snapshot and thus removes the
  # side effects.
  def __add__(self, other):
    return PrettyTensor.__add__(self.as_layer(), other)

  def __radd__(self, other):
    return PrettyTensor.__radd__(self.as_layer(), other)

  def __sub__(self, other):
    return PrettyTensor.__sub__(self.as_layer(), other)

  def __rsub__(self, other):
    return PrettyTensor.__rsub__(self.as_layer(), other)

  def __mul__(self, other):
    return PrettyTensor.__mul__(self.as_layer(), other)

  def __rmul__(self, other):
    return PrettyTensor.__rmul__(self.as_layer(), other)

  def __div__(self, other):
    return PrettyTensor.__rdiv__(self.as_layer(), other)

  def __rdiv__(self, other):
    return PrettyTensor.__rdiv__(self.as_layer(), other)

  def __truediv__(self, other):
    return PrettyTensor.__truediv__(self.as_layer(), other)

  def __rtruediv__(self, other):
    return PrettyTensor.__rtruediv__(self.as_layer(), other)

  def __mod__(self, other):
    return PrettyTensor.__mod__(self.as_layer(), other)

  def __rmod__(self, other):
    return PrettyTensor.__rmod__(self.as_layer(), other)

  def __lt__(self, other):
    return PrettyTensor.__lt__(self.as_layer(), other)

  def __le__(self, other):
    return PrettyTensor.__le__(self.as_layer(), other)

  def __gt__(self, other):
    return PrettyTensor.__gt__(self.as_layer(), other)

  def __ge__(self, other):
    return PrettyTensor.__ge__(self.as_layer(), other)

  def __and__(self, other):
    return PrettyTensor.__and__(self.as_layer(), other)

  def __rand__(self, other):
    return PrettyTensor.__rand__(self.as_layer(), other)

  def __or__(self, other):
    return PrettyTensor.__or__(self.as_layer(), other)

  def __ror__(self, other):
    return PrettyTensor.__ror__(self.as_layer(), other)

  def __xor__(self, other):
    return PrettyTensor.__xor__(self.as_layer(), other)

  def __rxor__(self, other):
    return PrettyTensor.__rxor__(self.as_layer(), other)

  def __invert__(self):
    return PrettyTensor.__invert__(self.as_layer())

  def __neg__(self):
    return PrettyTensor.__neg__(self.as_layer())

  def __abs__(self):
    return PrettyTensor.__abs__(self.as_layer())

  # Side effecty incremental ops (e.g. +=)

  def __iadd__(self, other):
    return PrettyTensor.__add__(self, other)

  def __isub__(self, other):
    return PrettyTensor.__sub__(self, other)

  def __imul__(self, other):
    return PrettyTensor.__mul__(self, other)

  def __idiv__(self, other):
    return PrettyTensor.__div__(self, other)

  def __itruediv__(self, other):
    return PrettyTensor.__truediv__(self, other)

  def __ifloordiv__(self, other):
    return PrettyTensor.__floordiv__(self, other)

  def __imod__(self, other):
    return PrettyTensor.__mod__(self, other)

  def __ipow__(self, other):
    return PrettyTensor.__pow__(self, other)

  def __iand__(self, other):
    return PrettyTensor.__and__(self, other)

  def __ior__(self, other):
    return PrettyTensor.__or__(self, other)

  def __ixor__(self, other):
    return PrettyTensor.__xor__(self, other)


class VarStoreMethod(object):
  """Convenience base class for registered methods that create variables.

  This tracks the variables and requries subclasses to provide a __call__
  method.
  """

  def __init__(self):
    self.vars = {}

  def variable(self, var_name, shape, init, dt=tf.float32, train=None):
    """Adds a named variable to this bookkeeper or returns an existing one.

    Variables marked train are returned by the training_variables method. If
    the requested name already exists and it is compatible (same shape, dt and
    train) then it is returned. In case of an incompatible type, an exception is
    thrown.

    Args:
      var_name: The unique name of this variable.  If a variable with the same
        name exists, then it is returned.
      shape: The shape of the variable.
      init: The init function to use or a Tensor to copy.
      dt: The datatype, defaults to float.  This will automatically extract the
        base dtype.
      train: Whether or not the variable should be trained; defaults to
        True unless a default_scope has overridden it.
    Returns:
      A TensorFlow tensor.
    Raises:
      ValueError: if reuse is False (or unspecified and allow_reuse is False)
        and the variable already exists or if the specification of a reused
        variable does not match the original.
    """
    # Make sure it is a TF dtype and convert it into a base dtype.
    dt = tf.as_dtype(dt).base_dtype
    if var_name in self.vars:
      v = self.vars[var_name]
      if v.get_shape() != shape:
        raise ValueError(
            'Shape mismatch: %s vs %s. Perhaps a UnboundVariable had '
            'incompatible values within a graph.' % (v.get_shape(), shape))
      return v
    elif callable(init):
      if train is None:
        train = _defaults.get('trainable_variables', True)
      variable_collections = _defaults.get('variable_collections', ())
      if tf.GraphKeys.VARIABLES not in variable_collections:
        variable_collections = list(variable_collections) + [
            tf.GraphKeys.VARIABLES]

      v = tf.get_variable(var_name,
                          shape=shape,
                          dtype=dt,
                          initializer=init,
                          trainable=train,
                          collections=variable_collections)
      self.vars[var_name] = v
      return v
    else:
      v = tf.convert_to_tensor(init, name=var_name, dtype=dt)
      v.get_shape().assert_is_compatible_with(shape)
      self.vars[var_name] = v
      return v


def _gen_ipython_string(func, args, defaults, original_doc):
  """Provides auto-complete hint to ipython.

  If the first line in a docstring is fn(arg1=, arg2=) then they are added to
  auto-complete.  This cannot be called on an instance method.

  Args:
    func: The function that will be modified.
    args: The arguments that this function takes in order.
    defaults: The default arguments corresponding the last arguments.
    original_doc: Original docstring to assign after the magic string.
  Returns:
    The new doc string with the magic bit prepended.
  """
  magic_string = '%s(' % func.__name__

  if defaults:
    default_offset = len(args) - len(defaults)
  else:
    default_offset = len(args)
  for i, value in enumerate(args):
    if i >= default_offset:
      magic_string += '%s=%s, ' % (value, defaults[i - default_offset])
    else:
      magic_string += '%s, ' % value
  if args:
    magic_string = magic_string[:-2]
  magic_string += ')\n\n'
  if original_doc is not None:
    magic_string += original_doc
  return magic_string


def _set_ipython_string(func, args, defaults, original_doc):
  func.__doc__ = _gen_ipython_string(func, args, defaults, original_doc)


def _remove_first_arg_from_doc(func):
  arg_string = 'Args:\n'
  if not func.__doc__:
    return
  start = func.__doc__.find(arg_string)
  if start >= 0:
    start += len(arg_string)
    end = func.__doc__.find('\n', start) + 1
    if end > 0:
      func.__doc__ = func.__doc__[:start] + func.__doc__[end:]


def _should_defer(input_layer, args, kwargs):
  """Checks to see if any of the args are templates."""
  for arg in itertools.chain([input_layer], args, six.itervalues(kwargs)):
    if isinstance(arg, (_DeferredLayer, UnboundVariable)):
      return True
    elif (isinstance(arg, collections.Sequence) and
          not isinstance(arg, six.string_types)):
      if _should_defer(None, arg, {}):
        return True
    elif isinstance(arg, collections.Mapping):
      if _should_defer(None, (), arg):
        return True
  return False

# Remember the original doc so that we can create magic ipython strings.
_original_set_defaults_doc = PrettyTensor.with_defaults.__doc__
_original_defaults_scope_doc = defaults_scope.__doc__
_defaults_to_methods = collections.defaultdict(list)

# If we are in a method scope, then we don't want to add the full id.
_in_method_scope = False


@contextlib.contextmanager
def _method_scope(input_layer, name):
  """Creates a nested set of name and id scopes and avoids repeats."""
  global _in_method_scope
  # pylint: disable=protected-access

  with input_layer.g.as_default(), \
       scopes.var_and_name_scope(
           None if _in_method_scope else input_layer._scope), \
       scopes.var_and_name_scope((name, None)) as (scope, var_scope):
    was_in_method_scope = _in_method_scope
    yield scope, var_scope
    _in_method_scope = was_in_method_scope


class _RegisterBase(object):
  """Base class for the Register* decorators."""

  # pylint: disable=protected-access

  def __init__(self, assign_defaults=(), method_name=None, overwrite=False):
    """Assigns arguments to the decorator.

    Args:
      assign_defaults: A sequence of strings for the default values that should
        be provided.
      method_name: If provided, use this as the method_name instead of the
        wrapped function's name.
      overwrite: If False, throw an exception if this method has already been
        registered.  True should be used in interactive environments or with
        great care.
    """
    if isinstance(assign_defaults, str):
      self._assign_defaults = [assign_defaults]
    else:
      self._assign_defaults = assign_defaults
    self._method_name = method_name
    self._overwrite = overwrite
    _valid_defaults.update(self._assign_defaults)
    default_args = sorted(_valid_defaults)
    default_values = [None] * len(_valid_defaults)
    if six.PY2:
      default_func = PrettyTensor.with_defaults.__func__
    else:
      default_func = PrettyTensor.with_defaults
    _set_ipython_string(default_func, default_args, default_values,
                        _original_set_defaults_doc)
    _set_ipython_string(defaults_scope, default_args, default_values,
                        _original_defaults_scope_doc)

  def __new__(cls, *args, **kwargs):
    """Supports not including the parens."""
    if len(args) == 1 and isinstance(args[0], collections.Callable):
      assert not kwargs
      register = super(_RegisterBase, cls).__new__(cls)
      register.__init__()
      return register(args[0])
    else:
      return super(_RegisterBase, cls).__new__(cls)

  def fill_kwargs(self, input_layer, kwargs):
    """Applies name_suffix and defaults to kwargs and returns the result."""
    return input_layer._replace_args_with_defaults(_args=self._assign_defaults,
                                                   **kwargs)

  def __call__(self, obj):
    if inspect.isclass(obj):
      cls = obj
      doc = cls.__call__.__doc__
      self._name = cls.__name__
      method = self.create_method(obj)
      argspec = inspect.getargspec(obj.__call__)
      args = argspec.args[2:]
    else:
      func = obj
      self._name = func.__name__
      doc = func.__doc__
      method = self.create_method(func)
      argspec = inspect.getargspec(func)
      args = argspec.args[1:]
    name = self._method_name if self._method_name else self._name
    method.__module__ = obj.__module__
    method.__name__ = name
    _set_ipython_string(method, args, argspec.defaults, doc)
    _remove_first_arg_from_doc(method)
    self._has_name_param = 'name' in argspec.args
    if not self._overwrite:
      assert not hasattr(Layer, name), 'Method already defined: %s' % name
      assert not hasattr(SequentialLayerBuilder,
                         name), 'Clash with Sequential: %s' % name

    setattr(PrettyTensor, name, method)

    for default in self._assign_defaults:
      _defaults_to_methods[default].append(name)
    return method

  def create_method(self, func):
    """Returns the method that should be registered on PrettyTensor."""
    raise NotImplementedError('Abstract class')


class Register(_RegisterBase):
  """Decorator for registering a method in PrettyTensor.

  This is either used to decorate a bare function or an object that has a no-arg
  constructor and a __call__ method.

  The first argument to the function will be the PrettyTensor object. The
  registered method's return value must be one of the following:

  1. A PrettyTensor, i.e. the result of calling `with_tensor` or
      `with_sequence`.
  2. A Tensor.
  3. A Loss result from calling `add_loss`.

  `RegisterCompoundOp` is provided for more direct manipulations with some
  caveats.
  """

  # pylint: disable=protected-access

  def __init__(self, assign_defaults=(), method_name=None, overwrite=False):
    """Assigns arguments to the decorator.

    Args:
      assign_defaults: A sequence of strings for the default values that should
        be provided. Defaults are shared across methods.
      method_name: If provided, use this as the method_name instead of the
        wrapped function's name.
      overwrite: if true, overwrites definition if exists.

    """
    super(self.__class__, self).__init__(assign_defaults=assign_defaults,
                                         method_name=method_name,
                                         overwrite=overwrite)

  def create_deferred(self, func, input_layer, deferred_args, deferred_kwargs,
                      name):
    """Creates a deferred node with captured scope.

    Args:
      func: The original function to call.
      input_layer: The input_layer.
      deferred_args: The arguments that will be used bythe deferred function.
      deferred_kwargs: The keyword args for the deferred function.
      name: The name of this layer.
    Returns:
      A _DeferredLayer that will execute func in the correct scopes.
    """
    my_defaults = _defaults

    def _with_method_complete(*args, **kwargs):
      input_layer = args[0]
      with input_layer.g.as_default(), defaults_scope(**my_defaults), \
          tf.name_scope(name):
        return input_layer._method_complete(func(*args, **kwargs))
    # The deferred layer passes on the scope of the source layer so that the
    # construction scope matches that of the immediate version.
    full_args = [input_layer]
    full_args.extend(deferred_args)
    partial_context = {}
    if isinstance(input_layer, _DeferredLayer):
      partial_context = input_layer._partial_context
    return _DeferredLayer(input_layer.bookkeeper,
                          scopes.Template(None, _with_method_complete),
                          full_args,
                          deferred_kwargs,
                          scope=input_layer._scope,
                          defaults=input_layer.defaults,
                          partial_context=partial_context)

  def create_method(self, obj):

    # pylint: disable=missing-docstring
    @functools.wraps(obj)
    def method(*original_args, **kwargs):
      if inspect.isclass(obj):
        func = obj()
      else:
        func = obj
      # If someone is calling a direct function, then we want to wrap the
      # PrettyTensor and otherwise we use as_layer() so that extensions always
      # deal with a non-sequential.
      input_layer = original_args[0]
      if not isinstance(input_layer, PrettyTensor):
        input_layer = wrap(input_layer)
        non_seq_layer = input_layer
      else:
        non_seq_layer = input_layer.as_layer()
      args = original_args[1:]
      if kwargs.get('name', None):
        name = kwargs['name']
      else:
        name = self._name
      # The name scope should be the base scope (input_layer._scope)/layer_name
      scope_name = name.lstrip('_')
      with _method_scope(input_layer, scope_name) as (scope, _):
        if self._has_name_param and not kwargs.get('name', None):
          kwargs['name'] = scope.split('/')[-2]
        kwargs = self.fill_kwargs(input_layer, kwargs)
        if _should_defer(non_seq_layer, args, kwargs):
          result = self.create_deferred(func, non_seq_layer, args, kwargs, name)
        else:
          result = func(non_seq_layer, *args, **kwargs)
        return input_layer._method_complete(result)
      # Exit with because Template handles that.

    return method


class RegisterCompoundOp(_RegisterBase):
  """This is used to register a compound operation.

  The operation is executed immediately on the base PrettyTensor type. This has
  the following implications:

  1. `tensor` and `sequence` may not be available in the deferred case.
  2. The object passed in might be sequential or a layer.

  Also because this is intended to provide convenience chaining of other
  registered methods, it does not add a name or id scope automatically, which
  makes it behave as if the raw methods were called (unless the op itself does
  scoping).
  """

  def __init__(self, assign_defaults=(), method_name=None):
    """Assigns arguments to the decorator.

    Args:
      assign_defaults: A sequence of strings for the default values that should
        be provided. Defaults are shared across methods.
      method_name: If provided, use this as the method_name instead of the
        wrapped function's name.
    """
    super(self.__class__, self).__init__(assign_defaults=assign_defaults,
                                         method_name=method_name)

  def create_method(self, func):
    """Creates the method."""
    # pylint: disable=missing-docstring
    @functools.wraps(func)
    def method(input_layer, *args, **kwargs):
      return func(input_layer, *args, **self.fill_kwargs(input_layer, kwargs))

    return method


def _conversion_function(pt_wrapper, dtype=None, name=None, as_ref=False):
  """Allows PrettyTensors and Loss to work as a tensor."""
  # Ignore as_ref to not create backward compatibility issues.
  _ = name, as_ref
  t = pt_wrapper.tensor
  if dtype and not t.dtype.is_compatible_with(dtype):
    raise ValueError(
        'Tensor conversion requested dtype %s for Tensor with dtype %s: %r' %
        (dtype, t.dtype, t))
  return t


tf.register_tensor_conversion_function(
    (PrettyTensor, Loss), _conversion_function, 100)
