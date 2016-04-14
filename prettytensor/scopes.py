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
"""Contains methods related to making templates."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import functools
import traceback

from six.moves import zip  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.ops import variable_scope


@contextlib.contextmanager
def var_and_name_scope(names):
  """Creates a variable scope and a name scope.

  If a variable_scope is provided, this will reenter that variable scope.
  However, if none is provided then the variable scope will match the generated
  part of the name scope.

  Args:
    names: A tuple of name_scope, variable_scope or None.
  Yields:
    The result of name_scope and variable_scope as a tuple.
  """
  # pylint: disable=protected-access
  if not names:
    yield None, None
  else:
    name, var_scope = names
    with tf.name_scope(name) as scope:
      # TODO(eiderman): This is a workaround until the variable_scope updates land
      # in a TF release.
      old_vs = tf.get_variable_scope()
      if var_scope is None:
        count = len(name.split('/'))
        scoped_name = '/'.join(scope.split('/')[-count - 1:-1])
        full_name = (old_vs.name + '/' + scoped_name).lstrip('/')
      else:
        full_name = var_scope.name

      vs_key = tf.get_collection_ref(variable_scope._VARSCOPE_KEY)
      try:
        # TODO(eiderman): Remove this hack or fix the full file.
        try:
          vs_key[0] = tf.VariableScope(
              old_vs.reuse,
              name=full_name,
              initializer=old_vs.initializer,
              regularizer=old_vs.regularizer,
              caching_device=old_vs.caching_device)
        except AttributeError:
          vs_key[0] = variable_scope._VariableScope(
              old_vs.reuse,
              name=full_name,
              initializer=old_vs.initializer)

        vs_key[0].name_scope = scope
        yield scope, vs_key[0]
      finally:
        vs_key[0] = old_vs


def get_current_name_scope():
  """Gets the current name scope."""
  # pylint: disable=protected-access
  g = tf.get_default_graph()
  # TODO(eiderman): Remove this hack once TF update is released.
  if isinstance(g._name_stack, tuple):
    return g._name_stack[0] + '/'
  else:
    return g._name_stack + '/'


def _get_last_part_of_name_scope(scope):
  splits = scope.split('/')
  return splits[-2]


def make_template(name, func, *args, **kwargs):
  """Given an arbitrary function, wrap it so that it does parameter sharing."""
  if args or kwargs:
    func = functools.partial(func, *args, **kwargs)
  return Template(name, func)


def skip_common_stack_elements(stacktrace, base_case):
  """Skips items that the target stacktrace shares with the base stacktrace."""
  for i, (trace, base) in enumerate(zip(stacktrace, base_case)):
    if trace != base:
      return stacktrace[i:]
  return stacktrace[-1:]


class Template(object):
  """A Template captures variable and namescopes to help variable sharing."""

  def __init__(self, name, func):
    """Creates a template for the given function.

    Args:
      name: The variable_scope to use, if None the current scope is captured.
      func: The function to apply each time.
    """
    self._func = func
    if name:
      self._var_scope = None
      self._name = name
    else:
      self._var_scope = tf.get_variable_scope()
      self._name = None
    self._reuse = None
    self._stacktrace = traceback.format_stack()[:-3]

  def _call_func(self, args, kwargs):
    try:
      self._reuse = True
      return self._func(*args, **kwargs)
    except Exception as exc:
      # Reraise the exception, but append the original definition to the
      # trace.
      args = exc.args
      if not args:
        arg0 = ''
      else:
        arg0 = args[0]
      trace = ''.join(skip_common_stack_elements(self._stacktrace,
                                                 traceback.format_stack()))
      arg0 = '%s\n\noriginally defined at:\n%s' % (arg0, trace)
      new_args = [arg0]
      new_args.extend(args[1:])
      exc.args = tuple(new_args)
      raise

  def __call__(self, *args, **kwargs):
    if self._name:
      with var_and_name_scope((self._name, self._var_scope)) as (_, vs):
        if self._reuse:
          vs.reuse_variables()
        else:
          self._var_scope = vs
        return self._call_func(args, kwargs)
    else:
      with tf.variable_scope(self._var_scope, reuse=self._reuse) as vs:
        return self._call_func(args, kwargs)
