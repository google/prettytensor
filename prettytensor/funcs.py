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
"""Experimental functional API for PrettyTensor.

This exposes all of the standard PrettyTensor functions, but instead of
chaining, they are invoked like regular functions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import six

# pylint: disable=unused-import, wildcard-import
from prettytensor.pretty_tensor_image_methods import *
from prettytensor.pretty_tensor_loss_methods import *
from prettytensor.pretty_tensor_methods import *
from prettytensor.pretty_tensor_sparse_methods import *
from prettytensor.recurrent_networks import *


def _remove_non_methods():
  """Removes any object in dict that is not a registered method."""
  cur_module = sys.modules[__name__]
  my_globals = dict(globals())
  # Import here so that it doesn't get added to the global namespace or deleted.
  # pylint: disable=g-import-not-at-top
  from prettytensor.pretty_tensor_class import PrettyTensor
  for name, _ in six.iteritems(my_globals):
    if not hasattr(PrettyTensor, name):
      delattr(cur_module, name)
  # Remove a couple of special ones....
  if hasattr(cur_module, 'bookkeeper'):
    delattr(cur_module, 'bookkeeper')

_remove_non_methods()
