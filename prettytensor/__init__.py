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
"""PrettyTensor nice syntax layer on top of TensorFlow.

This will eventually be the preferred place to import prettytensor.

For now, please use pretty_tensor.py since this is in a state of flux.

see [README.md](https://github.com/google/prettytensor) for documentation.
see pretty_tensor_samples/ for usage examples.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import
from prettytensor import funcs
from prettytensor import train
from prettytensor.bookkeeper import apply_optimizer
from prettytensor.bookkeeper import Bookkeeper
from prettytensor.bookkeeper import create_composite_loss
from prettytensor.bookkeeper import for_default_graph as bookkeeper_for_default_graph
from prettytensor.bookkeeper import for_new_graph as bookkeeper_for_new_graph
from prettytensor.bookkeeper import global_step
from prettytensor.bookkeeper import GraphKeys
from prettytensor.bookkeeper import recurrent_state
from prettytensor.bookkeeper import set_recurrent_state_saver
from prettytensor.bookkeeper import with_update_ops

from prettytensor.pretty_tensor_class import construct_all
from prettytensor.pretty_tensor_class import defaults_scope
from prettytensor.pretty_tensor_class import DIM_REST
from prettytensor.pretty_tensor_class import DIM_SAME
from prettytensor.pretty_tensor_class import join_pretty_tensors
from prettytensor.pretty_tensor_class import Loss
from prettytensor.pretty_tensor_class import PAD_SAME
from prettytensor.pretty_tensor_class import PAD_VALID
from prettytensor.pretty_tensor_class import Phase
from prettytensor.pretty_tensor_class import PrettyTensor
from prettytensor.pretty_tensor_class import PrettyTensorTupleMixin
from prettytensor.pretty_tensor_class import PROVIDED
from prettytensor.pretty_tensor_class import Register
from prettytensor.pretty_tensor_class import RegisterCompoundOp
from prettytensor.pretty_tensor_class import template
from prettytensor.pretty_tensor_class import UnboundVariable
from prettytensor.pretty_tensor_class import VarStoreMethod
from prettytensor.pretty_tensor_class import wrap
from prettytensor.pretty_tensor_class import wrap_sequence

from prettytensor.pretty_tensor_normalization_methods import BatchNormalizationArguments
from prettytensor.scopes import make_template
