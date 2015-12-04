<!-- This file was automatically generated. -->
# Pretty Tensor Base Imports



[TOC]
- - -
## Class Bookkeeper

[see details in Bookkeeper.md](Bookkeeper.md)
- - -

## GraphKeys

Graphs can store data in graph keys for constructing the graph.
- - -

### Properties

* BOOKKEEPER
* LOSSES
* MARKED_LOSSES
* RECURRENT_STATE_VARIABLES
* REGULARIZATION_LOSSES
* TEST_VARIABLES
* UPDATE_OPS

- - -

## Loss

Wraps a layer to provide a handle to the tensor and disallows chaining.

A loss can be used as a regular Tensor.  You can also call `mark_as_required`
in order to put the loss into a collection. This is useful for auxilary heads
and other multi-loss structures.

- - -


### get_shape() {#get_shape}





### is_sequence() {#is_sequence}



Losses are never sequences.






### mark_as_required() {#mark_as_required}



Adds this loss to the MARKED_LOSSES collection.





### Properties

* dtype
* name
* shape
* tensor

- - -

## Phase

Some nodes are different depending on the phase of the graph construction.

The standard phases are train, test and infer.

- - -

### Properties

* infer
* test
* train

- - -
## Class PrettyTensor

[see details in PrettyTensor.md](PrettyTensor.md)
- - -
## Class PrettyTensorTupleMixin

[see details in PrettyTensorTupleMixin.md](PrettyTensorTupleMixin.md)
- - -

## Register

Decorator for registering a method in PrettyTensor.

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

- - -


### create_deferred(func, input_layer, deferred_args, deferred_kwargs, name) {#create_deferred}



Creates a deferred node with captured scope.

##### Args:


* func: The original function to call.
* input_layer: The input_layer.
* deferred_args: The arguments that will be used bythe deferred function.
* deferred_kwargs: The keyword args for the deferred function.
* name: The name of this layer.

##### Returns:

A _DeferredLayer that will execute func in the correct scopes.





### create_method(obj) {#create_method}





### fill_kwargs(input_layer, kwargs) {#fill_kwargs}



Applies name_suffix and defaults to kwargs and returns the result.






- - -

## RegisterCompoundOp

This is used to register a compound operation.

The operation is executed immediately on the base PrettyTensor type. This has
the following implications:

1. `tensor` and `sequence` may not be available in the deferred case.
2. The object passed in might be sequential or a layer.

Also because this is intended to provide convenience chaining of other
registered methods, it does not add a name or id scope automatically, which
makes it behave as if the raw methods were called (unless the op itself does
scoping).

- - -


### create_method(func) {#create_method}



Creates the method.






### fill_kwargs(input_layer, kwargs) {#fill_kwargs}



Applies name_suffix and defaults to kwargs and returns the result.






- - -

## UnboundVariable

A UnboundVariable is a variable with a value that is supplied using bind.

UnboundVariables are typically used so that input layers can be specified at a
later time or for hyper parameters. Supplying a UnboundVariable as an input
variable automatically forces the graph to be a template.

- - -


### has_default() {#has_default}





- - -

## VarStoreMethod

Convenience base class for registered methods that create variables.

This tracks the variables and requries subclasses to provide a __call__
method.

- - -


### variable(var_name, shape, init, dt=<dtype: 'float32'>, train=True) {#variable}



Adds a named variable to this bookkeeper or returns an existing one.

Variables marked train are returned by the training_variables method. If
the requested name already exists and it is compatible (same shape, dt and
train) then it is returned. In case of an incompatible type, an exception is
thrown.

##### Args:


* var_name: The unique name of this variable. If a variable with the same
 name exists, then it is returned.
* shape: The shape of the variable.
* init: The init function to use.
* dt: The datatype, defaults to float. This will automatically extract the
 base dtype.
* train: Whether or not the variable should be trained.

##### Returns:

A TensorFlow tensor.


##### Raises:


* ValueError: if reuse is False (or unspecified and allow_reuse is False)
 and the variable already exists or if the specification of a reused
 variable does not match the original.



- - -

## apply_optimizer(losses, regularize=True, include_marked=True) {#apply_optimizer}



Apply an optimizer to the graph and returns a train_op.

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

#### Args:


* optimizer: The optimizer the minimize.
* losses: A list of losses to apply.
* regularize: Whether or not to include the regularization losses.
* include_marked: Whether or not to use the marked losses.
 **kwargs: Additional arguments to pass into the optimizer.

#### Returns:

An operation to use for training that also updates any required ops such as
      moving averages.




- - -

## for_default_graph() {#for_default_graph}



Creates a bookkeeper for the default graph.

#### Args:


* args: Arguments to pass into Bookkeeper's constructor.
 **kwargs: Arguments to pass into Bookkeeper's constructor.

#### Returns:

A new Bookkeeper.


#### Raises:


* ValueError: If args or kwargs are provided and the Bookkeeper already
 exists.


- - -

## for_new_graph() {#for_new_graph}



Creates a Bookkeeper for a new graph.

You must use `m.g.as_default()` to put the graph in scope:

    m = Bookkeeper.for_new_graph()
    with m.g.as_default():
      ...

#### Args:


* args: Arguments to pass into Bookkeeper's constructor.
 **kwargs: Arguments to pass into Bookkeeper's constructor.

#### Returns:

A new Bookkeeper.




- - -

## construct_all() {#construct_all}



Constructs all the given templates in a single pass without redundancy.

This is useful when the templates have a common substructure and you want the
smallest possible graph.

#### Args:


* templates: A sequence of templates.
 **unbound_var_values: The unbound_var values to replace.

#### Returns:

A list of results corresponding to templates.


#### Raises:


* TypeError: If any value in templates is unsupported.
* ValueError: If the unbound_var values specified are not complete or contain
 unknown values.


- - -

## create_composite_loss(regularize=True, include_marked=True, name=cost) {#create_composite_loss}



Creates a loss that is the sum of all specified losses.

#### Args:


* losses: A sequence of losses to include.
* regularize: Whether or not to include regularization losses.
* include_marked: Whether or not to use the marked losses.
* name: The name for this variable.

#### Returns:

A single tensor that is the sum of all losses.




- - -


## defaults_scope(... {#defaults_scope}

Many Pretty Tensor methods support setting defaults. The supported defaults and methods that use them are:


* `activation_fn`:
    * [conv2d](PrettyTensor.md#conv2d)
    * [fully_connected](PrettyTensor.md#fully_connected)

* `batch_normalize`:
    * [conv2d](PrettyTensor.md#conv2d)

* `l2loss`:
    * [conv2d](PrettyTensor.md#conv2d)
    * [diagonal_matrix_mul](PrettyTensor.md#diagonal_matrix_mul)
    * [fully_connected](PrettyTensor.md#fully_connected)

* `learned_moments_update_rate`:
    * [batch_normalize](PrettyTensor.md#batch_normalize)

* `phase`:
    * [batch_normalize](PrettyTensor.md#batch_normalize)
    * [dropout](PrettyTensor.md#dropout)

* `scale_after_normalization`:
    * [batch_normalize](PrettyTensor.md#batch_normalize)

* `stddev`:
    * [conv2d](PrettyTensor.md#conv2d)
    * [diagonal_matrix_mul](PrettyTensor.md#diagonal_matrix_mul)
    * [fully_connected](PrettyTensor.md#fully_connected)
    * [lstm_cell](PrettyTensor.md#lstm_cell)
    * [gru_cell](PrettyTensor.md#gru_cell)

* `unroll`:
    * [cleave_sequence](PrettyTensor.md#cleave_sequence)

* `variance_epsilon`:
    * [batch_normalize](PrettyTensor.md#batch_normalize)

- - -

## global_step() {#global_step}



Returns the global step variable.





- - -

## join_pretty_tensors(output, join_function=None, name=join) {#join_pretty_tensors}



Joins the list of pretty_tensors and sets head of output_pretty_tensor.

#### Args:


* tensors: A sequence of Layers or SequentialLayerBuilders to join.
* output: A pretty_tensor to set the head with the result.
* join_function: A function to join the tensors, defaults to concat on the
 last dimension.
* name: A name that is used for the name_scope

#### Returns:

The result of calling with_tensor on output


#### Raises:


* ValueError: if pretty_tensors is None or empty.


- - -

## make_template(func) {#make_template}



Given an arbitrary function, wrap it so that it does parameter sharing.





- - -

## recurrent_state() {#recurrent_state}




- - -

## set_recurrent_state_saver() {#set_recurrent_state_saver}



Sets the state saver used for recurrent sequences.





- - -

## template(books=None, optional=False) {#template}



Starts a Pretty Tensor graph template.

A Layer in the resulting graph can be realized by calling
`bind(key=value)` and then `construct`.

#### Args:


* key: A key for this template, used for assigning the correct substitution.
* books: The bookkeeper.
* optional: If this template is an optional value.

#### Returns:

A template that can be constructed or attached to other layers and that
    guarantees parameter reuse when constructed/attached multiple times.




- - -

## with_update_ops() {#with_update_ops}



Creates a new op that runs all of the required updates when train_op runs.

#### Args:


* train_op: An operation that will run every step, usually the result of an
 optimizer.

#### Returns:

A new op that returns the same value as train_op, but also runs the
    updaters.




- - -

## wrap(books=None, tensor_shape=None) {#wrap}



Creates an input layer representing the given tensor.

#### Args:


* tensor: The tensor.
* books: The bookkeeper.
* tensor_shape: An optional shape that will be set on the Tensor or verified
 to match the tensor.

#### Returns:

A layer.




- - -

## wrap_sequence(books=None, tensor_shape=None) {#wrap_sequence}



Creates an input layer representing the given sequence of tensors.

#### Args:


* sequence: A sequence of tensors.
* books: The bookkeeper.
* tensor_shape: An optional shape that will be set on the Tensor or verified
 to match the tensor.

#### Returns:

A layer.




- - -


## Extensions



- - -
