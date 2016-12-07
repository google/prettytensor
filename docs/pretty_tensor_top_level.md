<!-- This file was automatically generated. -->
# Pretty Tensor Base Imports



[TOC]
- - -

## BatchNormalizationArguments

BatchNormalizationArguments(learned_moments_update_rate, variance_epsilon, scale_after_normalization)
- - -

### Properties

* count
* index
* learned_moments_update_rate
* scale_after_normalization
* variance_epsilon

- - -
## Class Bookkeeper

[see details in Bookkeeper.md](Bookkeeper.md)
- - -

## GraphKeys

Graphs can store data in graph keys for constructing the graph.
- - -

### Properties

* LOSSES
* MARKED_LOSSES
* RECURRENT_STATE_VARIABLES
* REGULARIZATION_LOSSES
* TEST_VARIABLES
* UPDATE_OPS

- - -
## Class Loss

[see details in Loss.md](Loss.md)
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


### <a name="create_deferred"></a>create_deferred(func, input_layer, deferred_args, deferred_kwargs, name)



Creates a deferred node with captured scope.

##### Args:


* func: The original function to call.
* input_layer: The input_layer.
* deferred_args: The arguments that will be used bythe deferred function.
* deferred_kwargs: The keyword args for the deferred function.
* name: The name of this layer.

##### Returns:

A _DeferredLayer that will execute func in the correct scopes.





### <a name="create_method"></a>create_method(obj)





### <a name="fill_kwargs"></a>fill_kwargs(input_layer, kwargs)



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


### <a name="create_method"></a>create_method(func)



Creates the method.






### <a name="fill_kwargs"></a>fill_kwargs(input_layer, kwargs)



Applies name_suffix and defaults to kwargs and returns the result.






- - -

## UnboundVariable

An UnboundVariable is a variable with a value that is supplied using bind.

UnboundVariables are typically used so that input layers can be specified at a
later time or for hyper parameters. Supplying a UnboundVariable as an input
variable automatically forces the graph to be a template.

- - -


### <a name="has_default"></a>has_default()





- - -

## VarStoreMethod

Convenience base class for registered methods that create variables.

This tracks the variables and requries subclasses to provide a __call__
method.

- - -


### <a name="variable"></a>variable(var_name, shape, init, dt=<dtype: 'float32'>, train=None)



Adds a named variable to this bookkeeper or returns an existing one.

Variables marked train are returned by the training_variables method. If
the requested name already exists and it is compatible (same shape, dt and
train) then it is returned. In case of an incompatible type, an exception is
thrown.

##### Args:


* var_name: The unique name of this variable. If a variable with the same
 name exists, then it is returned.
* shape: The shape of the variable.
* init: The init function to use or a Tensor to copy.
* dt: The datatype, defaults to float. This will automatically extract the
 base dtype.
* train: Whether or not the variable should be trained; defaults to
 True unless a default_scope has overridden it.

##### Returns:

A TensorFlow tensor.


##### Raises:


* ValueError: if reuse is False (or unspecified and allow_reuse is False)
 and the variable already exists or if the specification of a reused
 variable does not match the original.



- - -

## <a name="apply_optimizer"></a>apply_optimizer(losses, regularize=True, include_marked=True, clip_gradients_by_norm=None)



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
* clip_gradients_by_norm: If not None, clip gradients by the norm using
 `tf.clip_by_norm`.
 **kwargs: Additional arguments to pass into the optimizer.

#### Returns:

An operation to use for training that also updates any required ops such as
      moving averages.




- - -

## <a name="for_default_graph"></a>for_default_graph()



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

## <a name="for_new_graph"></a>for_new_graph()



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

## <a name="construct_all"></a>construct_all()



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

## <a name="create_composite_loss"></a>create_composite_loss(regularize=True, include_marked=True, name=cost)



Creates a loss that is the sum of all specified losses.

#### Args:


* losses: A sequence of losses to include.
* regularize: Whether or not to include regularization losses.
* include_marked: Whether or not to use the marked losses.
* name: The name for this variable.

#### Returns:

A single tensor that is the sum of all losses.




- - -


## <a name="defaults_scope"></a>defaults_scope(...

defaults_scope(activation_fn=None, batch_normalize=None, l2loss=None, learned_moments_update_rate=None, parameter_modifier=None, phase=None, scale_after_normalization=None, summary_collections=None, trainable_variables=None, unroll=None, variable_collections=None, variance_epsilon=None)

Creates a scope for the defaults that are used in a `with` block.

  Note: `defaults_scope` supports nesting where later defaults can be
  overridden. Also, an explicitly given keyword argument on a method always
  takes precedence.

  In addition to setting defaults for some methods, this also can control:

  * `summary_collections`: Choose which collection to place summaries in or
      disable with `None`.
  * `trainable_variables`: Boolean indicating if variables are trainable.
  * `variable_collections`: Default collections in which to place variables;
      `tf.GraphKeys.GLOBAL_VARIABLES` is always included.

  The supported defaults and methods that use them are:


* `activation_fn`:
    * [conv2d](PrettyTensor.md#conv2d)
    * [depthwise_conv2d](PrettyTensor.md#depthwise_conv2d)
    * [fully_connected](PrettyTensor.md#fully_connected)

* `batch_normalize`:
    * [conv2d](PrettyTensor.md#conv2d)
    * [depthwise_conv2d](PrettyTensor.md#depthwise_conv2d)

* `l2loss`:
    * [conv2d](PrettyTensor.md#conv2d)
    * [depthwise_conv2d](PrettyTensor.md#depthwise_conv2d)
    * [diagonal_matrix_mul](PrettyTensor.md#diagonal_matrix_mul)
    * [fully_connected](PrettyTensor.md#fully_connected)

* `learned_moments_update_rate`:
    * [batch_normalize](PrettyTensor.md#batch_normalize)

* `parameter_modifier`:
    * [conv2d](PrettyTensor.md#conv2d)
    * [depthwise_conv2d](PrettyTensor.md#depthwise_conv2d)
    * [softmax_classifier_with_sampled_loss](PrettyTensor.md#softmax_classifier_with_sampled_loss)
    * [softmax_classifier](PrettyTensor.md#softmax_classifier)
    * [diagonal_matrix_mul](PrettyTensor.md#diagonal_matrix_mul)
    * [fully_connected](PrettyTensor.md#fully_connected)
    * [lstm_cell](PrettyTensor.md#lstm_cell)
    * [sequence_lstm](PrettyTensor.md#sequence_lstm)
    * [gru_cell](PrettyTensor.md#gru_cell)
    * [sequence_gru](PrettyTensor.md#sequence_gru)
    * [embedding_lookup](PrettyTensor.md#embedding_lookup)

* `phase`:
    * [batch_normalize](PrettyTensor.md#batch_normalize)
    * [conv2d](PrettyTensor.md#conv2d)
    * [depthwise_conv2d](PrettyTensor.md#depthwise_conv2d)
    * [evaluate_precision_recall](PrettyTensor.md#evaluate_precision_recall)
    * [evaluate_classifier_fraction](PrettyTensor.md#evaluate_classifier_fraction)
    * [evaluate_classifier](PrettyTensor.md#evaluate_classifier)
    * [evaluate_classifier_fraction_sparse](PrettyTensor.md#evaluate_classifier_fraction_sparse)
    * [evaluate_classifier_sparse](PrettyTensor.md#evaluate_classifier_sparse)
    * [dropout](PrettyTensor.md#dropout)
    * [diagonal_matrix_mul](PrettyTensor.md#diagonal_matrix_mul)
    * [fully_connected](PrettyTensor.md#fully_connected)
    * [lstm_cell](PrettyTensor.md#lstm_cell)
    * [sequence_lstm](PrettyTensor.md#sequence_lstm)
    * [gru_cell](PrettyTensor.md#gru_cell)
    * [sequence_gru](PrettyTensor.md#sequence_gru)
    * [embedding_lookup](PrettyTensor.md#embedding_lookup)

* `scale_after_normalization`:
    * [batch_normalize](PrettyTensor.md#batch_normalize)

* `unroll`:
    * [cleave_sequence](PrettyTensor.md#cleave_sequence)

* `variance_epsilon`:
    * [batch_normalize](PrettyTensor.md#batch_normalize)

- - -

## <a name="global_step"></a>global_step()



Returns the global step variable.





- - -

## <a name="join_pretty_tensors"></a>join_pretty_tensors(output, join_function=None, name=join)



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

## <a name="make_template"></a>make_template(func)



Given an arbitrary function, wrap it so that it does parameter sharing.





- - -

## <a name="recurrent_state"></a>recurrent_state()




- - -

## <a name="set_recurrent_state_saver"></a>set_recurrent_state_saver()



Sets the state saver used for recurrent sequences.





- - -

## <a name="template"></a>template(books=None, optional=False)



Starts a Pretty Tensor graph template.

## Template Mode

Templates allow you to define a graph with some unknown
values. The most common use case is to leave the input undefined and then
define a graph normally. The variables are only defined once the first time
the graph is constructed.  For example:

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
values (e.g. train and test using different widths), then it will break. This
is because we guarantee variable reuse across instantiations.

    template = (pretty_tensor.template('input')
                .fully_connected(200, name='l1')
                .fully_connected(
                    pretty_tensor.UnboundVariable('width'), name='l2'))
    train_output = template.construct(input=train_data, width=200)

    # The following line will die because the shared parameter is the wrong
    # size.
    test_output = template.construct(input=test_data, width=100)


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

## <a name="with_update_ops"></a>with_update_ops()



Creates a new op that runs all of the required updates when train_op runs.

#### Args:


* train_op: An operation that will run every step, usually the result of an
 optimizer.

#### Returns:

A new op that returns the same value as train_op, but also runs the
    updaters.




- - -

## <a name="wrap"></a>wrap(books=None, tensor_shape=None)



Creates an input layer representing the given tensor.

#### Args:


* tensor: The tensor.
* books: The bookkeeper; this is usually not required unless you are building
 multiple `tf.Graphs.`
* tensor_shape: An optional shape that will be set on the Tensor or verified
 to match the tensor.

#### Returns:

A layer.




- - -

## <a name="wrap_sequence"></a>wrap_sequence(books=None, tensor_shape=None)



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
