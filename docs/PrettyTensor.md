<!-- This file was automatically generated. -->

# PrettyTensor

A PrettyTensor is a Tensor with a builder interface facade.

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

- - -

[TOC]


## <a name="add_loss"></a>add_loss(loss, name=None)



Adds a loss and returns a wrapper for that loss.





- - -

## <a name="apply"></a>apply(operation)



Applies the given operation to this before without adding any summaries.

#### Args:


* operation: An operation that takes a tensor and the supplied args.
 *op_args: Extra arguments for operation.
 **op_kwargs: Keyword arguments for the operation.

#### Returns:

A new layer with operation applied.




- - -

## <a name="apply_with_summary"></a>apply_with_summary(operation)



Applies the given operation to this and sets the new head.

#### Args:


* operation: An operation that takes a tensor and the supplied args.
 *op_args: Extra arguments for operation.
 **op_kwargs: Keyword arguments for the operation.

#### Returns:

A new layer with operation applied.




- - -

## <a name="as_layer"></a>as_layer()



Returns a PrettyTensor snapshotted to the current tensor or sequence.

The primary use case of this is to break out of a sequential.


#### Returns:

An immutable PrettyTensor.




- - -

## <a name="attach_template"></a>attach_template(_template, _key)



Attaches the template to this such that _key=this layer.

Note: names were chosen to avoid conflicts with any likely unbound_var keys.

#### Args:


* _template: The template to construct.
* _key: The key that this layer should replace.
 **unbound_var_values: The values for the unbound_vars.

#### Returns:

A new layer with operation applied.


#### Raises:


* ValueError: If _key is specified twice or there is a problem computing the
 template.


- - -

## <a name="average_pool"></a>average_pool(kernel, stride, edges=SAME, name=None)



Performs average pooling. The current head must be a rank 4 Tensor.

#### Args:


* kernel: The size of the patch for the pool, either an int or a length 1 or
 2 sequence (if length 1 or int, it is expanded).
* stride: The strides as a length 1, 2 or 4 sequence or an integer. If an
 int, length 1 or 2, the stride in the first and last dimensions are 1.
* edges: Either PAD_SAME' or PAD_VALID to control the padding.
* name: The name for this operation is also used to create/find the
 parameter variables.

#### Returns:

Handle to this layer.




- - -

## <a name="batch_normalize"></a>batch_normalize(name=None, learned_moments_update_rate=None, variance_epsilon=None, scale_after_normalization=None, phase=Phase.train)



Batch normalize this layer.

This only supports global batch normalization and it can be enabled for all
convolutional layers by setting the default 'batch_normalize' to True.
learned_moments_update_rate, variance_epsilon and scale_after_normalization
need to either be set here or be set in defaults as well.

#### Args:


* name: The name for this operation is also used to create/find the
 parameter variables.
* learned_moments_update_rate: Update rate for the learned moments.
* variance_epsilon: A float. A small float number to avoid dividing by 0.
* scale_after_normalization: A bool indicating whether the resulted tensor
 needs to be multiplied with gamma.
* phase: The phase of construction.

#### Returns:

Handle to the generated layer.




- - -

## <a name="binary_cross_entropy_with_logits"></a>binary_cross_entropy_with_logits(target, name=None, loss_weight=None, per_example_weights=None, per_output_weights=None)



Calculates the binary cross entropy of the input_layer vs inputs.

Expects unscaled logits. Do not pass in results of sigmoid operation.

#### Args:


* target: A Float or Double tensor containing class label probabilities. Note
 that binary cross entropy is equivalent to logistic loss.
* name: The optional name.
* loss_weight: A scalar multiplier for the loss.
* per_example_weights: A `Tensor` with a weight per example.
* per_output_weights: A weight `Tensor` that is the same shape as the
 input_layer that can be used to scale individual prediction losses. See
 `tf.tile` to turn a per-column weight vector into a `per_output_weights`
 `Tensor`.

#### Returns:

Binary cross entropy loss after sigmoid operation.


#### Raises:


* ValueError: if target is None or the type is not float or double.


- - -

## <a name="cleave_sequence"></a>cleave_sequence(unroll=None)



Cleaves a tensor into a sequence, this is the inverse of squash.

Recurrent methods unroll across an array of Tensors with each one being a
timestep.  This cleaves the first dim so that each it is an array of Tensors.
It is the inverse of squash_sequence.

#### Args:


* unroll: The number of time steps.

#### Returns:

A PrettyTensor containing an array of tensors.


#### Raises:


* ValueError: If unroll is not specified and it has no default or it is <= 0.


- - -

## <a name="concat"></a>concat(concat_dim, other_tensors)



Concatenates input PrettyTensor with other_tensors along the specified dim.

This adds the Pretty Tensor passed via input_layer to the front of the list of
tensors to concat.

#### Args:


* concat_dim: The dimension along which to concat.
* other_tensors: The tensors to concatenate with.

#### Returns:

A new PrettyTensor.




- - -

## <a name="conv2d"></a>conv2d(kernel, depth, activation_fn=None, stride=None, l2loss=None, init=None, stddev=None, bias=True, bias_init=<function zeros_initializer at 0x394f9b0>, edges=SAME, batch_normalize=False, name=None)



Adds a convolution to the stack of operations.

The current head must be a rank 4 Tensor.

#### Args:


* kernel: The size of the patch for the pool, either an int or a length 1 or
 2 sequence (if length 1 or int, it is expanded).
* depth: The depth of the new Tensor.
* activation_fn: A tuple of (activation_function, extra_parameters). Any
 function that takes a tensor as its first argument can be used. More
 common functions will have summaries added (e.g. relu).
* stride: The strides as a length 1, 2 or 4 sequence or an integer. If an
 int, length 1 or 2, the stride in the first and last dimensions are 1.
* l2loss: Set to a value greater than 0 to use L2 regularization to decay
 the weights.
* init: An optional initialization. If not specified, uses Xavier
 initialization.
* stddev: A standard deviation to use in parameter initialization.
* bias: Set to False to not have a bias.
* bias_init: An initializer for the bias or a Tensor.
* edges: Either SAME to use 0s for the out of bounds area or VALID to shrink
 the output size and only uses valid input pixels.
* batch_normalize: Set to True to batch_normalize this layer.
* name: The name for this operation is also used to create/find the
 parameter variables.

#### Returns:

Handle to the generated layer.


#### Raises:


* ValueError: If head is not a rank 4 tensor or the depth of the input
 (4th dim) is not known.


- - -

## <a name="cross_entropy"></a>cross_entropy(labels, name=None, loss_weight=None, per_example_weights=None)



Calculates the Cross Entropy of the input_layer vs inputs.

#### Args:


* labels: A Float or Double tensor containing the labels.
* name: The optional name.
* loss_weight: A weight to scale the loss. Used when there are multiple
 losses.
* per_example_weights: A weighting for each example.

#### Returns:

A loss.


#### Raises:


* ValueError: if labels is None or the type is not float or double.


- - -

## <a name="diagonal_matrix_mul"></a>diagonal_matrix_mul(init=None, stddev=None, l2loss=None)



Performs a diagonal matrix multiplication with a learned vector.

This creates the parameter vector.

#### Args:


* init: An optional initialization or Tensor. If not specified, uses Xavier
 initialization.
* stddev: A standard deviation to use in parameter initialization.
* l2loss: An l2 weight decay to apply.

#### Returns:

A Pretty Tensor handle to the layer.


#### Raises:


* ValueError: if the head_shape is not rank 2 or the number of input nodes
 (second dim) is not known.


- - -

## <a name="dropout"></a>dropout(keep_prob, phase=Phase.train, name=None)



Aplies dropout if this is in the train phase.





- - -

## <a name="embedding_lookup"></a>embedding_lookup(embedding_count, embedding_shape, name=None, init=None)



Looks up values in a learned embedding lookup.

`embedding_count` embedding tensors are created each with shape
`embedding_shape`. The values are by defaulted initialized with a standard
deviation of 1, but in some cases zero is a more appropropriate initial
value.  The embeddings themselves are learned through normal
backpropagation.

You can initialize these to a fixed embedding and follow with
stop_gradients() to use a previously learned embedding.

N.B. This uses  tf.nn.embedding_lookup under the hood, so by default the
lookup is id % embedding_count

#### Args:


* embedding_count: Number of items in the embedding.
* embedding_shape: Shape of each embedding.
* name: The name of this layer.
* init: tf.*Initializer to use for initializing the input or a Tensor.
 Defaults to a truncated normal.

#### Returns:

input_layer


#### Raises:


* ValueError: If head is not a rank 2 Tensor with second dim of 1.


- - -

## <a name="eval"></a>eval(feed_dict=None, session=None)



Evaluates this tensor in a `Session`.

Calling this method will execute all preceding operations that
produce the inputs needed for the operation that produces this
tensor.

*N.B.* Before invoking `Tensor.eval()`, its graph must have been
launched in a session, and either a default session must be
available, or `session` must be specified explicitly.

#### Args:


* feed_dict: A dictionary that maps `Tensor` objects to feed values.
 See [`Session.run()`](../../api_docs/python/client.md#Session.run) for a
 description of the valid feed values.
* session: (Optional.) The `Session` to be used to evaluate this tensor. If
 none, the default session will be used.

#### Returns:

A numpy array corresponding to the value of this tensor.




- - -

## <a name="evaluate_classifier"></a>evaluate_classifier(labels, per_example_weights=None, topk=1, name=None, phase=Phase.train)



Calculates the total ratio of correct predictions across all examples seen.

In test and infer mode, this creates variables in the graph collection
pt.GraphKeys.TEST_VARIABLES and does not add them to
tf.GraphKeys.ALL_VARIABLES.  This means that you must initialize them
separately from tf.initialize_all_variables().

In the case of `topk == 1`, this breaks ties left-to-right, in all other cases
it follows `tf.nn.in_top_k`. *Note*: the tie behavior will change in the
future.

#### Args:


* labels: A float or double tensor containing the target for this layer.
* per_example_weights: Weights that are applied to every example.
* topk: Integer k for 'accuracy at top k' metric.
* name: The name of this layer.
* phase: In training mode the batch accuracy is returned and in eval/infer
 modes a total average is calculated.

#### Returns:

A Pretty Tensor with the ratio of correct to total examples seen.




- - -

## <a name="evaluate_precision_recall"></a>evaluate_precision_recall(labels, threshold=0.5, per_example_weights=None, name=None, phase=Phase.train)



Computes the precision and recall of the prediction vs the labels.

#### Args:


* labels: The target labels to learn as a float tensor.
* threshold: The threshold to use to decide if the prediction is true.
* per_example_weights: A Tensor with a weight per example.
* name: An optional name.
* phase: The phase of this model; non training phases compute a total across
 all examples.

#### Returns:

Precision and Recall.




- - -

## <a name="flatten"></a>flatten(preserve_batch=True)



Flattens this.

If preserve_batch is True, the result is rank 2 and the first dim (batch) is
unchanged. Otherwise the result is rank 1.

#### Args:


* preserve_batch: If True (the default), then preserve the first dimension.

#### Returns:

A LayerWrapper with the flattened tensor.




- - -

## <a name="fully_connected"></a>fully_connected(size, activation_fn=None, l2loss=None, init=None, stddev=None, bias=True, bias_init=<function zeros_initializer at 0x394f9b0>, transpose_weights=False, name=None)



Adds the parameters for a fully connected layer and returns a tensor.

The current head must be a rank 2 Tensor.

#### Args:


* size: The number of neurons
* activation_fn: A tuple of (activation_function, extra_parameters). Any
 function that takes a tensor as its first argument can be used. More
 common functions will have summaries added (e.g. relu).
* l2loss: Set to a value greater than 0 to use L2 regularization to decay
 the weights.
* init: An optional initialization. If not specified, uses Xavier
 initialization.
* stddev: A standard deviation to use in parameter initialization.
* bias: Set to False to not have a bias.
* bias_init: The initializer for the bias or a Tensor.
* transpose_weights: Flag indicating if weights should be transposed;
 this is useful for loading models with a different shape.
* name: The name for this operation is also used to create/find the
 parameter variables.

#### Returns:

A Pretty Tensor handle to the layer.


#### Raises:


* ValueError: if the head_shape is not rank 2 or the number of input nodes
 (second dim) is not known.


- - -

## <a name="get_shape"></a>get_shape()




- - -

## <a name="gru_cell"></a>gru_cell(state, num_units, bias=True, stddev=None, init=None)



Gated recurrent unit memory cell (GRU).

#### Args:


* state: The current state of the network. For GRUs, this is a list with
 one element (tensor) of shape [batch, num_units].
* num_units: How big is the hidden state.
* bias: Whether or not to use a bias.
* stddev: Standard deviation for Gaussian initialization of parameters.
* init: A tf.*Initializer that is used to initialize the variables.

#### Returns:

A RecurrentResult.




- - -

## <a name="is_sequence"></a>is_sequence()



Returns True if this holds a sequence and False if it holds a Tensor.





- - -

## <a name="is_sequential_builder"></a>is_sequential_builder()



Returns true if this is a sequential builder.

NB: A sequential builder is a mode of construction and is different from
whether or not this holds a sequence of tensors.


#### Returns:

Whether this is a sequential builder.




- - -

## <a name="join"></a>join(others, include_self=True, join_function=None)



Joins the provided PrettyTensors with this using the join function.

#### Args:


* others: Sequence of PrettyTensor objects.
* include_self: Whether or not this includes itself or if the value is only
 derived from others.
* join_function: The function to use for joining, must accept a list of
 tensors. Use None for concat on the final dimension.

#### Returns:

self.




- - -

## <a name="l1_regression"></a>l1_regression(target, name=None, loss_weight=None, per_example_weights=None)



Applies an L1 Regression (Sum of Absolute Error) to the target.





- - -

## <a name="l2_regression"></a>l2_regression(target, name=None, loss_weight=None, per_example_weights=None)



Applies an L2 Regression (Sum of Squared Error) to the target.





- - -

## <a name="lstm_cell"></a>lstm_cell(states, num_units, bias=True, peephole=True, stddev=None, init=None)



Long short-term memory cell (LSTM).

#### Args:


* states: The current state of the network, as
 [[batch, num_units], [batch, num_units]] (c, h).
* num_units: How big is the hidden state.
* bias: Whether or not to use a bias.
* peephole: Whether to use peephole connections as described in
 http://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf
* stddev: Standard deviation for Gaussian initialization of parameters.
* init: A tf.*Initializer that is used to initialize the variables.

#### Returns:

A RecurrentResult.




- - -

## <a name="map"></a>map(fn)



Maps the given function across this sequence.

To map an entire template across the sequence, use the `as_fn` method on the
template.

#### Args:


* fn: A function of 1 argument that is applied to each item in the sequence.

#### Returns:

A new sequence Pretty Tensor.


#### Raises:


* ValueError: If the input_layer does not hold a sequence.


- - -

## <a name="max_pool"></a>max_pool(kernel, stride, edges=SAME, name=None)



Performs max pooling. The current head must be a rank 4 Tensor.

#### Args:


* kernel: The size of the patch for the pool, either an int or a length 1 or
 2 sequence (if length 1 or int, it is expanded).
* stride: The strides as a length 1, 2 or 4 sequence or an integer. If an
 int, length 1 or 2, the stride in the first and last dimensions are 1.
* edges: Either PAD_SAME or PAD_VALID to control the padding.
* name: The name for this operation is also used to create/find the
 parameter variables.

#### Returns:

Handle to this layer.




- - -

## <a name="reshape"></a>reshape(shape_spec)



Reshapes this tensor to the given spec.

A shape_spec can be a list or tuple of numbers specifying the new shape, but
also may include the following shorthands for using values from the shape of
the input:

1. DIM_SAME ('_') will use the corresponding value from the current shape.
2. One -1 or DIM_REST ('*') can be used to specify the remainder of the
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

#### Args:


* shape_spec: The spec for the new shape.

#### Returns:

A Pretty Tensor with the reshaped tensor.


#### Raises:


* ValueError: If there are two many unknown dimensions or the shape_spec
* requires out of range DIM_SAME.


- - -

## <a name="sequence_gru"></a>sequence_gru(num_units, bias=True, name=None, stddev=None, init=None, lengths=None)



Creates an unrolled GRU to process sequence data.

The initial state is drawn from the bookkeeper's recurrent state and if it
supports state saving, then it is saved.

#### Args:


* num_units: Number of units in the hidden states.
* bias: Whether or not to use a bias.
* name: The name of this layer.
* stddev: Standard deviation for Gaussian initialization of parameters.
* init: A tf.*Initializer that is used to initialize the variables.
* lengths: An optional Tensor that encodes a length for each item in the
 minibatch. This is used to truncate computation.

#### Returns:

A sequence with the result at each timestep.


#### Raises:


* ValueError: if head is not a sequence, the shape is not rank 2 or the
 number of nodes (second dim) is not known.


- - -

## <a name="sequence_lstm"></a>sequence_lstm(num_units, bias=True, peephole=True, name=None, stddev=None, init=None, lengths=None)



Creates an unrolled LSTM to process sequence data.

The initial state is drawn from the bookkeeper's recurrent state and if it
supports state saving, then it is saved.

#### Args:


* num_units: Number of nodes in the hidden states and the output size.
* bias: Whether or not to use a bias.
* peephole: Whether to use peephole connections.
* name: The name of this layer.
* stddev: Standard deviation for Gaussian initialization of parameters.
* init: A tf.*Initializer that is used to initialize the variables.
* lengths: An optional Tensor that encodes a length for each item in the
 minibatch. This is used to truncate computation.

#### Returns:

A sequence with the result at each timestep.


#### Raises:


* ValueError: if head is not a sequence, the shape is not rank 2 or the
 number of nodes (second dim) is not known.


- - -

## <a name="slice"></a>slice(begin, size)



Extracts a slice from a tensor.

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

#### Args:


* begin: An int32 or int64 Tensor of length rank(input_layer)
* size: An int32 or int64 Tensor of length rank(input_layer)

#### Returns:

A tensor with the selected slice.




- - -

## <a name="softmax"></a>softmax(labels=None, name=None, loss_weight=None, per_example_weights=None)



Applies softmax and if labels is not None, then it also adds a loss.

#### Args:


* labels: The target labels to learn as a float tensor. Use None to not
 include a training loss.
* name: The optional name.
* loss_weight: A scalar multiplier for the loss.
* per_example_weights: A Tensor with a weight per example.

#### Returns:

A tuple of the a handle to softmax and a handle to the loss tensor.


#### Raises:


* ValueError: If the datatype is wrong.


- - -

## <a name="softmax_activation"></a>softmax_activation()



Computes the softmax.

#### Args:


* Returns:
 A new Pretty Tensor with the softmax applied.





- - -

## <a name="softmax_classifier"></a>softmax_classifier(class_count, labels=None, name=None, loss_weight=None, per_example_weights=None, weight_init=None, bias_init=<function zeros_initializer at 0x394f9b0>)



Creates a fully-connected linear layer followed by a softmax.

#### Args:


* class_count: The number of classes.
* labels: The target labels to learn as a float tensor. Use None to not
 include a training loss.
* name: The optional name.
* loss_weight: A scalar multiplier for the loss.
* per_example_weights: A Tensor with a weight per example.
* weight_init: The initializer for the weights (see `fully_connected`).
* bias_init: The initializer for the bias (see `fully_connected`).

#### Returns:

A tuple of the softmax's name and the loss tensor's name in m.bits.


#### Raises:


* ValueError: If the datatype is wrong.


- - -

## <a name="softmax_classifier_with_sampled_loss"></a>softmax_classifier_with_sampled_loss(num_classes, labels, num_sampled, num_true=None, sampled_values=None, remove_accidental_hits=True, loss_weight=None, per_example_weights=None, weight_init=None, bias_init=<function zeros_initializer at 0x394f9b0>, name=softmax_classifier)



Applies softmax and if labels is not None, then it adds a sampled loss.

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

#### Args:


* activations of the input network.
 num_classes: An `int`. The number of possible classes.
 labels: A `Tensor` of type `int64` and shape `[batch_size,
* num_true]`. The target classes. Note that this format differs from
* the `labels` argument of `nn.softmax_cross_entropy_with_logits`.
 num_sampled: An `int`. The number of classes to randomly sample per batch.
 num_true: An `int`. The number of target classes per training example,
 defaults to the second dim of labels if known or 1.
 sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
 `sampled_expected_count`) returned by a `*_candidate_sampler` function.
 (if None, we default to `log_uniform_candidate_sampler`)
 remove_accidental_hits: A `bool`. whether to remove "accidental hits"
* where a sampled class equals one of the target classes. Default is
* True.
 loss_weight: A scalar multiplier for the loss.
 per_example_weights: A Tensor with a weight per example.
 weight_init: The initializer for the weights (see `fully_connected`). Note:
 This is the transpose of a normal fully_connected input layer!
 bias_init: The initializer for the bias (see `fully_connected`).
 name: The optional name.

#### Returns:

A tuple of handles to the logits (fully connected layer) and loss.


#### Raises:


* ValueError: If inputs or labels do not have the right shape.


- - -

## <a name="split"></a>split(split_dim=0, num_splits=2)



Splits the head Tensor along the split_dim into num_splits Equal chunks.

Examples:

* `[1, 2, 3, 4] -> [1, 2], [3, 4]`
* `[[1, 1], [2, 2], [3, 3], [4, 4]] -> [[1, 1], [2, 2]], [[3, 3], [4, 4]]`

#### Args:


* split_dim: The dimension to split along. Defaults to batch.
* num_splits: The number of splits.

#### Returns:

A list of PrettyTensors.


#### Raises:


* ValueError: If split_dim is out of range or isn't divided evenly by
 num_splits.


- - -

## <a name="squash_sequence"></a>squash_sequence()



"Squashes a sequence into a single Tensor with dim 1 being time*batch.

A sequence is an array of Tensors, which is not appropriate for most
operations, this squashes them together into Tensor.

Defaults are assigned such that cleave_sequence requires no args.

#### Args:


* Returns:
 A PrettyTensor containing a single tensor with the first dim containing
 both time and batch.



#### Raises:


* ValueError: If the sequence is empty.


- - -

## <a name="squeeze"></a>squeeze(squeeze_dims=None)



Removes dimensions of size 1 from the shape of a tensor.

This operation returns a tensor of the same type with all singleton
dimensions removed. If you don't want to remove all singleton dimensions, you
can remove specific size 1 dimensions by specifying a list of squeeze_dims.

#### Args:


* squeeze_dims: An optional list of ints. Defaults to [].

#### Returns:

The sequeezed tensor.




- - -

## <a name="stop_gradient"></a>stop_gradient()



Cuts off the gradient at this point.

This works on both sequence and regular Pretty Tensors.

#### Args:


* Returns:
 A new Pretty Tensor of the same type with stop_gradient applied.





- - -

## <a name="to_dense_one_hot"></a>to_dense_one_hot(class_count)



Converts a vector that specified one-hot per batch into a dense version.

#### Args:


* class_count: The number of classes as an int.

#### Returns:

One dense vector for each item in the batch.


#### Raises:


* ValueError: If labels is not rank 1.
* TypeError: If class_count is not an integer or labels is not an integer
 Tensor.


- - -

## <a name="unzip"></a>unzip(split_dim=0, num_splits=2)



Unzips the head Tensor along the split_dim into num_splits Equal chunks.

Examples:

* `[1, 2, 3, 4] -> [1, 3], [2, 4]`
* `[[1, 1], [2, 2], [3, 3], [4, 4]] -> [[1, 1], [3, 3]], [[2, 2], [4, 4]]`

#### Args:


* split_dim: The dimension to split along. Defaults to batch.
* num_splits: The number of splits.

#### Returns:

A list of PrettyTensors.


#### Raises:


* ValueError: If split_dim is out of range or isn't divided evenly by
 num_splits.


- - -

## <a name="with_defaults"></a>with_defaults(...

defaults_scope(activation_fn=None, batch_normalize=None, l2loss=None, learned_moments_update_rate=None, phase=None, scale_after_normalization=None, stddev=None, summary_collections=None, trainable_variables=None, unroll=None, variable_collections=None, variance_epsilon=None)

Creates a scope for the defaults that are used in a `with` block.

  In addition to setting defaults for some methods, this also can control:

  * `summary_collections`: Choose which collection to place summaries in or
      disable with `None`.
  * `trainable_variables`: Boolean indicating if variables are trainable.
  * `variable_collections`: Default collections in which to place variables;
      `tf.GraphKeys.VARIABLES` is always included.

  The supported defaults and methods that use them are:


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
    * [evaluate_precision_recall](PrettyTensor.md#evaluate_precision_recall)
    * [evaluate_classifier](PrettyTensor.md#evaluate_classifier)
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

## <a name="with_name"></a>with_name(name)



Sets the name scope for future operations.





- - -

## <a name="with_sequence"></a>with_sequence(sequence, parameters=None)



Returns a PrettyTensor that points to sequence.





- - -

## <a name="with_tensor"></a>with_tensor(tensor, parameters=None)



Returns a PrettyTensor that points to tensor.





- - -
