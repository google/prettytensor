<!-- This file was automatically generated. -->

# PrettyTensor

A PrettyTensor is a wrapper on a Tensor that simplifies graph building.

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

- - -

[TOC]


## <a name="abs"></a>abs(name=None)



Computes the absolute value of a tensor.

Given a tensor of real numbers `x`, this operation returns a tensor
containing the absolute value of each element in `x`. For example, if x is
an input element and y is an output element, this operation computes
\\(y = |x|\\).

#### Args:


* name: A name for the operation (optional).

#### Returns:

A `Tensor` or `SparseTensor` the same size and type as `x` with absolute
      values.




- - -

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



Applies the given operation to `input_layer` and create a summary.

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



Performs average pooling.

`kernel` is the patch that will be pooled and it describes the pooling along
each of the 4 dimensions.  `stride` is how big to take each step.

Because more often than not, pooling is only done
on the width and height of the image, the following shorthands are supported:

* scalar (e.g. 3): Square pooling on the image
    (`[b, c, r, d] = [1, 3, 3, 1]`).
* singleton list (e.g. [3]): Square pooling on the image
    (`[b, c, r, d] = [1, 3, 3, 1]`).
* list of length 2 (e.g. [3, 2]): Square pooling on the image
    (`[b, c, r, d] = [1, 3, 2, 1]`).

#### Args:


* kernel: The size of the patch for the pool, either an int or a length 1 or
 2 sequence (if length 1 or int, it is expanded).
* stride: The strides as a length 1, 2 or 4 sequence or an integer. If an
 int, length 1 or 2, the stride in the first and last dimensions are 1.
* edges: Either `pt.PAD_SAME`' or `pt.PAD_VALID` to control the padding.
* name: The name for this operation is also used to create/find the
 parameter variables.

#### Returns:

Handle to this layer.




- - -

## <a name="batch_normalize"></a>batch_normalize(name=None, learned_moments_update_rate=0.0003, variance_epsilon=0.001, scale_after_normalization=False, phase=train)



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

## <a name="bilinear_sampling"></a>bilinear_sampling(x, y, name=None)



Performs bilinear sampling. This must be a rank 4 Tensor.

Implements the differentiable sampling mechanism with bilinear kernel
in https://arxiv.org/abs/1506.02025.

Given (x, y) coordinates for each output pixel, use bilinear sampling on
the input_layer to fill the output.

#### Args:


* x: A tensor of size [batch_size, height, width, 1] representing the sampling
 x coordinates normalized to range [-1,1].
* y: A tensor of size [batch_size, height, width, 1] representing the
 sampling y coordinates normalized to range [-1,1].
* name: The name for this operation is also used to create/find the
 parameter variables.

#### Returns:

Handle to this layer




- - -

## <a name="binary_cross_entropy_with_logits"></a>binary_cross_entropy_with_logits(target, name=None, loss_weight=None, per_example_weights=None, per_output_weights=None)



Calculates the binary cross entropy of the input_ vs inputs.

Expects unscaled logits. Do not pass in results of sigmoid operation.

#### Args:


* target: A rank 2 tf.float32 or tf.float64 tensor containing class label
 probabilities. Note that binary cross entropy is equivalent to logistic
 loss.
* name: The optional name.
* loss_weight: A scalar multiplier for the loss.
* per_example_weights: A `Tensor` with a weight per example.
* per_output_weights: A weight `Tensor` that is the same shape as the
 input_ that can be used to scale individual prediction losses. See
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

## <a name="concat"></a>concat(concat_dim, other_tensors=None)



Concatenates input PrettyTensor with other_tensors along the specified dim.

This adds the Pretty Tensor passed via input_layer to the front of the list of
tensors to concat.

#### Args:


* concat_dim: The dimension along which to concat.
* other_tensors: The tensors to concatenate with as an iterable or None if
 this is called on a sequence.

#### Returns:

A new PrettyTensor.


#### Raises:


* ValueError: If other_tensors is None and this is not a sequence.


- - -

## <a name="conv2d"></a>conv2d(kernel, depth, activation_fn=None, stride=(1, 1), l2loss=None, weights=None, bias=<google3.third_party.tensorflow.python.ops.init_ops.Zeros object at 0x3cf89d0>, edges=SAME, batch_normalize=False, phase=train, parameter_modifier=<function identity at 0x402faa0>, name=None)



Adds a convolution to the stack of operations.

`kernel` is the patch that will be pooled and it describes the pooling
along each of the 4 dimensions.  The stride is how big to take each step.

* scalar (e.g. 3): Square pooling on the image
    (`[b, c, r, d] = [1, 3, 3, 1]`).
* singleton list (e.g. [3]): Square pooling on the image
    (`[b, c, r, d] = [1, 3, 3, 1]`).
* list of length 2 (e.g. [3, 2]): Square pooling on the image
    (`[b, c, r, d] = [1, 3, 2, 1]`).

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
* weights: An initializer for weights or a Tensor. If not specified,
 uses He's initialization.
* bias: An initializer for the bias or a Tensor. No bias if set to None.
* edges: Either SAME to use 0s for the out of bounds area or VALID to shrink
 the output size and only uses valid input pixels.
* batch_normalize: Supply a BatchNormalizationArguments to set the
 parameters for batch normalization.
* phase: The phase of graph construction. See `pt.Phase`.
* parameter_modifier: A function to modify parameters that is applied after
 creation and before use.
* name: The name for this operation is also used to create/find the
 parameter variables.

#### Returns:

Handle to the generated layer.


#### Raises:


* ValueError: If input_layer is not a rank 4 tensor or the depth of the
 input (4th dim) is not known.


- - -

## <a name="cross_entropy"></a>cross_entropy(labels, name=None, loss_weight=None, per_example_weights=None)



Calculates the Cross Entropy of input_ vs labels.

#### Args:


* labels: A rank 2 tf.float32 or tf.float64 tensor containing the labels.
* name: The optional name.
* loss_weight: A weight to scale the loss. Used when there are multiple
 losses.
* per_example_weights: A weighting for each example.

#### Returns:

A loss.


#### Raises:


* ValueError: if labels is None or the type is not float or double.


- - -

## <a name="depthwise_conv2d"></a>depthwise_conv2d(kernel, channel_multiplier, activation_fn=None, stride=None, l2loss=None, weights=None, bias=<google3.third_party.tensorflow.python.ops.init_ops.Zeros object at 0x3cf8a90>, edges=SAME, batch_normalize=False, phase=train, parameter_modifier=<function identity at 0x402faa0>, name=None)



Adds a depth-wise convolution to the stack of operations.

A depthwise convolution performs the convolutions one channel at a time and
produces an output with depth `channel_multiplier * input_depth`.

`kernel` is the patch that will be pooled and it describes the pooling
along each of the 4 dimensions.  The stride is how big to take each step.

* scalar (e.g. 3): Square pooling on the image
    (`[b, c, r, d] = [1, 3, 3, 1]`).
* singleton list (e.g. [3]): Square pooling on the image
    (`[b, c, r, d] = [1, 3, 3, 1]`).
* list of length 2 (e.g. [3, 2]): Square pooling on the image
    (`[b, c, r, d] = [1, 3, 2, 1]`).

#### Args:


* kernel: The size of the patch for the pool, either an int or a length 1 or
 2 sequence (if length 1 or int, it is expanded).
* channel_multiplier: Output channels will be a multiple of input channels.
* activation_fn: A tuple of (activation_function, extra_parameters). Any
 function that takes a tensor as its first argument can be used. More
 common functions will have summaries added (e.g. relu).
* stride: The strides as a length 1, 2 or 4 sequence or an integer. If an
 int, length 1 or 2, the stride in the first and last dimensions are 1.
* l2loss: Set to a value greater than 0 to use L2 regularization to decay
 the weights.
* weights: An initializer for weights or a Tensor. If not specified,
 uses He's initialization.
* bias: An initializer for the bias or a Tensor. No bias if set to None.
* edges: Either `pt.DIM_SAME` to use 0s for the out of bounds area or
 `pt.DIM_VALID` to shrink the output size and only uses valid input
 pixels.
* batch_normalize: Supply a BatchNormalizationArguments to set the
 parameters for batch normalization.
* phase: The phase of graph construction. See `pt.Phase`.
* parameter_modifier: A function to modify parameters that is applied after
 creation and before use.
* name: The name for this operation is also used to create/find the
 parameter variables.

#### Returns:

Handle to the generated layer.


#### Raises:


* ValueError: If input_layer is not a rank 4 tensor or the depth of the
 input (4th dim) is not known.


- - -

## <a name="diagonal_matrix_mul"></a>diagonal_matrix_mul(weights=None, l2loss=None, phase=train, parameter_modifier=<function identity at 0x402faa0>)



Performs a diagonal matrix multiplication with a learned vector.

This creates the parameter vector.

#### Args:


* weights: An initializer for weights or a Tensor. If not specified,
 uses Xavier initialization.
* l2loss: An l2 weight decay to apply.
* phase: The phase of graph construction. See `pt.Phase`.
* parameter_modifier: A function to modify parameters that is applied after
 creation and before use.

#### Returns:

A Pretty Tensor handle to the layer.


#### Raises:


* ValueError: if this is not rank 2 or the number of input nodes
 (second dim) is not known.


- - -

## <a name="dropout"></a>dropout(keep_prob, phase=train, name=None)



Aplies dropout if this is in the train phase.





- - -

## <a name="embedding_lookup"></a>embedding_lookup(embedding_count, embedding_shape, weights=None, phase=train, parameter_modifier=<function identity at 0x402faa0>, name=None)



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
* weights: tf.*Initializer to use for initializing the input or a Tensor.
 Defaults to a truncated normal.
* phase: The phase of graph construction. See `pt.Phase`.
* parameter_modifier: A function to modify parameters that is applied after
 creation and before use.
* name: The name of this layer.

#### Returns:

input_layer


#### Raises:


* ValueError: If input_layer is not a rank 2 Tensor with second dim of 1.


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

## <a name="evaluate_classifier"></a>evaluate_classifier(labels, per_example_weights=None, topk=1, name=None, phase=train)



Calculates the total ratio of correct predictions across all examples seen.

In test and infer mode, this creates variables in the graph collection
pt.GraphKeys.TEST_VARIABLES and does not add them to
tf.GraphKeys.ALL_VARIABLES.  This means that you must initialize them
separately from tf.global_variables_initializer().

In the case of `topk == 1`, this breaks ties left-to-right, in all other cases
it follows `tf.nn.in_top_k`. *Note*: the tie behavior will change in the
future.

#### Args:


* labels: A float or double `Tensor` containing the target for this layer.
* per_example_weights: Weights that are applied to every example.
* topk: Integer k for 'accuracy at top k' metric.
* name: The name of this layer.
* phase: In training mode the batch accuracy is returned and in eval/infer
 modes a total average is calculated.

#### Returns:

A Pretty Tensor with the ratio of correct to total examples seen.


#### Raises:


* ValueError: If labels is not the correct shape.


- - -

## <a name="evaluate_classifier_fraction"></a>evaluate_classifier_fraction(labels, per_example_weights=None, topk=1, name=None, phase=train)



Calculates the total of correct predictions and example count.

In test and infer mode, this creates variables in the graph collection
pt.GraphKeys.TEST_VARIABLES and does not add them to
tf.GraphKeys.ALL_VARIABLES.  This means that you must initialize them
separately from tf.global_variables_initializer().

In the case of `topk == 1`, this breaks ties left-to-right, in all other cases
it follows `tf.nn.in_top_k`. *Note*: the tie behavior will change in the
future.

#### Args:


* labels: A float or double `Tensor` containing the target for this layer *or*
 an integer `Tensor` with the sparse one-hot indices.
* per_example_weights: Weights that are applied to every example.
* topk: Integer k for 'accuracy at top k' metric.
* name: The name of this layer.
* phase: In training mode the batch accuracy is returned and in eval/infer
 modes a total average is calculated.

#### Returns:

A Pretty Tensor that contains correct_predictions, num_examples.


#### Raises:


* ValueError: If labels is not the correct shape.


- - -

## <a name="evaluate_classifier_fraction_sparse"></a>evaluate_classifier_fraction_sparse(labels, per_example_weights=None, topk=1, name=None, phase=train)



Calculates the total of correct predictions and example count.

In test and infer mode, this creates variables in the graph collection
pt.GraphKeys.TEST_VARIABLES and does not add them to
tf.GraphKeys.ALL_VARIABLES.  This means that you must initialize them
separately from tf.global_variables_initializer().

This breaks ties left-to-right.

#### Args:


* labels: A float or double `Tensor` containing the target for this layer *or*
 an integer `Tensor` with the sparse one-hot indices.
* per_example_weights: Weights that are applied to every example.
* topk: Integer k for 'accuracy at top k' metric.
* name: The name of this layer.
* phase: In training mode the batch accuracy is returned and in eval/infer
 modes a total average is calculated.

#### Returns:

A Pretty Tensor that contains correct_predictions, num_examples.


#### Raises:


* ValueError: If labels is not the correct shape.


- - -

## <a name="evaluate_classifier_sparse"></a>evaluate_classifier_sparse(labels, per_example_weights=None, topk=1, name=None, phase=train)



Calculates the total ratio of correct predictions across all examples seen.

In test and infer mode, this creates variables in the graph collection
pt.GraphKeys.TEST_VARIABLES and does not add them to
tf.GraphKeys.ALL_VARIABLES.  This means that you must initialize them
separately from tf.global_variables_initializer().

This breaks ties left-to-right.

#### Args:


* labels: An integer `Tensor` with the sparse one-hot indices as
 [batch, num_true].
* per_example_weights: Weights that are applied to every example.
* topk: Integer k for 'accuracy at top k' metric.
* name: The name of this layer.
* phase: In training mode the batch accuracy is returned and in eval/infer
 modes a total average is calculated.

#### Returns:

A Pretty Tensor with the ratio of correct to total examples seen.


#### Raises:


* ValueError: If labels is not the correct shape.


- - -

## <a name="evaluate_precision_recall"></a>evaluate_precision_recall(labels, threshold=0.5, per_example_weights=None, name=None, phase=train)



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

## <a name="fully_connected"></a>fully_connected(size, activation_fn=None, l2loss=None, weights=None, bias=<google3.third_party.tensorflow.python.ops.init_ops.Zeros object at 0x3d05410>, transpose_weights=False, phase=train, parameter_modifier=<function identity at 0x402faa0>, name=None)



Adds the parameters for a fully connected layer and returns a tensor.

The current PrettyTensor must have rank 2.

#### Args:


* size: The number of neurons
* activation_fn: A tuple of (activation_function, extra_parameters). Any
 function that takes a tensor as its first argument can be used. More
 common functions will have summaries added (e.g. relu).
* l2loss: Set to a value greater than 0 to use L2 regularization to decay
 the weights.
* weights: An initializer for weights or a Tensor. If not specified,
 uses He's initialization.
* bias: An initializer for the bias or a Tensor. No bias if set to None.
* transpose_weights: Flag indicating if weights should be transposed;
 this is useful for loading models with a different shape.
* phase: The phase of graph construction. See `pt.Phase`.
* parameter_modifier: A function to modify parameters that is applied after
 creation and before use.
* name: The name for this operation is also used to create/find the
 parameter variables.

#### Returns:

A Pretty Tensor handle to the layer.


#### Raises:


* ValueError: if the Pretty Tensor is not rank 2 or the number of input
 nodes (second dim) is not known.


- - -

## <a name="get_shape"></a>get_shape()




- - -

## <a name="gru_cell"></a>gru_cell(state, num_units, bias=<google3.third_party.tensorflow.python.ops.init_ops.Zeros object at 0x3f4d8d0>, weights=None, phase=train, parameter_modifier=<function identity at 0x402faa0>)



Gated recurrent unit memory cell (GRU).

#### Args:


* state: The current state of the network. For GRUs, this is a list with
 one element (tensor) of shape [batch, num_units].
* num_units: How big is the hidden state.
* bias: An initializer for the bias or a Tensor. No bias if set to None.
* weights: An initializer for weights or a Tensor.
* phase: The phase of graph construction. See `pt.Phase`.
* parameter_modifier: A function to modify parameters that is applied after
 creation and before use.

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

## <a name="l1_normalize"></a>l1_normalize(dim, epsilon=1e-12, name=None)



l1 normalizes x.

#### Args:


* dim: The dimension to normalize along.
* epsilon: Lower bound on the norm, used to avoid exploding gradients as the
 norm approaches 0.
* name: Optional name for this op.

#### Returns:

x normalized along dim.




- - -

## <a name="l1_regression"></a>l1_regression(target, name=None, loss_weight=None, per_example_weights=None)



Applies an L1 Regression (Sum of Absolute Error) to the target.





- - -

## <a name="l2_normalize"></a>l2_normalize(dim, epsilon=1e-12, name=None)



Normalizes along dimension `dim` using an L2 norm.

For a 1-D tensor with `dim = 0`, computes

    output = x / sqrt(max(sum(x**2), epsilon))

For `x` with more dimensions, independently normalizes each 1-D slice along
dimension `dim`.

#### Args:


* dim: Dimension along which to normalize. A scalar or a vector of
 integers.
* epsilon: A lower bound value for the norm. Will use `sqrt(epsilon)` as the
 divisor if `norm < sqrt(epsilon)`.
* name: A name for this operation (optional).

#### Returns:

A `Tensor` with the same shape as `x`.




- - -

## <a name="l2_regression"></a>l2_regression(target, name=None, loss_weight=None, per_example_weights=None)



Applies an L2 Regression (Sum of Squared Error) to the target.





- - -

## <a name="leaky_relu"></a>leaky_relu(name=None)



Creates a leaky_relu.

This is an alternate non-linearity to relu. The leaky part of the relu may
prevent dead Neurons in a model since the gradient doesn't go completely to
0.

#### Args:


* name: Optional name for this op.

#### Returns:

x if x > 0 otherwise 0.01 * x.




- - -

## <a name="log"></a>log(name=None)



Computes natural logarithm of x element-wise.

I.e., \\(y = \log_e x\\).

#### Args:


* name: A name for the operation (optional).

#### Returns:

A `Tensor`. Has the same type as `x`.




- - -

## <a name="lstm_cell"></a>lstm_cell(states, num_units, bias=<google3.third_party.tensorflow.python.ops.init_ops.Zeros object at 0x3f4d6d0>, peephole=True, weights=None, phase=train, parameter_modifier=<function identity at 0x402faa0>)



Long short-term memory cell (LSTM).

#### Args:


* states: The current state of the network, as
 [[batch, num_units], [batch, num_units]] (c, h).
* num_units: How big is the hidden state.
* bias: An initializer for the bias or a Tensor. No bias if set to None.
* peephole: Whether to use peephole connections as described in
 http://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf
* weights: An initializer for weights or a Tensor.
* phase: The phase of graph construction. See `pt.Phase`.
* parameter_modifier: A function to modify parameters that is applied after
 creation and before use.

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



Performs max pooling.

`kernel` is the patch that will be pooled and it describes the pooling along
each of the 4 dimensions.  `stride` is how big to take each step.

Because more often than not, pooling is only done
on the width and height of the image, the following shorthands are supported:

* scalar (e.g. 3): Square pooling on the image
    (`[b, c, r, d] = [1, 3, 3, 1]`).
* singleton list (e.g. [3]): Square pooling on the image
    (`[b, c, r, d] = [1, 3, 3, 1]`).
* list of length 2 (e.g. [3, 2]): Square pooling on the image
    (`[b, c, r, d] = [1, 3, 2, 1]`).

#### Args:


* kernel: The size of the patch for the pool, either an int or a length 1 or
 2 sequence (if length 1 or int, it is expanded).
* stride: The strides as a length 1, 2 or 4 sequence or an integer. If an
 int, length 1 or 2, the stride in the first and last dimensions are 1.
* edges: Either `pt.PAD_SAME` or `pt.PAD_VALID` to control the padding.
* name: The name for this operation is also used to create/find the
 parameter variables.

#### Returns:

Handle to this layer.




- - -

## <a name="reduce_all"></a>reduce_all(axis=None, keep_dims=False, name=None, reduction_indices=None)



Computes the "logical and" of elements across dimensions of a tensor.

Reduces `input_tensor` along the dimensions given in `axis`.
Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
entry in `axis`. If `keep_dims` is true, the reduced dimensions
are retained with length 1.

If `axis` has no entries, all dimensions are reduced, and a
tensor with a single element is returned.

For example:

```python
# 'x' is [[True,  True]
#         [False, False]]
tf.reduce_all(x) ==> False
tf.reduce_all(x, 0) ==> [False, False]
tf.reduce_all(x, 1) ==> [True, False]
```

#### Args:


* axis: The dimensions to reduce. If `None` (the default),
 reduces all dimensions.
* keep_dims: If true, retains reduced dimensions with length 1.
* name: A name for the operation (optional).
* reduction_indices: The old (deprecated) name for axis.

#### Returns:

The reduced tensor.

  @compatibility(numpy)
  Equivalent to np.all
  @end_compatibility




- - -

## <a name="reduce_any"></a>reduce_any(axis=None, keep_dims=False, name=None, reduction_indices=None)



Computes the "logical or" of elements across dimensions of a tensor.

Reduces `input_tensor` along the dimensions given in `axis`.
Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
entry in `axis`. If `keep_dims` is true, the reduced dimensions
are retained with length 1.

If `axis` has no entries, all dimensions are reduced, and a
tensor with a single element is returned.

For example:

```python
# 'x' is [[True,  True]
#         [False, False]]
tf.reduce_any(x) ==> True
tf.reduce_any(x, 0) ==> [True, True]
tf.reduce_any(x, 1) ==> [True, False]
```

#### Args:


* axis: The dimensions to reduce. If `None` (the default),
 reduces all dimensions.
* keep_dims: If true, retains reduced dimensions with length 1.
* name: A name for the operation (optional).
* reduction_indices: The old (deprecated) name for axis.

#### Returns:

The reduced tensor.

  @compatibility(numpy)
  Equivalent to np.any
  @end_compatibility




- - -

## <a name="reduce_join"></a>reduce_join(axis=None, keep_dims=False, separator=, name=None, reduction_indices=None)



Joins a string Tensor across the given dimensions.

Computes the string join across dimensions in the given string Tensor of shape
`[d_0, d_1, ..., d_n-1]`.  Returns a new Tensor created by joining the input
strings with the given separator (default: empty string).  Negative indices are
counted backwards from the end, with `-1` being equivalent to `n - 1`.

For example:

```
# tensor `a` is [["a", "b"], ["c", "d"]]
tf.reduce_join(a, 0) ==> ["ac", "bd"]
tf.reduce_join(a, 1) ==> ["ab", "cd"]
tf.reduce_join(a, -2) = tf.reduce_join(a, 0) ==> ["ac", "bd"]
tf.reduce_join(a, -1) = tf.reduce_join(a, 1) ==> ["ab", "cd"]
tf.reduce_join(a, 0, keep_dims=True) ==> [["ac", "bd"]]
tf.reduce_join(a, 1, keep_dims=True) ==> [["ab"], ["cd"]]
tf.reduce_join(a, 0, separator=".") ==> ["a.c", "b.d"]
tf.reduce_join(a, [0, 1]) ==> ["acbd"]
tf.reduce_join(a, [1, 0]) ==> ["abcd"]
tf.reduce_join(a, []) ==> ["abcd"]
```

#### Args:


* axis: A `Tensor` of type `int32`.
 The dimensions to reduce over. Dimensions are reduced in the
 order specified. Omitting `axis` is equivalent to passing
 `[n-1, n-2, ..., 0]`. Negative indices from `-n` to `-1` are supported.
* keep_dims: An optional `bool`. Defaults to `False`.
 If `True`, retain reduced dimensions with length `1`.
* separator: An optional `string`. Defaults to `""`.
 The separator to use when joining.
* name: A name for the operation (optional).

#### Returns:

A `Tensor` of type `string`.
    Has shape equal to that of the input with reduced dimensions removed or
    set to `1` depending on `keep_dims`.




- - -

## <a name="reduce_max"></a>reduce_max(axis=None, keep_dims=False, name=None, reduction_indices=None)



Computes the maximum of elements across dimensions of a tensor.

Reduces `input_tensor` along the dimensions given in `axis`.
Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
entry in `axis`. If `keep_dims` is true, the reduced dimensions
are retained with length 1.

If `axis` has no entries, all dimensions are reduced, and a
tensor with a single element is returned.

#### Args:


* axis: The dimensions to reduce. If `None` (the default),
 reduces all dimensions.
* keep_dims: If true, retains reduced dimensions with length 1.
* name: A name for the operation (optional).
* reduction_indices: The old (deprecated) name for axis.

#### Returns:

The reduced tensor.

  @compatibility(numpy)
  Equivalent to np.max
  @end_compatibility




- - -

## <a name="reduce_mean"></a>reduce_mean(axis=None, keep_dims=False, name=None, reduction_indices=None)



Computes the mean of elements across dimensions of a tensor.

Reduces `input_tensor` along the dimensions given in `axis`.
Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
entry in `axis`. If `keep_dims` is true, the reduced dimensions
are retained with length 1.

If `axis` has no entries, all dimensions are reduced, and a
tensor with a single element is returned.

For example:

```python
# 'x' is [[1., 1.]
#         [2., 2.]]
tf.reduce_mean(x) ==> 1.5
tf.reduce_mean(x, 0) ==> [1.5, 1.5]
tf.reduce_mean(x, 1) ==> [1.,  2.]
```

#### Args:


* axis: The dimensions to reduce. If `None` (the default),
 reduces all dimensions.
* keep_dims: If true, retains reduced dimensions with length 1.
* name: A name for the operation (optional).
* reduction_indices: The old (deprecated) name for axis.

#### Returns:

The reduced tensor.

  @compatibility(numpy)
  Equivalent to np.mean
  @end_compatibility




- - -

## <a name="reduce_min"></a>reduce_min(axis=None, keep_dims=False, name=None, reduction_indices=None)



Computes the minimum of elements across dimensions of a tensor.

Reduces `input_tensor` along the dimensions given in `axis`.
Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
entry in `axis`. If `keep_dims` is true, the reduced dimensions
are retained with length 1.

If `axis` has no entries, all dimensions are reduced, and a
tensor with a single element is returned.

#### Args:


* axis: The dimensions to reduce. If `None` (the default),
 reduces all dimensions.
* keep_dims: If true, retains reduced dimensions with length 1.
* name: A name for the operation (optional).
* reduction_indices: The old (deprecated) name for axis.

#### Returns:

The reduced tensor.

  @compatibility(numpy)
  Equivalent to np.min
  @end_compatibility




- - -

## <a name="reduce_prod"></a>reduce_prod(axis=None, keep_dims=False, name=None, reduction_indices=None)



Computes the product of elements across dimensions of a tensor.

Reduces `input_tensor` along the dimensions given in `axis`.
Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
entry in `axis`. If `keep_dims` is true, the reduced dimensions
are retained with length 1.

If `axis` has no entries, all dimensions are reduced, and a
tensor with a single element is returned.

#### Args:


* axis: The dimensions to reduce. If `None` (the default),
 reduces all dimensions.
* keep_dims: If true, retains reduced dimensions with length 1.
* name: A name for the operation (optional).
* reduction_indices: The old (deprecated) name for axis.

#### Returns:

The reduced tensor.

  @compatibility(numpy)
  Equivalent to np.prod
  @end_compatibility




- - -

## <a name="reduce_sum"></a>reduce_sum(axis=None, keep_dims=False, name=None, reduction_indices=None)



Computes the sum of elements across dimensions of a tensor.

Reduces `input_tensor` along the dimensions given in `axis`.
Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
entry in `axis`. If `keep_dims` is true, the reduced dimensions
are retained with length 1.

If `axis` has no entries, all dimensions are reduced, and a
tensor with a single element is returned.

For example:

```python
# 'x' is [[1, 1, 1]
#         [1, 1, 1]]
tf.reduce_sum(x) ==> 6
tf.reduce_sum(x, 0) ==> [2, 2, 2]
tf.reduce_sum(x, 1) ==> [3, 3]
tf.reduce_sum(x, 1, keep_dims=True) ==> [[3], [3]]
tf.reduce_sum(x, [0, 1]) ==> 6
```

#### Args:


* axis: The dimensions to reduce. If `None` (the default),
 reduces all dimensions.
* keep_dims: If true, retains reduced dimensions with length 1.
* name: A name for the operation (optional).
* reduction_indices: The old (deprecated) name for axis.

#### Returns:

The reduced tensor.

  @compatibility(numpy)
  Equivalent to np.sum
  @end_compatibility




- - -

## <a name="relu"></a>relu(name=None)



Computes rectified linear: `max(features, 0)`.

#### Args:


* name: A name for the operation (optional).

#### Returns:

A `Tensor`. Has the same type as `features`.




- - -

## <a name="relu6"></a>relu6(name=None)



Computes Rectified Linear 6: `min(max(features, 0), 6)`.

#### Args:


* name: A name for the operation (optional).

#### Returns:

A `Tensor` with the same type as `features`.




- - -

## <a name="reshape"></a>reshape(shape_spec)



Reshapes this tensor to the given spec.

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

#### Args:


* shape_spec: The spec for the new shape.

#### Returns:

A Pretty Tensor with the reshaped tensor.


#### Raises:


* ValueError: If there are two many unknown dimensions or the shape_spec
* requires out of range DIM_SAME.


- - -

## <a name="sequence_gru"></a>sequence_gru()



Unrolls `gru_cell` over the input.

This takes an input that is a list of length timesteps where each element
is a `Tensor` of `[batch, *Dims]` and unrolls the recurrent cell. The input
and state to the cell are managed by this method, but the rest of the
arguments are passed through.

  Gated recurrent unit memory cell (GRU).

#### Args:


* num_units: How big is the hidden state.
* bias: An initializer for the bias or a Tensor. No bias if set to None.
* weights: An initializer for weights or a Tensor.
* phase: The phase of graph construction. See `pt.Phase`.
* parameter_modifier: A function to modify parameters that is applied after
 creation and before use.

#### Returns:

A RecurrentResult.




- - -

## <a name="sequence_lstm"></a>sequence_lstm()



Unrolls `lstm_cell` over the input.

This takes an input that is a list of length timesteps where each element
is a `Tensor` of `[batch, *Dims]` and unrolls the recurrent cell. The input
and state to the cell are managed by this method, but the rest of the
arguments are passed through.

  Long short-term memory cell (LSTM).

#### Args:


* num_units: How big is the hidden state.
* bias: An initializer for the bias or a Tensor. No bias if set to None.
* peephole: Whether to use peephole connections as described in
 http://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf
* weights: An initializer for weights or a Tensor.
* phase: The phase of graph construction. See `pt.Phase`.
* parameter_modifier: A function to modify parameters that is applied after
 creation and before use.

#### Returns:

A RecurrentResult.




- - -

## <a name="sigmoid"></a>sigmoid(name=None)



Computes sigmoid of `x` element-wise.

Specifically, `y = 1 / (1 + exp(-x))`.

#### Args:


* name: A name for the operation (optional).

#### Returns:

A Tensor with the same type as `x` if `x.dtype != qint32`
      otherwise the return type is `quint8`.

  @compatibility(numpy)
  Equivalent to np.scipy.special.expit
  @end_compatibility




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

## <a name="softmax_classifier"></a>softmax_classifier(num_classes, labels=None, loss_weight=None, per_example_weights=None, weights=None, bias=<google3.third_party.tensorflow.python.ops.init_ops.Zeros object at 0x3cfbb50>, parameter_modifier=<function identity at 0x402faa0>, name=None)



Creates a fully-connected linear layer followed by a softmax.

This returns `(softmax, loss)` where `loss` is the cross entropy loss.

#### Args:


* num_classes: The number of classes.
* labels: The target labels to learn as a float tensor. Use None to not
 include a training loss.
* loss_weight: A scalar multiplier for the loss.
* per_example_weights: A Tensor with a weight per example.
* weights: The initializer for the weights (see `fully_connected`).
* bias: The initializer for the bias (see `fully_connected`).
* parameter_modifier: A modifier for the parameters that compute the logits.
* name: The optional name.

#### Returns:

A named tuple holding:

    softmax: The result of this layer with softmax normalization.
    loss: The cross entropy loss.


#### Raises:


* ValueError: If the datatype is wrong.


- - -

## <a name="softmax_classifier_with_sampled_loss"></a>softmax_classifier_with_sampled_loss(num_classes, labels, num_sampled, num_true=None, sampled_values=None, remove_accidental_hits=True, loss_weight=None, per_example_weights=None, weights=None, bias=<google3.third_party.tensorflow.python.ops.init_ops.Zeros object at 0x3cfbad0>, parameter_modifier=<function identity at 0x402faa0>, name=softmax_classifier)



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


* num_classes: An `int`. The number of possible classes.
* labels: A `Tensor` of type `int64` and shape `[batch_size,
 num_true]`. The target classes. Note that this format differs from
 the `labels` argument of `nn.softmax_cross_entropy_with_logits`.
* num_sampled: An `int`. The number of classes to randomly sample per batch.
* num_true: An `int`. The number of target classes per training example,
 defaults to the second dim of labels if known or 1.
* sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
 `sampled_expected_count`) returned by a `*_candidate_sampler` function.
 (if None, we default to `log_uniform_candidate_sampler`)
* remove_accidental_hits: A `bool`. whether to remove "accidental hits"
 where a sampled class equals one of the target classes. Default is
 True.
* loss_weight: A scalar multiplier for the loss.
* per_example_weights: A Tensor with a weight per example.
* weights: The initializer for the weights (see `fully_connected`). Note:
 This is the transpose of a normal fully_connected input layer!
* bias: The initializer for the bias (see `fully_connected`).
* parameter_modifier: A modifier for the parameters that compute the logits.
* name: The optional name.

#### Returns:

A tuple of handles to the logits (fully connected layer) and loss.


#### Raises:


* ValueError: If inputs or labels do not have the right shape.


- - -

## <a name="softplus"></a>softplus(name=None)



Computes softplus: `log(exp(features) + 1)`.

#### Args:


* name: A name for the operation (optional).

#### Returns:

A `Tensor`. Has the same type as `features`.




- - -

## <a name="softsign"></a>softsign(name=None)



Computes softsign: `features / (abs(features) + 1)`.

#### Args:


* name: A name for the operation (optional).

#### Returns:

A `Tensor`. Has the same type as `features`.




- - -

## <a name="sparse_cross_entropy"></a>sparse_cross_entropy(labels, name=None, loss_weight=None, per_example_weights=None)



Calculates the Cross Entropy of input_ vs labels.

#### Args:


* labels: A rank 1 integer `Tensor` with class ordinals
* name: The optional name.
* loss_weight: A weight to scale the loss. Used when there are multiple
 losses.
* per_example_weights: A weighting for each example.

#### Returns:

A loss.


#### Raises:


* ValueError: if labels is None or the type is not float or double.


- - -

## <a name="split"></a>split(split_dim=0, num_splits=2)



Splits this Tensor along the split_dim into num_splits Equal chunks.

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

## <a name="sqrt"></a>sqrt(name=None)



Computes square root of x element-wise.

I.e., \(y = \sqrt{x} = x^{1/2}\).

#### Args:


* name: A name for the operation (optional).

#### Returns:

A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.




- - -

## <a name="square"></a>square(name=None)



Computes square of x element-wise.

I.e., \(y = x * x = x^2\).

#### Args:


* name: A name for the operation (optional).

#### Returns:

A `Tensor` or `SparseTensor`. Has the same type as `x`.




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

## <a name="stack"></a>stack(axis=0, name=stack)



Stacks a list of rank-`R` tensors into one rank-`(R+1)` tensor.

Packs the list of tensors in `values` into a tensor with rank one higher than
each tensor in `values`, by packing them along the `axis` dimension.
Given a list of length `N` of tensors of shape `(A, B, C)`;

if `axis == 0` then the `output` tensor will have the shape `(N, A, B, C)`.
if `axis == 1` then the `output` tensor will have the shape `(A, N, B, C)`.
Etc.

For example:

```prettyprint
# 'x' is [1, 4]
# 'y' is [2, 5]
# 'z' is [3, 6]
stack([x, y, z]) => [[1, 4], [2, 5], [3, 6]]  # Pack along first dim.
stack([x, y, z], axis=1) => [[1, 2, 3], [4, 5, 6]]
```

This is the opposite of unstack.  The numpy equivalent is

    tf.stack([x, y, z]) = np.asarray([x, y, z])

#### Args:


* axis: An `int`. The axis to stack along. Defaults to the first dimension.
 Supports negative indexes.
* name: A name for this operation (optional).

#### Returns:

output: A stacked `Tensor` with the same type as `values`.


#### Raises:


* ValueError: If `axis` is out of the range [-(R+1), R+1).


- - -

## <a name="stop_gradient"></a>stop_gradient()



Cuts off the gradient at this point.

This works on both sequence and regular Pretty Tensors.

#### Args:


* Returns:
 A new Pretty Tensor of the same type with stop_gradient applied.





- - -

## <a name="tanh"></a>tanh(name=None)



Computes hyperbolic tangent of `x` element-wise.

#### Args:


* name: A name for the operation (optional).

#### Returns:

A Tensor or SparseTensor respectively with the same type as `x` if
    `x.dtype != qint32` otherwise the return type is `quint8`.




- - -

## <a name="tensor_shape"></a>tensor_shape(name=None, out_type=<dtype: 'int32'>)



Returns the shape of a tensor.

This operation returns a 1-D integer tensor representing the shape of `input`.

For example:

```python
# 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
shape(t) ==> [2, 2, 3]
```

#### Args:


* name: A name for the operation (optional).
* out_type: (Optional) The specified output type of the operation
 (`int32` or `int64`). Defaults to `tf.int32`.

#### Returns:

A `Tensor` of type `out_type`.




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

## <a name="to_double"></a>to_double(name=ToDouble)



Casts a tensor to type `float64`.

#### Args:


* name: A name for the operation (optional).

#### Returns:

A `Tensor` or `SparseTensor` with same shape as `x` with type `float64`.


#### Raises:


* TypeError: If `x` cannot be cast to the `float64`.


- - -

## <a name="to_float"></a>to_float(name=ToFloat)



Casts a tensor to type `float32`.

#### Args:


* name: A name for the operation (optional).

#### Returns:

A `Tensor` or `SparseTensor` with same shape as `x` with type `float32`.


#### Raises:


* TypeError: If `x` cannot be cast to the `float32`.


- - -

## <a name="to_int32"></a>to_int32(name=ToInt32)



Casts a tensor to type `int32`.

#### Args:


* name: A name for the operation (optional).

#### Returns:

A `Tensor` or `SparseTensor` with same shape as `x` with type `int32`.


#### Raises:


* TypeError: If `x` cannot be cast to the `int32`.


- - -

## <a name="to_int64"></a>to_int64(name=ToInt64)



Casts a tensor to type `int64`.

#### Args:


* name: A name for the operation (optional).

#### Returns:

A `Tensor` or `SparseTensor` with same shape as `x` with type `int64`.


#### Raises:


* TypeError: If `x` cannot be cast to the `int64`.


- - -

## <a name="unstack"></a>unstack(num=None, axis=0, name=unstack)



Unpacks the given dimension of a rank-`R` tensor into rank-`(R-1)` tensors.

Unpacks `num` tensors from `value` by chipping it along the `axis` dimension.
If `num` is not specified (the default), it is inferred from `value`'s shape.
If `value.shape[axis]` is not known, `ValueError` is raised.

For example, given a tensor of shape `(A, B, C, D)`;

If `axis == 0` then the i'th tensor in `output` is the slice
  `value[i, :, :, :]` and each tensor in `output` will have shape `(B, C, D)`.
  (Note that the dimension unpacked along is gone, unlike `split`).

If `axis == 1` then the i'th tensor in `output` is the slice
  `value[:, i, :, :]` and each tensor in `output` will have shape `(A, C, D)`.
Etc.

This is the opposite of pack.  The numpy equivalent is

    tf.unstack(x, n) = list(x)

#### Args:


* num: An `int`. The length of the dimension `axis`. Automatically inferred
 if `None` (the default).
* axis: An `int`. The axis to unstack along. Defaults to the first
 dimension. Supports negative indexes.
* name: A name for the operation (optional).

#### Returns:

The list of `Tensor` objects unstacked from `value`.


#### Raises:


* ValueError: If `num` is unspecified and cannot be inferred.
* ValueError: If `axis` is out of the range [-R, R).


- - -

## <a name="unzip"></a>unzip(split_dim=0, num_splits=2)



Unzips this Tensor along the split_dim into num_splits Equal chunks.

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

## <a name="with_name"></a>with_name(name)



Sets the name scope for future operations.





- - -

## <a name="with_sequence"></a>with_sequence(sequence, parameters=None)



Returns a PrettyTensor that points to sequence.





- - -

## <a name="with_tensor"></a>with_tensor(tensor, parameters=None)



Returns a PrettyTensor that points to tensor.





- - -
