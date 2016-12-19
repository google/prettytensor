## 0.7.3

1. Maintenance release w/ some deprecation notice fixes. Note: this may change the names of the summaries.

## 0.7.2

1. Maintenance release w/ change to project pip dependencies to better support GPU builds.

## 0.7.1

### General
1. Changed `weights` to `init` and `bias_init` to `bias` and made these support initialization functions or `Tensors`.
2. Added `parameter_modifier`. These are functions that are applied after creating a `Variable`, but before it is used in the graph. They allow you to apply a function like normalization or drop connect to the graph. See `pt.parameters` for details.
3. Added support for directly chaining many useful TensorFlow functions.  See [pretty_tensor_methods.py](https://github.com/google/prettytensor/blob/master/prettytensor/pretty_tensor_methods.py#L700) for details. Note: when a function is removed from tf (e.g. complex_abs), it will be removed here.
1. Changed internal calls to TF to comply with API changes.
2. Internally changed the name of the first parameter to be more consistent. This should not be user visible since it is the variable to the left of the '.'.

### Losses

1. Added `per_output_weights` to `binary_cross_entropy_with_logits` and  that allow you to weight the loss from classes and examples.
2. Added `sparse_cross_entropy` to efficiently calculate the loss of when you have a vector of 1 hot labels as indices (`tf.int32`/`tf.int64`). Also added `evaluate_classifier_sparse`.
3. Fixed `softmax_classifier_with_sampled_loss` to support specified parameters and parameter modification.
4. Standardized on `num_classes` and changed the parameter name in `softmax_classifier` accordingly.

### Optimizer
1. Added `clip_gradients_by_norm` to `apply_optimizer`.

### Images

1. Added a differentiable sampling method for images called `bilinear_sampling`.


## 0.6.2

Add Depthwise Convolution

### Batch Normalization
1. Make Batch Normalization work with arbitrary dimensionality.
2. Allow passing through arguments to BN using a namedtuple.
3. Add BN default values.
4. Remove requirement to use with_update_ops to make BN accumulate values for
    inference.



## 0.6.0

1. Adding scoped control of summary creation.
2. Scoped variable collections.
3. Can initialize variables from literals.
4. Fixed operators -- Sequential's plus no longer has side effects.
5. Operators now work on Pretty Tensors that contain lists.


Note: (4) may be breaking!

## 0.5.3

1. Fixing tutorials (thanks jkahn!)
2. Adding a precicion and recall evaluation.
3. Various bug fixes.

Tested on TF 0.7.1

## 0.5.2

1. Various bug fixes
2. Reordered the arguments to a better positional order.
3. Added a length argument to recurrent networks to support short circuiting.
4. Improvements to reshape.
5. Python 3 support.

## 0.5.0

Initial Release
