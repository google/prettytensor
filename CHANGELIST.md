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
