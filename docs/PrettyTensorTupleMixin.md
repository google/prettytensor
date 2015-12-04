<!-- This file was automatically generated. -->

# PrettyTensorTupleMixin

Adds methods to any sequence type so that it can be used with binding.

Generally this can be used with namedtuples to provide clean multi-value
returns:

class MyType(namedtuple(...), PrettyTensorTupleMixin):
  pass

Subclasses with nested structure should note that this does not unpack
nested structure by default.  You must implement flatten and
build_from_flattened.

- - -

[TOC]


## as_fn() {#as_fn}



Creates a function by binding the arguments in the given order.

#### Args:


* binding_order: The unbound variables. This must include all values.

#### Returns:

A function that takes the arguments of binding_order.


#### Raises:


* ValueError: If the bindings are missing values or include unknown values.


- - -

## bind() {#bind}



Makes the bindings to each item in this and returns a new tuple.





- - -

## build_from_flattened(flattened) {#build_from_flattened}



Given a flattened structure from flatten, make a new version of this.





- - -

## construct() {#construct}




- - -

## flatten() {#flatten}



Subclasses with nested structure should implement this method.


#### Returns:

A list of data that should be bound and constructed, by default just self.




- - -

## has_unbound_vars() {#has_unbound_vars}



Returns whether there are any unbound vars in this tuple.





- - -
## Properties

* unbound_vars
