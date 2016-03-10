<!-- This file was automatically generated. -->

# Bookkeeper

Small class to gather needed pieces from a Graph being built.

The following properties are exposed:

* batch_size: The size of the batches.
* bits: A dict where named layers and losses can be placed for reference
    later.
* global_step: A global step counter. Every time training is executed is
    considered a step.
* g: The graph.
* train_op: The training operation, if setup_training was called.
* loss: A list of losses.

- - -

[TOC]


## add_average_summary(var, tag=None, decay=0.999, ignore_nan=True) {#add_average_summary}



Add a summary with the moving average of var.

Adds a variable to keep track of the exponential moving average and adds an
update operation to the bookkeeper. The name of the variable is
'%s_average' % name prefixed with the current variable scope.

#### Args:


* var: The variable for which a moving average should be computed.
* tag: The tag of the summary. If None var.name[:-2] is used to strip off
 the ':0' that is added by TF (bookkeeper keeps all var names unique, so
 it is only ever the first one.)
* decay: How much history to use in the moving average.
 Higher, means more history values [0.9, 1) accepted.
* ignore_nan: If the value is NaN or Inf, skip it. Note that this default
 is different than the exponential_moving_average one.

#### Returns:

The averaged variable.


#### Raises:


* ValueError: if decay is not in [0.9, 1).


- - -

## add_histogram_summary(tensor, tag=None) {#add_histogram_summary}



Add a summary operation to visualize any tensor.





- - -

## add_loss(loss, name=None, regularization=False, add_summaries=True) {#add_loss}



Append a loss to the total loss for the network.

#### Args:


* loss: append this loss operation
* name: The name for this loss, defaults to loss.op.name
* regularization: Set to True if this is a regularization loss.
* add_summaries: Set to True if you want to see scalar and average summary.





- - -

## add_losses(losses, regularization=False) {#add_losses}




- - -

## add_scalar_summary(x, tag=None) {#add_scalar_summary}



Adds a scalar summary for x.





- - -

## check_summary(tag) {#check_summary}




- - -

## create_composite_loss(losses, regularize=True, include_marked=True, name=cost) {#create_composite_loss}



Creates a loss that is the sum of all specified losses.

#### Args:


* losses: A sequence of losses to include.
* regularize: Whether or not to include regularization losses.
* include_marked: Whether or not to use the marked losses.
* name: The name for this variable.

#### Returns:

A single tensor that is the sum of all losses.


#### Raises:


* ValueError: if there are no losses.


- - -

## exponential_moving_average(var, avg_var=None, decay=0.999, ignore_nan=False) {#exponential_moving_average}



Calculates the exponential moving average.

Adds a variable to keep track of the exponential moving average and adds an
update operation to the bookkeeper. The name of the variable is
'%s_average' % name prefixed with the current variable scope.

#### Args:


* var: The variable for which a moving average should be computed.
* avg_var: The variable to set the average into, if None create a zero
 initialized one.
* decay: How much history to use in the moving average.
 Higher, means more history values [0, 1) accepted.
* ignore_nan: If the value is NaN or Inf, skip it.

#### Returns:

The averaged variable.


#### Raises:


* ValueError: if decay is not in [0, 1).


- - -

## with_update_ops(train_op) {#with_update_ops}




- - -
## Properties

* g
* global_step
* marked_losses
* recurrent_state
* regularization_losses
* summaries
* update_ops
