"""RNN modules for Flax."""

from typing import Any, TypeVar
from collections.abc import Mapping
from typing import Callable
import logging

import jax
import jax.numpy as jnp

from flax import nnx
from flax.nnx import filterlib, rnglib
from flax.nnx.module import Module
from flax.nnx.nn import initializers
from flax.nnx.transforms import iteration

default_kernel_init = initializers.lecun_normal()
default_bias_init = initializers.zeros_init()

A = TypeVar("A")
Array = jax.Array
Output = Any
Carry = Any

from flax.nnx.nn.recurrent import RNNCellBase, flip_sequences, _select_last_carry, RNNBase
from flax.nnx.nn.recurrent import _concatenate

class RNN(Module):
  """The ``RNN`` module takes any :class:`RNNCellBase` instance and applies it over a sequence

  using :func:`flax.nnx.scan`.
  """

  state_axes: dict[str, int | type[iteration.Carry] | None]

  __data__ = ('cell', 'rngs')

  def __init__(
    self,
    cell: RNNCellBase,
    dropout_rate: float = 0.2,
    time_major: bool = False,
    return_carry: bool = False,
    reverse: bool = False,
    keep_order: bool = False,
    unroll: int = 1,
    rngs: rnglib.Rngs | None = None,
    state_axes: Mapping[str, int | type[iteration.Carry] | None] | None = None,
    broadcast_rngs: filterlib.Filter = None,
  ):
    self.cell = cell
    self.time_major = time_major
    self.return_carry = return_carry
    self.reverse = reverse
    self.keep_order = keep_order
    self.unroll = unroll
    if rngs is None:
      rngs = rnglib.Rngs(0)
    self.rngs = rngs
    self.state_axes = state_axes or {...: iteration.Carry}  # type: ignore
    self.broadcast_rngs = broadcast_rngs
    self.recurrent_dropout = nnx.Dropout(
      rate=dropout_rate, rng_collection='recurrent_dropout', rngs=rngs
    )

  def __call__(
    self,
    inputs: Array,
    masks: Array, # Mask is inclusion criteria, not exclusion
    *,
    initial_carry: Carry | None = None,
    seq_lengths: Array | None = None,
    return_carry: bool | None = None,
    time_major: bool | None = None,
    reverse: bool | None = None,
    keep_order: bool | None = None,
    rngs: rnglib.Rngs | None = None,
  ):
    if return_carry is None:
      return_carry = self.return_carry
    if time_major is None:
      time_major = self.time_major
    if reverse is None:
      reverse = self.reverse
    if keep_order is None:
      keep_order = self.keep_order

    # Infer the number of batch dimensions from the input shape.
    # Cells like ConvLSTM have additional spatial dimensions.
    time_axis = 0 if time_major else inputs.ndim - (self.cell.num_feature_axes + 1)

    # make time_axis positive
    if time_axis < 0:
      time_axis += inputs.ndim

    if time_major:
      # we add +1 because we moved the time axis to the front
      batch_dims = inputs.shape[1 : -self.cell.num_feature_axes]
    else:
      batch_dims = inputs.shape[:time_axis]

    # maybe reverse the sequence
    if reverse:
      inputs = jax.tree_util.tree_map(
                lambda x: flip_sequences(
                    x,
                    seq_lengths,
                    num_batch_dims=len(batch_dims),
                    time_major=time_major,  # type: ignore
                ),
                inputs,
            )
      # Changed
      masks = jax.tree_util.tree_map(
                lambda x: flip_sequences(
                    x,
                    seq_lengths,
                    num_batch_dims=len(batch_dims),
                    time_major=time_major,  # type: ignore
                ),
                masks,
            )
    if rngs is None:
      rngs = self.rngs
    carry: Carry = (
            self.cell.initialize_carry(
                inputs.shape[:time_axis] + inputs.shape[time_axis + 1 :], rngs
            )
            if initial_carry is None
            else initial_carry
        )

    slice_carry = seq_lengths is not None and return_carry
    broadcast_rngs = nnx.All(nnx.RngState, self.broadcast_rngs)
    state_axes = iteration.StateAxes({broadcast_rngs: None, **self.state_axes})  # type: ignore[misc]

    # we use split_rngs with splits=1 and squeeze=True to get unique rngs
    # every time RNN is called
    @nnx.split_rngs(splits=1, only=self.broadcast_rngs, squeeze=True)
    @nnx.scan(
      in_axes=(state_axes, iteration.Carry, time_axis, time_axis),
      out_axes=(iteration.Carry, (0, time_axis))
      if slice_carry
      else (iteration.Carry, time_axis),
      unroll=self.unroll,
    )
    def scan_fn(
      cell: RNNCellBase, carry: Carry, x: Array, mask: Array
    ) -> tuple[Carry, Array] | tuple[Carry, tuple[Carry, Array]]:
      mask_broadcast = jnp.expand_dims(mask, axis=-1)

      # Step 1
      x_masked = mask_broadcast * x
      new_carry, y = cell(carry, x_masked)
      not_mask = ~mask_broadcast

      # Step 2
      carry = jax.tree.map(lambda old, new: old * not_mask + new * mask_broadcast,
                           carry,
                           new_carry)

      # Step 3
      y = mask_broadcast * y
      if slice_carry:
        return carry, (carry, y)
      return carry, y

    scan_output = scan_fn(self.cell, carry, inputs, masks)

    # Next we select the final carry. If a segmentation mask was provided and
    # return_carry is True we slice the carry history and select the last valid
    # carry for each sequence. Otherwise we just use the last carry.
    if slice_carry:
      assert seq_lengths is not None
      _, (carries, outputs) = scan_output
      # seq_lengths[None] expands the shape of the mask to match the
      # number of dimensions of the carry.
      carry = _select_last_carry(carries, seq_lengths)
    else:
      carry, outputs = scan_output

    if reverse and keep_order:
      outputs = jax.tree_util.tree_map(
                lambda x: flip_sequences(
                    x,
                    seq_lengths,
                    num_batch_dims=len(batch_dims),
                    time_major=time_major,  # type: ignore
                ),
                outputs,
            )

    if return_carry:
      return carry, outputs
    else:
      return outputs

class Bidirectional(Module):
    """Processes the input in both directions and merges the results.

    Example usage::

      >>> from flax import nnx
      >>> import jax
      >>> import jax.numpy as jnp

      >>> # Define forward and backward RNNs
      >>> forward_rnn = RNN(GRUCell(in_features=3, hidden_features=4, rngs=nnx.Rngs(0)))
      >>> backward_rnn = RNN(GRUCell(in_features=3, hidden_features=4, rngs=nnx.Rngs(0)))

      >>> # Create Bidirectional layer
      >>> layer = Bidirectional(forward_rnn=forward_rnn, backward_rnn=backward_rnn)

      >>> # Input data
      >>> x = jnp.ones((2, 3, 3))

      >>> # Apply the layer
      >>> out = layer(x)
      >>> print(out.shape)
      (2, 3, 8)

    """

    forward_rnn: RNNBase
    backward_rnn: RNNBase
    merge_fn: Callable[[Array, Array], Array] = _concatenate
    time_major: bool = False
    return_carry: bool = False

    def __init__(
        self,
        forward_rnn: RNNBase,
        backward_rnn: RNNBase,
        *,
        merge_fn: Callable[[Array, Array], Array] = _concatenate,
        time_major: bool = False,
        return_carry: bool = False,
        rngs: rnglib.Rngs | None = None,
    ):
        self.forward_rnn = forward_rnn
        self.backward_rnn = backward_rnn
        self.merge_fn = merge_fn
        self.time_major = time_major
        self.return_carry = return_carry
        if rngs is None:
            rngs = rnglib.Rngs(0)
        self.rngs = rngs

    def __call__(
        self,
        inputs: Array,
        masks: Array,
        *,
        initial_carry: tuple[Carry, Carry] | None = None,
        rngs: rnglib.Rngs | None = None,
        seq_lengths: Array | None = None,
        return_carry: bool | None = None,
        time_major: bool | None = None,
        reverse: bool | None = None,  # unused
        keep_order: bool | None = None,  # unused
    ) -> Output | tuple[tuple[Carry, Carry], Output]:
        if time_major is None:
            time_major = self.time_major
        if return_carry is None:
            return_carry = self.return_carry
        if rngs is None:
            rngs = self.rngs
        if initial_carry is not None:
            initial_carry_forward, initial_carry_backward = initial_carry
        else:
            initial_carry_forward = None
            initial_carry_backward = None
        # Throw a warning in case the user accidentally re-uses the forward RNN
        # for the backward pass and does not intend for them to share parameters.
        if self.forward_rnn is self.backward_rnn:
            logging.warning(
                "forward_rnn and backward_rnn is the same object, so "
                "they will share parameters."
            )

        # Encode in the forward direction.
        carry_forward, outputs_forward = self.forward_rnn(
            inputs,
            masks,
            initial_carry=initial_carry_forward,
            rngs=rngs,
            seq_lengths=seq_lengths,
            return_carry=True,
            time_major=time_major,
            reverse=False,
        )

        # Encode in the backward direction.
        carry_backward, outputs_backward = self.backward_rnn(
            inputs,
            masks,
            initial_carry=initial_carry_backward,
            rngs=rngs,
            seq_lengths=seq_lengths,
            return_carry=True,
            time_major=time_major,
            reverse=True,
            keep_order=True,
        )

        carry = (carry_forward, carry_backward) if return_carry else None
        outputs = jax.tree_util.tree_map(
            self.merge_fn, outputs_forward, outputs_backward
        )

        if return_carry:
            return carry, outputs
        else:
            return outputs