import jax
import jax.numpy as jnp
import haiku as hk
from typing import Optional, Tuple, Callable, Iterable

class MLP(hk.Module):
  """
  This is just the
  `Haiku MLP <https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/nets/mlp.py>`_
  extended with layer normalization.

  Args:
    output_sizes: Sequence of layer sizes.
    w_init: Initializer for :class:`haiku.Linear` weights.
    b_init: Initializer for :class:`haiku.Linear` bias. Must be ``None`` if
      ``with_bias=False``.
    with_bias: Whether or not to apply a bias in each layer.
    with_layer_norm: Whether or not to apply layer normalization in each layer.
    activation: Activation function to apply between :class:`~haiku.Linear`
      layers. Defaults to ReLU.
    activate_final: Whether or not to activate the final layer of the MLP.
    name: Optional name for this module.
    
  Raises:
    ValueError: If ``with_bias`` is ``False`` and ``b_init`` is not ``None``.

  """
  def __init__(
      self,
      output_sizes: Iterable[int],
      w_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None,
      with_bias: bool = True,
      with_layer_norm: bool = False,
      activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
      activate_final: bool = False,
      name: Optional[str] = None,
  ):
    if not with_bias and b_init is not None:
      raise ValueError("When with_bias=False b_init must not be set.")

    super().__init__(name=name)
    self.with_bias = with_bias
    self.with_layer_norm = with_layer_norm
    self.w_init = w_init
    self.b_init = b_init
    self.activation = activation
    self.activate_final = activate_final
    layers = []
    output_sizes = tuple(output_sizes)
    for index, output_size in enumerate(output_sizes):
      linear = hk.Linear(output_size=output_size,
                              w_init=w_init,
                              b_init=b_init,
                              with_bias=with_bias,
                              name="linear_%d" % index)
      norm = None
      if with_layer_norm:
        norm = hk.LayerNorm(
              axis=-1, create_scale=True, create_offset=True)
      layers.append((linear, norm))

    self.layers = tuple(layers)
    self.output_size = output_sizes[-1] if output_sizes else None

  def __call__(
      self,
      inputs: jnp.ndarray,
      dropout_rate: Optional[float] = None,
      rng=None,
  ) -> jnp.ndarray:
    """Connects the module to some inputs.
    
    Args:
      inputs: A Tensor of shape ``[batch_size, input_size]``.
      dropout_rate: Optional dropout rate.
      rng: Optional RNG key. Require when using dropout.
      
    Returns:
      The output of the model of size ``[batch_size, output_size]``.
    """
    if dropout_rate is not None and rng is None:
      raise ValueError("When using dropout an rng key must be passed.")
    elif dropout_rate is None and rng is not None:
      raise ValueError("RNG should only be passed when using dropout.")

    rng = hk.PRNGSequence(rng) if rng is not None else None
    num_layers = len(self.layers)

    out = inputs
    for i, layer in enumerate(self.layers):
      linear, norm = layer
      out = linear(out)
      if self.with_layer_norm:
        out = norm(out)
      if i < (num_layers - 1) or self.activate_final:
        # Only perform dropout if we are activating the output.
        if dropout_rate is not None:
          out = hk.dropout(next(rng), dropout_rate, out)
        out = self.activation(out)

    return out
