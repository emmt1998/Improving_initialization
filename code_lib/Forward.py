
import jax as jx
import jax.numpy as jnp

def init_params(layers, key, init='glorot_normal', activation='relu', **kwargs):
  #  https://pytorch.org/docs/stable/nn.init.html
  Ws = []
  bs = []
  gain = {
      'relu': jnp.sqrt(2),
      'tanh': 5/3,
      'selu': 3/4,
  }[activation]
    
  key, subkey = jx.random.split(key)
  fan_mode = 0 if kwargs.get('fan_mode', 'fan_in') else 1

  for i in range(len(layers) - 1):
    match init:
      case 'glorot_normal':
        std_glorot = jnp.sqrt(2/(layers[i] + layers[i + 1]))
        Ws.append(jx.random.normal(subkey, (layers[i], layers[i + 1]))*std_glorot)
        bs.append(jnp.zeros(layers[i + 1]))
      case 'he_normal':
        std_he = gain / jnp.sqrt(layers[i + fan_mode])
        Ws.append(jx.random.normal(subkey, (layers[i], layers[i + 1]))*std_he)
        bs.append(jnp.zeros(layers[i + 1]))
      case 'uniform':
        Ws.append(jx.random.uniform(subkey, (layers[i], layers[i+1])))
        bs.append(jnp.zeros(layers[i + 1]))
      case 'glorot_uniform':
        bound = gain * jnp.sqrt(6 / layers[i + fan_mode])
        Ws.append(jx.random.uniform(subkey, (layers[i], layers[i+1]), minval=-bound, maxval=bound))
        bs.append(jnp.zeros(layers[i + 1]))
      case 'he_uniform':
        bound = gain * jnp.sqrt(3 / layers[i + fan_mode])
        Ws.append(jx.random.uniform(subkey, (layers[i], layers[i+1]), minval=-bound, maxval=bound))
        bs.append(jnp.zeros(layers[i + 1]))
      case _:
        raise NotImplementedError(
            'Not available. Try:' +\
            ' glorot_normal, glorot_uniform' +\
            ', he_normal, he_uniform' +\
            ', uniform.'
        )
  return [Ws, bs]

@jx.jit
def forward_pass_multi(H, params, s):
  # Forward pass with tanh
  Ws = params[0]
  bs = params[1]
  N_layers = len(Ws)
  for i in range(N_layers - 1):
    H = jnp.matmul(H, Ws[i]) + bs[i]
    H = jx.nn.relu(H)**2
  Y = jnp.matmul(H, s)
  return H, Y

forward_pass_multi_grad = jx.jit(jx.vmap(jx.jacrev(forward_pass_multi), in_axes=(0, None, None)))
forward_pass_multi_grad2 = jx.jit(jx.vmap(jx.jacrev(jx.jacrev(forward_pass_multi)), in_axes=(0, None, None)))

@jx.jit
def forward_pass(H, params):
  # Forward pass with tanh
  Ws = params[0]
  bs = params[1]
  N_layers = len(Ws)
  for i in range(N_layers - 1):
    H = jnp.matmul(H, Ws[i]) + bs[i]
    H = jx.nn.relu(H)**2
  Y = jnp.matmul(H, Ws[-1]) + bs[-1]
  return Y


@jx.jit
def one_pass(H, params):
  # Forward pass with tanh
  Ws = params[0]
  bs = params[1]
  N_layers = len(Ws)
  for i in range(N_layers - 1):
    H = jnp.matmul(H, Ws[i]) + bs[i]
    H = jx.nn.relu(H)**2
  return H

@jx.jit
def forward_grad_pass(H, params):
  # Forward pass with tanh
  Ws = params[0]
  bs = params[1]
  N_layers = len(Ws)
  for i in range(N_layers - 1):
    H = jnp.matmul(H, Ws[i]) + bs[i]
    H = Ws[i]*jx.nn.relu(H)*2
  Y = jnp.matmul(H, Ws[-1])
  return Y


@jx.jit
def one_grad_pass(H, params):
  # Forward pass with tanh
  Ws = params[0]
  bs = params[1]
  N_layers = len(Ws)
  for i in range(N_layers - 1):
    H = jnp.matmul(H, Ws[i]) + bs[i]
    H = Ws[i]*jx.nn.relu(H)*2
  return H

@jx.jit
def forward_grad2_pass(H, params):
  # Forward pass with tanh
  Ws = params[0]
  bs = params[1]
  N_layers = len(Ws)
  for i in range(N_layers - 1):
    H = jnp.matmul(H, Ws[i]) + bs[i]
    H = 2*Ws[i]**2*(H>=0)
  Y = jnp.matmul(H, Ws[-1])
  return Y

@jx.jit
def one_grad2_pass(H, params):
  # Forward pass with tanh
  Ws = params[0]
  bs = params[1]
  N_layers = len(Ws)
  for i in range(N_layers - 1):
    H = jnp.matmul(H, Ws[i]) + bs[i]
    H = 2*Ws[i]**2*(H>=0)
  return H