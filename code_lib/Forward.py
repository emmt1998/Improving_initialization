
import jax as jx
import jax.numpy as jnp

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