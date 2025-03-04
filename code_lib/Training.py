import jax as jx
import jax.numpy as jnp
from jax.example_libraries import optimizers
from functools import partial

from tqdm import tqdm
from .Forward import *

class Training():
    def __init__(self, optimizer, opt_args):
        self.opt_init, self.opt_update, self.get_params = optimizer(opt_args["lr"])

    @partial(jx.jit, static_argnums=(0,1,))
    def step(self, loss, i, opt_state, X_batch, Y_batch, X_c):
        # Optimizer step
        params = self.get_params(opt_state)
        g = jx.grad(loss)(params, X_batch, Y_batch, X_c)
        return self.opt_update(i, g, opt_state)

    def train(self, loss, loss2, X, Y, X_c, opt_state, X_bd = None, X_bn = None, nIter = 10000, stop=1e-5):
        train_loss = []
        val_loss = []
        key = jx.random.PRNGKey(0)
        for it in (pbar := tqdm(range(nIter))):
            opt_state = self.step(loss, it, opt_state, X, Y)
            if it % 50 == 0:
                params = self.get_params(opt_state)
                train_loss_value = loss(params, X, Y)
                train_loss.append(train_loss_value)
                to_print = "it %i, train loss = %e" % (it, train_loss_value)
                pbar.set_description(f"{to_print}")
                if train_loss_value<stop:break
        return self.get_params(opt_state), train_loss, val_loss

    def train_ls(self, loss, loss2, X, Y, X_c, opt_state, X_bd = None, X_bn = None, bsize=16, nIter = 10000, stop=1e-5):
        train_loss = []
        val_loss = []
        key = jx.random.PRNGKey(0)
        for it in (pbar := tqdm(range(nIter))):
            perm = jx.random.choice(key, jnp.arange(X.shape[0]), (bsize,))
            key, subkey = jx.random.split(key)

            X_b = X[perm]
            Y_b = Y[perm]
            opt_state = self.step(loss, it, opt_state, X_b, Y_b)
            params = self.get_params(opt_state)

            # res_last = one_pass(X, params)
            res_last = one_grad_pass(X, params)
            vec_ext = res_last
            # vec_ext = jnp.c_[res_last, 1+0*res_last[:,0]]
            ls = (jnp.linalg.pinv(vec_ext)@Y_b)

            params[0][-1] = ls
            params[1][-1] = 0.#ls[-1:]
            opt_state = self.opt_init(params)

            if it % 50 == 0:
                params = self.get_params(opt_state)
                train_loss_value = loss(params, X, Y)
                train_loss.append(train_loss_value)
                to_print = "it %i, train loss = %e" % (it, train_loss_value)
                pbar.set_description(f"{to_print}")
                if train_loss_value<stop:break
        return self.get_params(opt_state), train_loss, val_loss


    def train_ls_auto(self, loss, loss2, X, Y, X_c, opt_state, X_bd = None, X_bn = None, bsize=16, nIter = 10000, stop=1e-5):
        train_loss = []
        val_loss = []
        key = jx.random.PRNGKey(0)
        for it in (pbar := tqdm(range(nIter))):        
            perm = jx.random.choice(key, jnp.arange(X.shape[0]), (bsize,))
            key, subkey = jx.random.split(key)

            X_b = X[perm]
            Y_b = Y[perm]
            opt_state = self.step(loss, it, opt_state, X_b, Y_b, X_c)
    
            params = self.get_params(opt_state)
            last_homo = one_pass(X_c, params)
            last_homo_g = one_grad_pass(X_b, params)
            last_homo_g2 = one_grad2_pass(X_b, params)
            s_init = jnp.c_[params[0][-1].T, params[1][-1]]
            s_init = s_init[0,:][:-1]
            gL2 = jx.grad(loss2)
            B = gL2(s_init*0, last_homo, last_homo_g,last_homo_g2, X_b, Y_b)
            ggL2 = jx.jacfwd(gL2)
            A = 0.5*ggL2(s_init*0, last_homo, last_homo_g, last_homo_g2, X_b, Y_b)
            s_fin = -0.5*jnp.linalg.pinv(A)@B
            params[0][-1] = s_fin[:,None]
            params[1][-1] = s_fin[-1:][:,None]
            
            opt_state = self.opt_init(params)

            if it % 50 == 0:
                params = self.get_params(opt_state)
                train_loss_value = loss(params, X, Y, X_c)
                train_loss.append(train_loss_value)
                to_print = "it %i, train loss = %e" % (it, train_loss_value)
                pbar.set_description(f"{to_print}")
                #if train_loss_value<stop:break
        return self.get_params(opt_state), train_loss, val_loss

class Training2():
    def __init__(self, optimizer, opt_args, mode="R"):
        self.opt_init, self.opt_update, self.get_params = optimizer(opt_args["lr"])
        options = {
            "R":self.batchsampler,
            "MC":self.uniformsampler,
            "QMC1": self.quadsampler
        }
        self.sampler = options[mode]

    @partial(jx.jit, static_argnums=(0,1,))
    def step(self, loss, i, opt_state, s, X_batch, Y_batch, X_c):
        # Optimizer step
        params = self.get_params(opt_state)
        g = jx.grad(loss)(params, s, X_batch, Y_batch, X_c)
        return self.opt_update(i, g, opt_state)
    
    @partial(jx.jit, static_argnums=(0,1,))
    def leastsquares(self, loss, params, s, X, Y, X_c):
        gl = jx.grad(loss, argnums=1)
        B = gl(params, s*0, X, Y, X_c)
        A = 0.5*jx.jacfwd(gl, argnums=1)(params, s*0, X, Y, X_c).squeeze()
        s = -0.5*jnp.linalg.inv(A+1e-6*jnp.eye(B.shape[0]))@B
        # s = -0.5*jnp.linalg.inv(A)@B
        # s = -0.5*jnp.linalg.pinv(A)@B
        # np.cond(A.T@A)
        return s
    
    @partial(jx.jit, static_argnums=(0,4))
    def batchsampler(self, key, X, Y, bsize):
        perm = jx.random.choice(key, jnp.arange(X.shape[0]), (bsize,))
        key, subkey = jx.random.split(key)
        X_b = X[perm]
        Y_b = Y[perm]
        return key, X_b, Y_b
    
    @partial(jx.jit, static_argnums=(0,4))
    def uniformsampler(self, key, X, Y, bsize):
        vmin = 0
        vmax = 1
        X_b = jx.random.uniform(key, (bsize, X.shape[-1]), minval =vmin, maxval=vmax)
        key, subkey = jx.random.split(key)
        Y_b = X_b
        return key, X_b, Y_b

    @partial(jx.jit, static_argnums=(0,4))
    def quadsampler(self, key, X, Y, bsize):
        vmin = 0
        vmax = 1
        line, h = jnp.linspace(vmin, vmax, bsize//2, retstep=True)
        u = jx.random.uniform(key, (bsize//2, X.shape[-1]), minval =0, maxval=h)
        X_b = jnp.vstack([line[:,None]+u, line[:,None]+h-u])

        key, subkey = jx.random.split(key)
        Y_b = X_b
        return key, X_b, Y_b

    def train(self, loss, X, Y, X_c, opt_state, X_bd = None, X_bn = None, bsize=16, nIter = 10000, stop=1e-8):
        train_loss = []
        val_loss = []
        key = jx.random.PRNGKey(0)
        params = self.get_params(opt_state)
        s0 = params[0][-1]
        key, X_b, Y_b = self.sampler(key, X, Y, bsize)
        s = self.leastsquares(loss, params, s0, X_b, Y_b, X_c)
        # s = s0
        for it in (pbar := tqdm(range(nIter))):        
            key, X_b, Y_b = self.sampler(key, X, Y, bsize)
            params = self.get_params(opt_state)
            s = self.leastsquares(loss, params, s, X_b, Y_b, X_c)
            opt_state = self.step(loss, it, opt_state, s, X_b, Y_b, X_c)
            if it % 50 == 0:
                params = self.get_params(opt_state)
                train_loss_value = loss(params, s, X, Y, X_c)
                train_loss.append(train_loss_value)
                to_print = "it %i, train loss = %e" % (it, train_loss_value)
                pbar.set_description(f"{to_print}")
                # if train_loss_value<stop:break
                if jnp.isnan(train_loss_value):break
        params = self.get_params(opt_state)
        s = self.leastsquares(loss, params, s, X_b, Y_b, X_c)
        params[0][-1] = s
        return params, train_loss, val_loss

