# Matrices perfect-fit en última capa #

Consideramos $N_{\theta}$ una red neuronal dada por
$$
    N_{\theta}(x) = W^1 \sigma(W^0 x+ b^{0})
$$
donde $W^{0} \in \mathbb{R}^{n\times 1}$, $W^{1} \in \mathbb{R}^{1\times n}$ y $b^0 \in \mathbb{R}^{n}$.
La cantidad de neuronas de la capa oculta está dada por $n$ y $\theta$ denota los parámetros $W^0, W^1$  y $b^0$.

## Problema de Ajuste: Minimos Cuadrados ##

Sean $\vec x = (x_1, \ldots, x_N) \in \mathbb{R}^{N}$ y 
$\vec y = (y_1, \ldots, y_N) \in \mathbb{R}^N$ los datos
de ajuste, donde buscamos aproximar una función $f$ tal que $f(x_i) = y_i$.
La función de pérdida viene dada por
$$
\begin{aligned}
    L(\theta) 
    &=
    \sum_{i=1}^{N} \left( y_i - N_{\theta}(x_i) \right)^2 
    \\&= 
    \lVert \vec y \rVert_2^2 
    - 2 \sum_{i=1}^{N} y_i W^1 \sigma(W^0 x_i + b^0)
    + \sum_{i=1}^{N} \left( W^1 \sigma(W^0 x_i + b^0) \right)^2.
\end{aligned}
$$
denotemos por $\sigma_i = \sigma(W^0 x_i + b^0) \in \mathbb{R}^n$. Entonces,
$$
\begin{aligned}
    L(\theta) 
    &=
    \lVert \vec y \rVert_2^2 
    - 2 W^1 
    \begin{bmatrix}
        | & & |\\
        \sigma_1 & \cdots & \sigma_N\\
        | & & |
    \end{bmatrix} \vec y
    + 
    W^1 \left(\sum_{i=1}^{N} \sigma_i \sigma_i^T \right) (W^1)^T
    \\&\eqqcolon
    C + W^1 B + W^1 A [W^1]^T.
\end{aligned}
$$

**Observación** La matriz $A$ es simétrica.

Queremos obtener la matriz $W^1$ de tal forma que sea la que se mejor ajuste dado los datos
$W^1$ y $b^0$. Observando que:

* $\frac{d}{d W^1} L(\theta) = 0$ implica que $B + 2 W^1 A = 0$.
* $\frac{d^2}{d^2 W^1} L(\theta) = A$

Vemos que podemos obtener $A$ y $B$ a través de la función de pérdida. En cualquier caso,
obtenemos que $W^1 = - \frac{1}{2} A^{-1} B$ es la matriz óptima.

## Problema de PINNs: Laplaciano 1D ##

Consideramos la ecuación:
$$
    \frac{d^2}{d^2 x} u(x) = f(x)
    \quad
    \text{ en } \Omega \subset \mathbb{R}
$$
con $\Omega$ compacto y conexo.

La función de pérdida es:
$$
    L(\theta) = \frac{1}{N} \sum_{j=1}^{N} \left( \frac{d^2}{d^2 x} N_{\theta}(x_i) - f(x_i) \right)^2
$$
que aproxima la integral $\int_{\Omega} (\frac{d^2}{d^2x} u - f)^2$. 

Buscamos matrices $A, B$ y escalar $C$ tal que $L(\theta) = W^1 A [W^1]^T + W^1 B + C$.
Expandiendo:
$$
    L(\theta) 
    = \frac{1}{N} \sum_{i=1}^{N} [N_{\theta}''(x_i)]^2 - 2 N_{\theta}''(x_i)f(x_i) + f(x_i)^2
$$

Examinamos el término $N_{\theta}''(x)$:
$$
\begin{aligned}
    N_{\theta}''(x)
    &=
    \left( W^1 \sigma(W^0 x + b^0) \right)''
    \\&=
    \left( W^1 W^0 \sigma'(W^0 x + b^0) \right)'
    \\&=
    W^1 W^0 [W^0]^T \sigma''(W^0 x + b^0)
\end{aligned}
$$
Luego,
$$
    \sum_{i=1}^{N} N_{\theta}''(x_i)^2 
    =
    W^1 [W^0 [W^0]^T] \left( \sum_{i=1}^{N} \sigma''_i \sigma_i''^T \right) [W^1]^T
$$
donde $\sigma''_i = \sigma''_i(W^0 x_i + b^0)$. Así, obtenemos que
$$
    A = [W^0 [W^0]^T] \sum_{i=1}^{N} \sigma''_i \sigma_i''^T
    \quad
    B = 2 W^0 [W^0]^T 
    \begin{bmatrix} 
        | & & |\\
        \sigma''_1 & \cdots & \sigma''_2\\
        | & & |
    \end{bmatrix} 
    f(\vec x).
$$
