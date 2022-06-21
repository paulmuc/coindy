# CoinDy - Stochastic Differential Equations

CoinDy is a package allowing the symbolic computation of Stochastic Differential Equations and their simulation for mechanical as well as Itō systems.
This package can help in deriving the complex derivative terms needed to apply explicit algorithms such as the ones described by Kloeden and Platen. It can also be used as a teaching tool, to confirm hand-derived formulas or to obtain quick results.

Build Status: Passed

## Installation

CoinDy is not live yet, it is available on TestPyPI though

    pip install -i https://test.pypi.org/pypi/ --extra-index-url https://pypi.org/simple coindy

It is for now supported on Python 3.9 and over.

## Theory

The first Kolmogorov operator:
$$\mathrm{d}g = \left(\frac{\partial g}{\partial t} + \sum_{i=1}^{p} a_i \frac{\partial g}{\partial y_{i}} + \frac{1}{2} \sum_{i=1}^{p} \sum_{j=1}^p \sum_{k=1}^n B_{ij} B_{kj} \frac{\partial^2 g}{\partial y_i \partial y_j}\right)\mathrm{d}t \\ + \sum_{i=1}^p \sum_{j=1}^n B_{ij} \frac{\partial g}{\partial y_i} \mathrm{d}w_j$$

## Classes

- `SDEModel(n_dof, n_rvs, algorithm='all')` - Main class, which performs the computation of derivative terms need for explicit integration schemes and their simulation. It makes use of the utilities listed below to provide an easy to manipulate object. For a more detailed investigation of symbolic terms, the utilities can be used. A use case for using `SDEModel` is demonstrated in `coindy.sde_demo.py`.

## Utility functions

In `coindy.ito_utils`:
- `a, B = ito_sde_form(M, C, K, f, x, B_init)` - Translates a system from a mechanical formulation to an Itō SDE formulation. The terms `a` and `B` represent the drift vector and diffusion matrix respectively where $d\mathbf{x} = \mathbf{a}*dt + \mathbf{B} * d\mathbf{w}_t$. These results would be similar to the ones used by querying `SDEModel.sde_terms['a']` and `SDEModel.sde_terms['B']`.
- `L0(X, a, B, x, t)` - Compute the result of applying the first Kolmogorov operator to vector `X` with `a` and `B` resulting from `ito_sde_form`.
- `LJ(X, B, x, j)` - Compute the result of applying the second Kolmogorov operator to `X` where `X` can be a vector or a matrix at index `j`. 

In `coindy.simulation_utils`:
- `dW = generate_wiener_increment(time_step, time_stop, n_dof)` - Utility that generates an n_dof * (time_stop/time_step) matrix of Wiener increments

## Scripts

- `sde_demo.py` - Script demonstrating the use of SDEModel for derivation and simulation of Itо̄ SDEs.

## Example:
Integrate a one-dimensional mechanical oscillator with mass 1kg, damping 2.5 N s/m, stiffness 5 N/m
and stochastic force amplitude 0.01 N. The initial conditions are ``x0 = 0.01, x1 = 0``.

```python
from coindy import SDEModel

n_dof = 1
n_rvs = 1

equations = {'M': [['m']], 'C': [['c']], 'K': [['k']], 'f': [['0', '0', 's']]}

constant_map = {'m': 1, 'c': 2.5, 'k': 5, 's': 0.01}

initial_values = {'x0': 0.01, 'x1': 0}

SDEModel.show_progress = True

sde_model = SDEModel(n_dof, n_rvs)

sde_model.equations = equations

sde_model.compute_ito_sde_terms()
    
# Print drift term
print(sde_model.sde_terms['a'])

sde_model.simulate([0.01, 10], constant_map, initial_values)
```

## References
 - Kloeden, P. E., Platen, E. and Schurz, H. (2003) [*Numerical Solution of SDE Through Computer Experiments*](https://doi.org/10.1007/978-3-642-57913-4). Berlin, Heidelberg: Springer Berlin Heidelberg (Universitext).
 - Kloeden, P. E. and Platen, E. (1992) [*Numerical Solution of Stochastic Differential Equations*](https://doi.org/10.1007/978-3-662-12616-5), Springer. Berlin, Heidelberg: Springer Berlin Heidelberg.
 - Roy, D. and Visweswara Rao, G. (2017) [*Stochastic dynamics, filtering and optimization, Stochastic Dynamics, Filtering and Optimization*](https://doi.org/10.1017/9781316863107).
 - Cyganowski, S., Kloeden, P. and Ombach, J. (2002) [*From Elementary Probability to Stochastic Differential Equations with MAPLE®*](https://doi.org/10.1007/978-3-642-56144-3). Berlin, Heidelberg: Springer Berlin Heidelberg (Universitext).
