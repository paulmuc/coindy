# CoinDy - Stochastic Differential Equations

CoinDy is a package allowing the symbolic computation of Stochastic Differential Equations and their simulation for mechanical as well as Itō systems.

Build Status: Passed

## Installation

CoinDy is not live yet, it is available on TestPyPI though

    pip install -i https://test.pypi.org/simple/ coindy

It is for now supported on Python 3.9.

## Classes

`SDEModel(n_dof, n_rvs, algorithm='all')` - Main class, which performs the computation of derivative terms need for explicit integration schemes. A use case is demonstrated in coindy.sde_demo.py

## Utility functions

`generate_wiener_increment(time_step, time_stop, n_dof)` - Utility that generates a n_dof * (time_stop/time_step) matrix of Wiener increments)

## Example:
Integrate a one-dimensional mechanical oscillator with mass 1kg, damping 2.5 N s/m, stiffness 5 N/m
and stochastic force amplitude 0.01 N. The initial conditions are ``x0 = 0.01, x1 = 0``.

```
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

    sde_model.simulate([0.01, 10], constant_map, initial_values)
```
