from rich import print


def main():
    print(u'This is CoinDy, a small library for computing simulations of systems under random excitation using SDEs.\n\n'
          'It enables the formulation of equations (from a mechanical system of It\u014d SDE formulation) and their \nsolving'
          ' using multiple explicit SDE integration algorithms with increasing convergence to the real solution.\nThe '
          'package makes use of one main class, [magenta]SDEModel[/magenta]. One can set nature of the SDE using either a mass-damping\nstiffness formulation '
          'characteristic of mechanical systems or the standard form of an SDE, using the drift and diffusion terms.\n'
          'An example is provided in [italic][magenta]coindy/sde_demo.py[/magenta][/italic].')


if __name__ == '__main__':
    main()
