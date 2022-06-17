import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from rich import print

from coindy.sde import SDEModel


def update_lines(num, data_p, traces_p, max_num_points_p):
    traces_p[0].set_data(data_p[:, 0, num], data_p[:, 1, num])
    traces_p[0].set_3d_properties(data_p[:, 2, num])
    if num < max_num_points_p:
        plot_ind = range(0, num)
    else:
        plot_ind = range(num - max_num_points_p, num)
    traces_p[1].set_data(data_p[0, 0, plot_ind], data_p[0, 1, plot_ind])
    traces_p[1].set_3d_properties(data_p[0, 2, plot_ind])
    traces_p[2].set_data(data_p[1, 0, plot_ind], data_p[1, 1, plot_ind])
    traces_p[2].set_3d_properties(data_p[1, 2, plot_ind])

    return traces_p


if __name__ == '__main__':
    """
    This script demonstrates the use of SDEModel and SDESimulator to help compute a simulation of a given structural 
    system under random excitation using stochastic calculus techniques. 
    Here the example is a cart and pendulum system where a 2D pendulum is attached to a rolling cart. 
    All motion occurs with the y-z plane.
    """
    n_dof = 2  # Number of degrees of freedom
    n_rvs = 2  # Number of random variables

    # Create a dict of constants in you equations
    constant_map = {"m": 1,  # Mass of the pendulum ball
                    "M": 50,  # Mass of the cart
                    "L": 0.2022,  # Length of the pendulum rod
                    "Cx": 2.5,  # Damping of the cart
                    "ch_x": 0.7658 * (0.8 * 0.2022) ** 2,  # ch_x = c*h^2 where c is the damping of an auxiliary damper
                    # attached at the pendulum and h is 0.8*L (the point of attachment of the damper)
                    "Kx": 2500,  # Stiffness of the cart
                    "g": 9.81,  # Acceleration due to gravity
                    "s0": 1,  # Magnitude of noise (1st degree of freedom i.e. the cart displacement)
                    "s1": 0}  # Magnitude of noise (2nd degree of freedom i.e. the pendulum angle)

    # Equations in MCK format (structural systems) ---------------------------------------------------------------------
    # Write equations in a string format, if it is a matrix, it can be written as a 2D matrix in the form of nested
    # lists
    M_str = [["m+M", "m*L*cos(x2)"], ["m*L*cos(x2)", "m*L**2"]]
    C_str = [["Cx", "-sin(x2)*x3"], ["0", "ch_x*cos(x2)**2"]]
    K_str = [["Kx", "0"], ["0", "0"]]
    # Note that f has dimension n_dof x 2 + n_rvs. The first column of f_str is the system forcing (which sometimes,
    # due to nonlinearity, can't be formulated in the MCK format). The second column account for external deterministic
    # forcing (for example a sinusoidal excitation) and the last n_rvs columns are for the stochastic forcing matrix B
    f_str = [["0", "0", "s0", "0"], ["-m*L*g*sin(x2)", "0", "0", "s1"]]

    # Uncomment for SDE format -----------------------------------------------------------------------------------------
    # a_str = [["x1"],
    #          ["(L*g*m*sin(x2)*cos(x2) + L*(-Cx*x1 - Kx*x0 + x3**2*sin(x2)) + ch_x*x3*cos(x2)**3)/(L*(M + m*sin("
    #              "x2)**2))"],
    #          ["x3"],
    #          ["(L*m*(Cx*x1 + Kx*x0 - x3**2*sin(x2))*cos(x2) - (M + m)*(L*g*m*sin(x2) + ch_x*x3*cos(x2)**2))/(L**2*m*(M "
    #           "+ m*sin(x2)**2))"]]
    #
    # B_str = [["0", "0"],
    #          ["s0/(M + m*sin(x2)**2)", "-s1*cos(x2)/(L*(M + m*sin(x2)**2))"],
    #          ["0", "0"],
    #          ["-s0*cos(x2)/(L*(M + m*sin(x2)**2))", "s1*(M + m)/(L**2*m*(M + m*sin(x2)**2))"]]

    # Activate progress printing in console
    SDEModel.show_progress = True

    # Create SDEModel instance with number of degrees of freedom and number of random variables (Wiener processes)
    sde_model = SDEModel(n_dof, n_rvs)

    # Set the SDEModel equations
    # sde_model.equations = {'a': a_str, 'B': B_str}
    sde_model.equations = {'M': M_str, 'C': C_str, 'K': K_str, 'f': f_str}

    # Compute the Ito SDE terms
    sde_model.compute_ito_sde_terms()

    # Set initial values for state variables (e.g. at time = 0)
    initial_values = {'x0': 0, 'x1': 0, 'x2': 0, 'x3': 0}
    dt = 0.01
    T = 30
    Nt = int(T / dt)

    sde_model.simulate([dt, T], constant_map, initial_values)

    # Extract simulation results
    # Some simulations may fail, in which case the flag in flags corresponding to the simulation technique
    # (Euler-Maruyama, Milstein or Ito-Taylor 1.5) will be set to False
    results = sde_model.results

    flags = results['failed_flags']

    techniques = {'em': 'Euler-Maruyama', 'mi': 'Milstein', 'it': 'It\u014d-Taylor 1.5'}
    for key, value in flags.items():
        if value == 0:
            outcome = ' failed\n'
        else:
            outcome = ' succeeded\n'
        print(f'[magenta]' + techniques[key] + '[/magenta]' + outcome)

    Y = results['it']  # Selecting only Ito-Taylor 1.5 results

    L = 0.2022
    px = L * np.sin(Y[2, :])
    pz = -L * np.cos(Y[2, :])
    py = np.zeros(Y[0, :].shape)

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    data = np.array([[Y[0, :], py, py], [Y[0, :] + px, py, pz]])

    traces = list()
    traces.append(ax.plot(data[:, 0, 0], data[:, 1, 0], data[:, 2, 0], marker='o', mfc='r', color='b')[0])
    traces.append(ax.plot(data[0, 0, 0], data[0, 1, 0], data[0, 2, 0])[0])
    traces.append(ax.plot(data[1, 0, 0], data[1, 1, 0], data[1, 2, 0])[0])

    # Setting the axes properties
    ax.set_xlim3d([-L, L])
    ax.set_xlabel('X')

    ax.set_ylim3d([-L, L])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-L, L])
    ax.set_zlabel('Z')

    ax.set_title('2D Pendulum')

    max_num_points = 100
    # Creating the Animation object
    line_ani = animation.FuncAnimation(
        fig, update_lines, frames=Nt, fargs=(data, traces, max_num_points), interval=T, blit=True, repeat=True)

    plt.show()
