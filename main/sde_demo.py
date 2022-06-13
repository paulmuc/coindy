"""
Created on 4 avr. 2021

@author: Paul
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sympy import init_printing, sympify
import matplotlib.animation as animation
from rich import print

from main.sde_model_coindy import SDEModel
from main.sde_simulator_coindy import SDESimulator

matplotlib.use('QtAgg')


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
    init_printing()

    n_dof = 2
    n_rvs = 2

    const_map = {sympify("m"): 1, sympify("M"): 50, sympify("L"): 0.2022,
                 sympify("Cx"): 2.5, sympify("ch_x"): 0.7658 * (0.8 * 0.2022) ** 2,
                 sympify("Kx"): 2500, sympify("g"): 9.81, sympify("s0"): 1, sympify("s1"): 0}

    M_str = [["m+M", "m*L*cos(x2)"], ["m*L*cos(x2)", "m*L**2"]]
    C_str = [["Cx", "-sin(x2)*x3"], ["0", "ch_x*cos(x2)**2"]]
    K_str = [["Kx", "0"], ["0", "0"]]
    F_str = [["0", "0", "s0", "0"], ["-m*L*g*sin(x2)", "0", "0", "s1"]]

    SDEModel.show_progress = True
    sde_model = SDEModel(n_dof, n_rvs)

    sde_model.equations = {'M': M_str, 'C': C_str, 'K': K_str, 'F': F_str}

    sde_model.compute_ito_sde_terms()

    a, B, L0a, L0b, Lja, Ljb, L1L1b = sde_model.sde_terms.values()

    initial_values = {'x0': 0, 'x1': 0, 'x2': 0, 'x3': 0}
    dt = 0.01
    T = 10
    Nt = int(T / dt)

    SDESimulator.show_progress = True
    sde_sim = SDESimulator(sde_model, [dt, T], const_map, initial_values)

    sde_sim.simulate()

    time, Y, flags = sde_sim.results

    techniques_list = ['Euler-Maruyama', 'Milstein', 'It\u014d-Taylor 1.5']
    for i in range(0, len(flags)):
        if flags[i] == 0:
            outcome = ' failed\n'
        else:
            outcome = ' succeeded\n'
        print(f'[magenta]'+techniques_list[i]+'[/magenta]' + outcome)

    Y = Y[8:]  # Selecting only IT 1.5 results

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
