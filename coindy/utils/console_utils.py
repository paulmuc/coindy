from typing import Union

import matplotlib.pyplot as plt
import sympy as sym
from matplotlib.backends.backend_agg import FigureCanvasAgg


def progress_bar(progress: float, total: float):
    """ Prints a command line progress bar
    :param progress: Float indicating the progress with respect to total progress
    :param total: Float indicating the total progress
    """
    percent = 100 * (progress / float(total))
    bar = '█' * int(percent) + '-' * (100 - int(percent))
    if int(percent) == 100:
        print(f"\r|{bar}| {percent:.2f}%", end="\n\n")
    else:
        print(f"\r|{bar}| {percent:.2f}%", end="\r")


def display_matrix(matrix: Union[list[str], list[list[str]], sym.Matrix], name=''):
    if isinstance(matrix, sym.matrices.dense.DenseMatrix):
        convert_to_latex(matrix, name, color='black')
    else:
        for i in range(0, len(matrix)):
            print(end='\n')
            print('|', end='  ')
            for j in range(0, len(matrix[0])):
                print(matrix[i][j], end='  ')
            print('|', end='\r')
    print(end='\n\n')


def convert_to_latex(text, name='', color='white'):
    if isinstance(text, sym.Basic):
        latex_string = sym.latex(text)
        latex_string = '$' + name + '=' + latex_string + '$'
    else:
        latex_string = text
    mathTex_to_QPixmap(latex_string, fontsize=16, color=color)


def mathTex_to_QPixmap(mathTex, **kwargs):
    # ---- set up a mpl figure instance ----
    plt.rcParams.update({
        "text.usetex": True})
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}'
                                  r'\usepackage{amssymb}')

    fig = plt.figure()
    fig.patch.set_facecolor('none')
    fig.set_canvas(FigureCanvasAgg(fig))
    renderer = fig.canvas.get_renderer()

    # ---- plot the mathTex expression ----

    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.patch.set_facecolor('None')
    t = ax.text(0, 0.5, mathTex, ha='left', va='bottom', **kwargs)

    # ---- fit figure size to text artist ----

    fwidth, fheight = fig.get_size_inches()
    fig_bbox = fig.get_window_extent(renderer)

    text_bbox = t.get_window_extent(renderer)

    tight_fwidth = text_bbox.width * fwidth / fig_bbox.width
    tight_fheight = text_bbox.height * fheight / fig_bbox.height

    fig.set_size_inches(tight_fwidth, tight_fheight, forward=True)

    plt.show()


