import matplotlib.pyplot as plt
from cycler import cycler


def ide_plot():
    """plot settings for github dark theme
    """
    plt.rcParams["axes.facecolor"] = '#0d1117'
    plt.rcParams["figure.facecolor"] = '#0d1117'

    plt.rcParams['figure.dpi'] = 100

    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False

    plt.rcParams["axes.edgecolor"] = "#eef7f4"

    plt.rcParams["xtick.color"] = '#eef7f4'
    plt.rcParams["ytick.color"] = '#eef7f4'


    plt.rcParams["axes.labelcolor"] = '#eef7f4'

    plt.rcParams["grid.color"] = '#eef7f4'

    plt.rcParams["legend.frameon"] = False

    plt.rcParams["legend.labelcolor"] = 'w'
    plt.rcParams["axes.titlecolor"] = "w"

    plt.rcParams['axes.prop_cycle'] = cycler(color=['g', 'r', 'b', 'y', 'purple'])

def pres_plot():
    """plot settings for presentation theme
    """
    plt.rcParams["axes.facecolor"] = '#121313'
    plt.rcParams["figure.facecolor"] = '#121313'

    plt.rcParams['figure.dpi'] = 200

    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False

    plt.rcParams["axes.edgecolor"] = "#eef7f4"

    plt.rcParams["xtick.color"] = '#eef7f4'
    plt.rcParams["ytick.color"] = '#eef7f4'


    plt.rcParams["axes.labelcolor"] = '#eef7f4'

    plt.rcParams["grid.color"] = '#eef7f4'

    plt.rcParams["legend.frameon"] = False

    plt.rcParams["axes.titlecolor"] = '#eef7f4'

    plt.rcParams['axes.prop_cycle'] = cycler(color=['#121313', '#0068ff', '#0068ff', 'y'])



