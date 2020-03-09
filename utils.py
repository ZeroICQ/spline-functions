import numpy as np
import matplotlib.pyplot as plt
import math


def prepare_plot():
    plt.xlabel('x')
    plt.ylabel('y')

    plt.legend()

    plt.axhline(0, lw=0.5, color="black")
    plt.axvline(0, lw=0.5, color="black")
