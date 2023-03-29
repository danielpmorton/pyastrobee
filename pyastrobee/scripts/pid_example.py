import numpy as np
import matplotlib.pyplot as plt

from pyastrobee.control.pid import PID
from pyastrobee.utils.plotting import plot_controller_history


def SISO_example():
    kp = 0.3
    ki = 0
    kd = 0.02
    dt = 0.1
    controller = PID(kp, ki, kd, -1.0, 1.0)
    target = 10
    x = 0
    history = [x]
    for _ in range(20):
        controller.update(target - x, dt)
        x += controller.cmd
        history.append(x)

    plt.plot(range(len(history)), [target] * len(history), "k--")
    plt.plot(range(len(history)), history, "b")
    plt.show()


def MIMO_example():
    np.random.seed(0)
    n = 10
    kp = np.random.rand(n)
    ki = 1 / 10 * np.random.rand(n)
    kd = 1 / 30 * np.random.rand(n)
    i_min = -10 * np.random.rand(n)
    i_max = 10 * np.random.rand(n)
    dt = 0.1
    controller = PID(kp, ki, kd, i_min, i_max)
    target = 10 * np.random.rand(n)
    x = np.zeros(n)
    n_timesteps = 20
    history = np.zeros((n_timesteps + 1, n))
    history[0, :] = x
    for i in range(1, n_timesteps + 1):
        controller.update(target - x, dt)
        x += controller.cmd
        history[i, :] = x

    plot_controller_history(history, target, kp, ki, kd)


if __name__ == "__main__":
    # SISO_example()
    MIMO_example()
