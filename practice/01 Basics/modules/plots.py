import matplotlib.pyplot as plt
import numpy as np


def plot_ts(ts_set: np.ndarray, plot_title: str = 'Набор входных временных рядов'):
    """
    Plot the time series set using Matplotlib (More stable than Plotly)
    """

    # Если массив одномерный (один ряд), делаем его двумерным (1, N)
    if ts_set.ndim == 1:
        ts_set = ts_set.reshape(1, -1)

    # Создаем фигуру
    plt.figure(figsize=(12, 5))

    # Рисуем каждый временной ряд
    for i in range(ts_set.shape[0]):
        plt.plot(ts_set[i], label=f"Временной ряд {i}", linewidth=2)

    # Настройки оформления
    plt.title(plot_title, fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Values", fontsize=12)

    # Если рядов немного (меньше 10), показываем легенду, иначе она засорит график
    if ts_set.shape[0] < 10:
        plt.legend()

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Отображаем
    plt.show()
