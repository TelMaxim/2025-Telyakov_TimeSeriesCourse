import numpy as np
from modules.utils import z_normalize


def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculates the Euclidean distance
    """
    return np.sqrt(np.sum((ts1 - ts2) ** 2))


def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculates the normalized Euclidean distance
    """
    ts1_norm = z_normalize(ts1)
    ts2_norm = z_normalize(ts2)
    return ED_distance(ts1_norm, ts2_norm)


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1.0) -> float:
    """
    Calculate DTW distance with Sakoe-Chiba band constraint

    Parameters
    ----------
    ts1: first time series
    ts2: second time series
    r: warping window size (fraction of length, 0.0 to 1.0)
    """
    n = len(ts1)
    m = len(ts2)

    # Вычисляем ширину окна в индексах
    # Обычно берется доля от длины ряда. Если r=1, окно покрывает всё.
    w = int(np.floor(r * max(n, m)))

    # Инициализация матрицы бесконечностью
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        # Оптимизация Сако-Чиба:
        # j бежит не от 1 до m, а только внутри окна вокруг i
        start_j = max(1, i - w)
        end_j = min(m, i + w) + 1

        for j in range(start_j, end_j):
            # Квадрат разности (Squared Euclidean) для совместимости с sktime
            cost = (ts1[i - 1] - ts2[j - 1]) ** 2

            last_min = min(dtw_matrix[i - 1, j],  # вставка
                           dtw_matrix[i, j - 1],  # удаление
                           dtw_matrix[i - 1, j - 1])  # совпадение

            dtw_matrix[i, j] = cost + last_min

    return dtw_matrix[n, m]
