import numpy as np
from modules.utils import z_normalize

def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    ed_dist: euclidean distance between ts1 and ts2
    """
    
    ed_dist = 0

    # Формула: корень из суммы квадратов разностей
    ed_dist = np.sqrt(np.sum((ts1 - ts2) ** 2))

    return ed_dist


def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the normalized Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    norm_ed_dist: normalized Euclidean distance between ts1 and ts2s
    """

    norm_ed_dist = 0

    ts1_norm = z_normalize(ts1)
    ts2_norm = z_normalize(ts2)

    norm_ed_dist = ED_distance(ts1_norm, ts2_norm)


    return norm_ed_dist


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    """
    Calculate DTW distance

    Parameters
    ----------
    ts1: first time series
    ts2: second time series
    r: warping window size
    
    Returns
    -------
    dtw_dist: DTW distance between ts1 and ts2
    """

    dtw_dist = 0

    n = len(ts1)
    m = len(ts2)

    # Инициализация матрицы (n+1) x (m+1) значением бесконечность
    dtw_matrix = np.full((n + 1, m + 1), np.inf)

    # Базовый случай: расстояние между пустыми последовательностями равно 0
    dtw_matrix[0, 0] = 0

    # Заполнение матрицы
    for i in range(1, n + 1):
        for j in range(1, m + 1):

            cost = (ts1[i - 1] - ts2[j - 1]) ** 2

            # Рекуррентное соотношение: берем минимум из соседей
            last_min = min(dtw_matrix[i - 1, j],     # вставка
                           dtw_matrix[i, j - 1],     # удаление
                           dtw_matrix[i - 1, j - 1]) # совпадение

            dtw_matrix[i, j] = cost + last_min

    # Результат находится в последней ячейке
    dtw_dist = dtw_matrix[n, m]

    return dtw_dist
