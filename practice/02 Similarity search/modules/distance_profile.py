import numpy as np

from modules.utils import z_normalize
from modules.metrics import ED_distance, norm_ED_distance


def brute_force(ts: np.ndarray, query: np.ndarray, is_normalize: bool = True) -> np.ndarray:
    """
    Calculate the distance profile using the brute force algorithm

    Parameters
    ----------
    ts: time series
    query: query, shorter than time series
    is_normalize: normalize or not time series and query

    Returns
    -------
    dist_profile: distance profile between query and time series
    """

    n = len(ts)
    m = len(query)
    N = n-m+1

    dist_profile = np.zeros(shape=(N,))

    # 1. Нормализация запроса (строка 2-3 псевдокода)
    curr_query = query.copy()
    if is_normalize:
        curr_query = z_normalize(curr_query)

    # 2. Основной цикл 
    for i in range(N):
        # Выделение подпоследовательности
        subsequence = ts[i: i + m]

        # Нормализация подпоследовательности
        if is_normalize:
            # Важно: z_normalize возвращает копию, исходный ряд ts не меняется
            curr_sub = z_normalize(subsequence)
        else:
            curr_sub = subsequence

        dist = ED_distance(curr_query, curr_sub)

        dist_profile[i] = dist

    return dist_profile
