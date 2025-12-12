import numpy as np

from modules.utils import *


def top_k_discords(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k discords based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k: number of discords

    Returns
    --------
    discords: top-k discords (indices, distances to its nearest neighbor and the nearest neighbors indices)
    """

    discords_idx = []
    discords_dist = []
    discords_nn_idx = []

    # Извлекаем данные из структуры матричного профиля
    mp = matrix_profile['mp'].astype(np.float64).copy()  # Копия для модификации
    mpi = matrix_profile['mpi'].astype(np.int64)
    m = matrix_profile['m']
    excl_zone = matrix_profile['excl_zone']

    # Заменяем бесконечные значения на -бесконечность для корректного поиска максимума
    mp_search = mp.copy()
    mp_search[np.isinf(mp_search)] = -np.inf

    # Поиск top-k диссонансов
    for _ in range(top_k):
        # Находим индекс с максимальным расстоянием в матричном профиле
        max_idx = np.argmax(mp_search)

        # Если все оставшиеся значения - минус бесконечность, прекращаем поиск
        if np.isneginf(mp_search[max_idx]):
            break

        # Получаем индекс ближайшего соседа
        nn_idx = mpi[max_idx]

        # Сохраняем индекс диссонанса, расстояние и индекс ближайшего соседа
        discords_idx.append(max_idx)
        discords_dist.append(mp[max_idx])
        discords_nn_idx.append(nn_idx)

        # Применяем зону исключения вокруг найденного диссонанса,
        # чтобы избежать тривиальных совпадений в следующих итерациях
        apply_exclusion_zone(mp_search, max_idx, excl_zone, -np.inf)

    return {
        'indices': discords_idx,
        'distances': discords_dist,
        'nn_indices': discords_nn_idx
    }