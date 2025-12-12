import numpy as np

from modules.utils import *


def top_k_motifs(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k motifs based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k : number of motifs

    Returns
    --------
    motifs: top-k motifs (left and right indices and distances)
    """

    motifs_idx = []
    motifs_dist = []

    # Извлекаем данные из структуры матричного профиля
    mp = matrix_profile['mp'].astype(np.float64).copy()  # Копия для модификации
    mpi = matrix_profile['mpi'].astype(np.int64)
    m = matrix_profile['m']
    excl_zone = matrix_profile['excl_zone']

    # Поиск top-k мотивов
    for _ in range(top_k):
        # Находим индекс с минимальным расстоянием в матричном профиле
        min_idx = np.argmin(mp)

        # Если все оставшиеся значения - бесконечность, прекращаем поиск
        if np.isinf(mp[min_idx]):
            break

        # Получаем индекс ближайшего соседа из индекса матричного профиля
        nn_idx = mpi[min_idx]

        # Сохраняем пару индексов (левый и правый) и расстояние между ними
        motifs_idx.append((min_idx, nn_idx))
        motifs_dist.append(mp[min_idx])

        # Применяем зону исключения вокруг обоих найденных индексов,
        # чтобы избежать тривиальных совпадений в следующих итерациях
        apply_exclusion_zone(mp, min_idx, excl_zone, np.inf)
        apply_exclusion_zone(mp, nn_idx, excl_zone, np.inf)

    return {
        "indices": motifs_idx,
        "distances": motifs_dist
    }