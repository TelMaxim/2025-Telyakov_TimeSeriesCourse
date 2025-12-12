import numpy as np
import math
import copy
import heapq

from modules.utils import sliding_window, z_normalize
from modules.metrics import DTW_distance


# --- Вспомогательные функции (определяем здесь, чтобы не зависеть от версий в utils.py) ---

def apply_exclusion_zone(array: np.ndarray, idx: int, excl_zone: int) -> np.ndarray:
    """
    Apply an exclusion zone to an array (inplace)
    """
    zone_start = max(0, idx - excl_zone)
    zone_stop = min(array.shape[-1], idx + excl_zone)
    array[zone_start: zone_stop + 1] = np.inf
    return array


def topK_match(dist_profile: np.ndarray, excl_zone: int, topK: int = 3, max_distance: float = np.inf) -> dict:
    """
    Search the topK match subsequences based on distance profile
    """
    topK_match_results = {
        'indices': [],
        'distances': []
    }

    dist_profile_len = len(dist_profile)
    # Работаем с копией
    dist_profile = np.copy(dist_profile).astype(float)

    for k in range(topK):
        # Находим индекс минимума
        min_idx = np.argmin(dist_profile)
        min_dist = dist_profile[min_idx]

        # Если минимум плохой (inf/nan) или больше порога - останавливаемся
        if (np.isnan(min_dist)) or (np.isinf(min_dist)) or (min_dist > max_distance):
            break

        # Применяем зону исключения
        dist_profile = apply_exclusion_zone(dist_profile, min_idx, excl_zone)

        topK_match_results['indices'].append(int(min_idx))
        topK_match_results['distances'].append(float(min_dist))

    return topK_match_results


# --- Классы ---

class BestMatchFinder:
    """
    Base Best Match Finder
    """

    def __init__(self, excl_zone_frac: float = 1, topK: int = 3, is_normalize: bool = True, r: float = 0.05) -> None:
        self.excl_zone_frac: float = excl_zone_frac
        self.topK: int = topK
        self.is_normalize: bool = is_normalize
        self.r: float = r

    def _calculate_excl_zone(self, m: int) -> int:
        excl_zone = math.ceil(m * self.excl_zone_frac)
        return excl_zone

    def perform(self):
        raise NotImplementedError


class NaiveBestMatchFinder(BestMatchFinder):
    """
    Naive Best Match Finder
    """

    def __init__(self, excl_zone_frac: float = 1, topK: int = 3, is_normalize: bool = True, r: float = 0.05):
        super().__init__(excl_zone_frac, topK, is_normalize, r)

    def perform(self, ts_data: np.ndarray, query: np.ndarray) -> dict:
        query = copy.deepcopy(query)

        # Подготовка данных (разбиение на окна)
        if (len(ts_data.shape) != 2):
            ts_data = sliding_window(ts_data, len(query), 1)

        N, m = ts_data.shape
        excl_zone = self._calculate_excl_zone(m)

        # Профиль расстояний
        dist_profile = np.full(N, np.inf)

        # Нормализация запроса один раз
        if self.is_normalize:
            query = z_normalize(query)

        # Основной цикл
        for i in range(N):
            subseq = ts_data[i]

            if self.is_normalize:
                subseq = z_normalize(subseq)

            # Считаем DTW
            dist = DTW_distance(query, subseq, r=self.r)
            dist_profile[i] = dist

        # Поиск лучших совпадений
        # topK_match возвращает словарь {'indices': [...], 'distances': [...]}
        bestmatch = {
            'index': [],
            'distance': []
        }

        found_matches = topK_match(dist_profile, excl_zone, self.topK)

        # Перекладываем результаты в формат вывода
        bestmatch['index'] = found_matches['indices']
        bestmatch['distance'] = found_matches['distances']

        return bestmatch


class UCR_DTW(BestMatchFinder):
    """
    UCR-DTW Match Finder
    """

    def __init__(self, excl_zone_frac: float = 1, topK: int = 3, is_normalize: bool = True, r: float = 0.05):
        super().__init__(excl_zone_frac, topK, is_normalize, r)
        self.not_pruned_num = 0
        self.lb_Kim_num = 0
        self.lb_KeoghQC_num = 0
        self.lb_KeoghCQ_num = 0

    def _LB_Kim(self, subs1: np.ndarray, subs2: np.ndarray) -> float:
        """
        Compute LB_Kim lower bound (First + Last points)
        """
        # LB_KimFL: Квадрат разности первой и последней точки
        d1 = (subs1[0] - subs2[0]) ** 2
        d2 = (subs1[-1] - subs2[-1]) ** 2
        return d1 + d2

    def _LB_Keogh(self, subs1: np.ndarray, subs2: np.ndarray, r: float) -> float:
        """
        Compute LB_Keogh lower bound between subs1 and subs2.
        Calculates envelope around subs2 and checks distance from subs1 to that envelope.
        """
        lb_Keogh = 0
        m = len(subs2)
        w = int(np.floor(r * m))

        # В цикле вычисляем конверт и накапливаем расстояние
        for i in range(m):
            # Границы окна
            start = max(0, i - w)
            end = min(m, i + w + 1)

            # Конверт (минимум и максимум в окне)
            lower_bound = np.min(subs2[start:end])
            upper_bound = np.max(subs2[start:end])

            # Если точка выходит за конверт, добавляем штраф
            if subs1[i] > upper_bound:
                lb_Keogh += (subs1[i] - upper_bound) ** 2
            elif subs1[i] < lower_bound:
                lb_Keogh += (lower_bound - subs1[i]) ** 2

        return lb_Keogh

    def get_statistics(self) -> dict:
        statistics = {
            'not_pruned_num': self.not_pruned_num,
            'lb_Kim_num': self.lb_Kim_num,
            'lb_KeoghCQ_num': self.lb_KeoghCQ_num,
            'lb_KeoghQC_num': self.lb_KeoghQC_num
        }
        return statistics

    def perform(self, ts_data: np.ndarray, query: np.ndarray) -> dict:
        query = copy.deepcopy(query)
        if (len(ts_data.shape) != 2):
            ts_data = sliding_window(ts_data, len(query), 1)

        N, m = ts_data.shape
        excl_zone = self._calculate_excl_zone(m)
        dist_profile = np.full(N, np.inf)

        # BSF (Best So Far) - это порог отсечения.
        # Для Top-K нам нужно поддерживать K лучших значений.
        # Изначально порог - бесконечность.
        bsf = np.inf
        # Храним найденные лучшие дистанции, чтобы динамически обновлять bsf
        best_distances_heap = []

        # 1. Нормализация запроса
        if self.is_normalize:
            query = z_normalize(query)

        # Сброс статистики
        self.not_pruned_num = 0
        self.lb_Kim_num = 0
        self.lb_KeoghQC_num = 0
        self.lb_KeoghCQ_num = 0

        # 2. Основной цикл
        for i in range(N):
            subseq = ts_data[i]

            if self.is_normalize:
                subseq = z_normalize(subseq)

            # --- Каскад Lower Bounds ---

            # 1. LB_Kim
            lb_kim_val = self._LB_Kim(query, subseq)
            if lb_kim_val > bsf:
                self.lb_Kim_num += 1
                continue  # Pruned

            # 2. LB_Keogh QC (Query enveloped)
            # Примечание: для оптимизации конверт запроса лучше считать 1 раз вне цикла,
            # но здесь используем метод класса для простоты структуры
            lb_keogh_qc_val = self._LB_Keogh(subseq, query, self.r)
            if lb_keogh_qc_val > bsf:
                self.lb_KeoghQC_num += 1
                continue  # Pruned

            # 3. LB_Keogh CQ (Candidate enveloped)
            lb_keogh_cq_val = self._LB_Keogh(query, subseq, self.r)
            if lb_keogh_cq_val > bsf:
                self.lb_KeoghCQ_num += 1
                continue  # Pruned

            # 4. Если не отсекли - считаем полный DTW
            self.not_pruned_num += 1
            dist = DTW_distance(query, subseq, r=self.r)
            dist_profile[i] = dist

            # --- Обновление порога BSF ---
            if dist < bsf:
                # Если список еще не полон, просто добавляем
                if len(best_distances_heap) < self.topK:
                    heapq.heappush(best_distances_heap, -dist)  # храним отрицательные для max-heap
                else:
                    # Если нашли что-то лучше, чем худшее из лучших
                    # -best_distances_heap[0] это самое большое расстояние в топе (худшее)
                    if dist < -best_distances_heap[0]:
                        heapq.heapreplace(best_distances_heap, -dist)

                # Обновляем bsf как "худшее" из топ-K
                if len(best_distances_heap) == self.topK:
                    bsf = -best_distances_heap[0]

        # Формируем ответ
        bestmatch = {
            'index': [],
            'distance': []
        }

        found_matches = topK_match(dist_profile, excl_zone, self.topK)
        bestmatch['index'] = found_matches['indices']
        bestmatch['distance'] = found_matches['distances']

        return bestmatch
