import numpy as np

from modules.metrics import ED_distance, norm_ED_distance, DTW_distance
from modules.utils import z_normalize


class PairwiseDistance:
    """
    Distance matrix between time series 

    Parameters
    ----------
    metric: distance metric between two time series
            Options: {euclidean, dtw}
    is_normalize: normalize or not time series
    """

    def __init__(self, metric: str = 'euclidean', is_normalize: bool = False) -> None:

        self.metric: str = metric
        self.is_normalize: bool = is_normalize
    

    @property
    def distance_metric(self) -> str:
        """Return the distance metric

        Returns
        -------
            string with metric which is used to calculate distances between set of time series
        """

        norm_str = ""
        if (self.is_normalize):
            norm_str = "normalized "
        else:
            norm_str = "non-normalized "

        return norm_str + self.metric + " distance"


    def _choose_distance(self):
        """ Choose distance function for calculation of matrix
        
        Returns
        -------
        dict_func: function reference
        """

        dist_func = None

        if self.metric == 'euclidean':
            dist_func = ED_distance
        elif self.metric == 'dtw':
            dist_func = DTW_distance
        elif self.metric == 'norm_euclidean':
            dist_func = norm_ED_distance
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        return dist_func


    def calculate(self, input_data: np.ndarray) -> np.ndarray:
        """ Calculate distance matrix
        
        Parameters
        ----------
        input_data: time series set
        
        Returns
        -------
        matrix_values: distance matrix
        """
        
        matrix_shape = (input_data.shape[0], input_data.shape[0])
        matrix_values = np.zeros(shape=matrix_shape)

        n_samples = input_data.shape[0]

        # Получаем функцию расстояния
        dist_func = self._choose_distance()

        # Предварительная нормализация данных
        # Если включена нормализация И метрика не является norm_euclidean (которая нормализует внутри себя),
        # то нормализуем весь массив заранее для ускорения вычислений.
        if self.is_normalize and self.metric != 'norm_euclidean':
            # Применяем z_normalize к каждому ряду
            X_processed = np.array([z_normalize(x) for x in input_data])
        else:
            X_processed = input_data

        # Заполнение матрицы (только верхний треугольник + симметрия)
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                # Вычисление расстояния
                dist = dist_func(X_processed[i], X_processed[j])

                # Заполнение симметричных ячеек
                matrix_values[i, j] = dist
                matrix_values[j, i] = dist

        return matrix_values
