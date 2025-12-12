import numpy as np
import pandas as pd
import stumpy
from stumpy import config

def compute_mp(ts1, m: int, exclusion_zone: int = None, ts2 = None):
    """
    Compute the matrix profile.
    Robust version: handles Pandas objects and forces float conversion.
    """

    # --- БЛОК ПРЕДОБРАБОТКИ ДАННЫХ ---
    # 1. Если ts1 - это DataFrame или Series, превращаем в одномерный массив numpy
    if isinstance(ts1, (pd.Series, pd.DataFrame)):
        ts1 = ts1.squeeze().to_numpy()
    # 2. Принудительно приводим к float (защита от object/int ошибок)
    ts1 = ts1.astype(float)

    # То же самое для ts2, если он передан
    if ts2 is not None:
        if isinstance(ts2, (pd.Series, pd.DataFrame)):
            ts2 = ts2.squeeze().to_numpy()
        ts2 = ts2.astype(float)
    # ---------------------------------

    # Установка зоны исключения по умолчанию
    if exclusion_zone is None:
        exclusion_zone = int(np.ceil(m / 2))

    # Хак для изменения зоны исключения в stumpy (как в вашем исходном коде)
    original_excl_zone_denom = config.STUMPY_EXCL_ZONE_DENOM
    if exclusion_zone > 0:
        # Избегаем деления на ноль, если exclusion_zone очень маленькая
        try:
            config.STUMPY_EXCL_ZONE_DENOM = m / exclusion_zone
        except ZeroDivisionError:
            pass

    # --- ВЫЧИСЛЕНИЕ ---
    if ts2 is None:
        # Self-join
        mp = stumpy.stump(ts1, m)
    else:
        # AB-join
        # ignore_trivial=False обязательно при сравнении разных рядов
        mp = stumpy.stump(ts1, m, T_B=ts2, ignore_trivial=False)

    # Восстанавливаем настройки stumpy
    config.STUMPY_EXCL_ZONE_DENOM = original_excl_zone_denom

    return {'mp': mp[:, 0],
            'mpi': mp[:, 1],
            'm': m,
            'excl_zone': exclusion_zone,
            'data': {'ts1': ts1, 'ts2': ts2}
            }