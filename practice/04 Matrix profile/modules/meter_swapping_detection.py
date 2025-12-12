import numpy as np
import datetime

import plotly
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
import plotly.express as px
plotly.offline.init_notebook_mode(connected=True)

from modules.mp import *


def heads_tails(consumptions: dict, cutoff, house_idx: list) -> tuple[dict, dict]:
    """
    Split time series into two parts: Head and Tail

    Parameters
    ---------
    consumptions: set of time series
    cutoff: pandas.Timestamp
        Cut-off point
    house_idx: indices of houses

    Returns
    --------
    heads: heads of time series
    tails: tails of time series
    """

    heads, tails = {}, {}
    for i in house_idx:
        heads[f'H_{i}'] = consumptions[f'House{i}'][consumptions[f'House{i}'].index < cutoff]
        tails[f'T_{i}'] = consumptions[f'House{i}'][consumptions[f'House{i}'].index >= cutoff]
    
    return heads, tails


def meter_swapping_detection(heads, tails, house_idx, m):
    """
    Detects meter swapping by calculating swap scores for all pairs.
    """
    min_score_val = np.inf
    min_score_struct = {}
    eps = 1e-6

    print(f"Starting detection among {len(house_idx)} houses...")

    for i in house_idx:
        head_key = f'H_{i}'
        tail_self_key = f'T_{i}'


        # 1. Знаменатель: Сравнение Head_i с родным Tail_i

        mp_self = compute_mp(heads[head_key], m, ts2=tails[tail_self_key])

        #Принудительно приводим к float перед проверкой isnan
        vals_self = mp_self['mp'].astype(float)

        # Берем минимум
        if np.all(np.isnan(vals_self)):
            min_dist_self = np.inf
        else:
            min_dist_self = np.nanmin(vals_self)

        for j in house_idx:
            if i == j:
                continue

            tail_cross_key = f'T_{j}'


            # 2. Числитель: Сравнение Head_i с чужим Tail_j

            mp_cross = compute_mp(heads[head_key], m, ts2=tails[tail_cross_key])

            # Принудительно приводим к float
            vals_cross = mp_cross['mp'].astype(float)

            if np.all(np.isnan(vals_cross)):
                min_dist_cross = np.inf
            else:
                min_dist_cross = np.nanmin(vals_cross)

            # 3. Расчет Score

            if np.isinf(min_dist_cross) or np.isinf(min_dist_self):
                score = np.inf
            else:
                score = min_dist_cross / (min_dist_self + eps)

            if np.isnan(score):
                continue

            # Обновление минимума
            if score < min_score_val:
                min_score_val = score
                min_score_struct = {
                    'i': i,
                    'j': j,
                    'score': score,
                    'mp_j': vals_cross  # Сохраняем уже сконвертированный массив
                }

    if not min_score_struct:
        print("WARNING: No valid swap score found. Returning placeholder.")
        return {'i': -1, 'j': -1, 'score': np.inf, 'mp_j': []}

    return min_score_struct


def plot_consumptions_ts(consumptions: dict, cutoff, house_idx: list):
    """
    Plot a set of input time series and cutoff vertical line

    Parameters
    ---------
    consumptions: set of time series
    cutoff: pandas.Timestamp
        Cut-off point
    house_idx: indices of houses
    """

    num_ts = len(consumptions)

    fig = make_subplots(rows=num_ts, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02)

    for i in range(num_ts):
        fig.add_trace(go.Scatter(x=list(consumptions.values())[i].index, y=list(consumptions.values())[i].iloc[:,0], name=f"House {house_idx[i]}"), row=i+1, col=1)
        fig.add_vline(x=cutoff, line_width=3, line_dash="dash", line_color="red",  row=i+1, col=1)

    fig.update_annotations(font=dict(size=22, color='black'))
    fig.update_xaxes(showgrid=False,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18, color='black'),
                     linewidth=2,
                     tickwidth=2)
    fig.update_yaxes(showgrid=False,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18), color='black',
                     zeroline=False,
                     linewidth=2,
                     tickwidth=2)

    fig.update_layout(title='Houses Consumptions',
                      title_x=0.5,
                      title_font=dict(size=26, color='black'),
                      plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor='rgba(0,0,0,0)', 
                      height=800,
                      legend=dict(font=dict(size=20, color='black'))
                      )

    fig.show()
