# core/preprocessing.py

import numpy as np


def filter_static_sensors(df, sensors, N=100, eps=1e-6):
    healthy = []
    excluded = []

    for s in sensors:
        std_val = df[s].iloc[:N].std()

        if np.isnan(std_val) or std_val < eps:
            excluded.append(s)
        else:
            healthy.append(s)

    return healthy, excluded