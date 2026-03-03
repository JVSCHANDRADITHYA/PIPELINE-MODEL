# core/peer_detection.py

import numpy as np


def peer_deviation_group(values_dict, sensors, z_threshold=3.0, delta=1e-6):
    """
    Peer deviation inside a single group.
    Returns:
        healthy_sensors
        deviating_dict {sensor: z_score}
    """

    if len(sensors) < 3:
        return sensors, {}

    vals = np.array([values_dict[s] for s in sensors])

    median = np.median(vals)
    sigma = np.std(vals) + delta

    healthy = []
    deviating = {}

    for i, s in enumerate(sensors):
        z = abs(vals[i] - median) / sigma
        if z > z_threshold:
            deviating[s] = z
        else:
            healthy.append(s)

    return healthy, deviating


def peer_deviation_all(values_dict, group_map, z_threshold=3.0):
    """
    Runs peer detection for P, T, F separately.
    """

    results = {}
    all_deviations = {}

    for group in ["P", "T", "F"]:
        healthy, deviating = peer_deviation_group(
            values_dict,
            group_map[group],
            z_threshold=z_threshold
        )

        results[group] = healthy
        all_deviations[group] = deviating

    return results, all_deviations