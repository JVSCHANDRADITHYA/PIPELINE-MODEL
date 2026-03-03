# core/classification.py

from core.config import MAIN_SENSORS, OP_KEYS


def classify_sensors(columns):
    group_map = {
        "P": [],
        "T": [],
        "F": [],
        "OP": []
    }

    for col in columns:

        # Skip meta columns
        if col in ["Seconds"]:
            continue

        assigned = False

        # Main groups
        for group, keywords in MAIN_SENSORS.items():
            for key in keywords:
                if key in col.upper():
                    group_map[group].append(col)
                    assigned = True
                    break
            if assigned:
                break

        # Operational group
        if not assigned:
            for key in OP_KEYS:
                if key in col.upper():
                    group_map["OP"].append(col)
                    assigned = True
                    break

    return group_map
