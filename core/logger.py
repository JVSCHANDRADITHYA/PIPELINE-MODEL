# core/logger.py

import csv


class SensorLogger:
    def __init__(self, sensor_list, wide_path, long_path):
        self.sensor_list = sensor_list

        self.wide_file = open(wide_path, "w", newline="")
        self.long_file = open(long_path, "w", newline="")

        self.wide_writer = csv.writer(self.wide_file)
        self.long_writer = csv.writer(self.long_file)

        # Write headers
        self.wide_writer.writerow(["t"] + sensor_list)
        self.long_writer.writerow(["t", "sensor", "state"])

    def log(self, t, state_dict):
        """
        state_dict: {sensor_name: state_string}
        """

        # --- Wide format ---
        row = [t] + [state_dict.get(s, "HEALTHY") for s in self.sensor_list]
        self.wide_writer.writerow(row)

        # --- Long format ---
        for s, state in state_dict.items():
            self.long_writer.writerow([t, s, state])

    def close(self):
        self.wide_file.close()
        self.long_file.close()