# ui/dashboard.py

import matplotlib.pyplot as plt
from collections import deque

plt.rcParams.update({
    "font.size": 9,
    "font.family": "DejaVu Sans"
})


class PipelineDashboard:

    def __init__(self, groups, window=200):

        self.groups = groups
        self.window = window
        self.time_buffer = deque(maxlen=window)

        # Data buffers
        self.raw_buffers = {"P": {}, "T": {}, "F": {}}

        self.pc_actual = {
            "P": deque(maxlen=window),
            "T": deque(maxlen=window),
            "F": deque(maxlen=window)
        }

        self.pc_recon = {
            "P": deque(maxlen=window),
            "T": deque(maxlen=window),
            "F": deque(maxlen=window)
        }

        self.err_buffer = deque(maxlen=window)
        self.thr_buffer = deque(maxlen=window)
        self.op_buffer = deque(maxlen=window)

        # ---- 3 ROW LAYOUT ----
        self.fig, self.ax = plt.subplot_mosaic(
            [
                ["P", "T", "F"],
                ["PC_ACT", "PC_REC", "PC_REC"],
                ["ERR", "OP", "STATE"]
            ],
            figsize=(18, 10),
            constrained_layout=True
        )

        self._init_axes()
        self._init_lines()

        plt.ion()
        plt.show()

    # -------------------------------------------------------------

    def _init_axes(self):

        for g in ["P", "T", "F"]:
            self.ax[g].set_title(f"{g} Sensors")
            self.ax[g].grid(True)

        self.ax["PC_ACT"].set_title("PC1 Sensors")
        self.ax["PC_REC"].set_title("PC1 Reconstructed (AE)")
        self.ax["ERR"].set_title("Reconstruction Error vs Threshold")
        self.ax["OP"].set_title("Operational Energy")
        self.ax["STATE"].set_title("Failing Sensors")

        self.ax["STATE"].axis("off")

    # -------------------------------------------------------------

    def _init_lines(self):

        # ---- RAW SENSOR LINES ----
        self.sensor_lines = {"P": {}, "T": {}, "F": {}}

        for g in ["P", "T", "F"]:
            for s in self.groups[g]:
                line, = self.ax[g].plot([], [], linewidth=0.7)
                self.sensor_lines[g][s] = line
                self.raw_buffers[g][s] = deque(maxlen=self.window)

        # ---- PC ACTUAL ----
        self.pc_act_lines = {}
        for g in ["P", "T", "F"]:
            line, = self.ax["PC_ACT"].plot([], [], label=g)
            self.pc_act_lines[g] = line
        self.ax["PC_ACT"].legend()

        # ---- PC RECONSTRUCTED ----
        self.pc_rec_lines = {}
        for g in ["P", "T", "F"]:
            line, = self.ax["PC_REC"].plot([], [], label=g)
            self.pc_rec_lines[g] = line
        self.ax["PC_REC"].legend()

        # ---- ERROR & THRESHOLD ----
        self.err_line, = self.ax["ERR"].plot([], [], label="Error")
        self.thr_line, = self.ax["ERR"].plot([], [], label="Threshold")
        self.ax["ERR"].legend()

        # ---- OPERATIONAL ----
        self.op_line, = self.ax["OP"].plot([], [], label="Operational")
        self.ax["OP"].legend()

    # -------------------------------------------------------------

    def update(self,
               t,
               row,
               excluded,
               deviating,
               pc_actual_vals,
               pc_recon_vals,
               E_t,
               theta_t,
               O_t):

        self.time_buffer.append(t)

        # ---------- RAW SENSOR UPDATE ----------
        for g in ["P", "T", "F"]:
            for s in self.groups[g]:
                self.raw_buffers[g][s].append(row[s])

                line = self.sensor_lines[g][s]
                line.set_data(self.time_buffer,
                              self.raw_buffers[g][s])

                # Color logic
                if s in excluded:
                    line.set_color("red")
                elif s in deviating:
                    line.set_color("orange")
                else:
                    line.set_color("green")

            self.ax[g].relim()
            self.ax[g].autoscale_view()

        # ---------- PC ACTUAL ----------
        for i, g in enumerate(["P", "T", "F"]):
            self.pc_actual[g].append(pc_actual_vals[i])
            self.pc_act_lines[g].set_data(
                self.time_buffer,
                self.pc_actual[g]
            )

        self.ax["PC_ACT"].relim()
        self.ax["PC_ACT"].autoscale_view()

        # ---------- PC RECON ----------
        for i, g in enumerate(["P", "T", "F"]):
            self.pc_recon[g].append(pc_recon_vals[i])
            self.pc_rec_lines[g].set_data(
                self.time_buffer,
                self.pc_recon[g]
            )

        self.ax["PC_REC"].relim()
        self.ax["PC_REC"].autoscale_view()

        # ---------- ERROR ----------
        self.err_buffer.append(E_t)
        self.thr_buffer.append(theta_t)

        self.err_line.set_data(self.time_buffer,
                               self.err_buffer)
        self.thr_line.set_data(self.time_buffer,
                               self.thr_buffer)

        self.ax["ERR"].relim()
        self.ax["ERR"].autoscale_view()

        # ---------- OPERATIONAL ----------
        self.op_buffer.append(O_t)
        self.op_line.set_data(self.time_buffer,
                              self.op_buffer)

        self.ax["OP"].relim()
        self.ax["OP"].autoscale_view()

        # ---------- FAILING SENSOR TABLE ----------
        self.ax["STATE"].clear()
        self.ax["STATE"].axis("off")
        self.ax["STATE"].set_title("Failing Sensors")

        rows = []

        for s in sorted(excluded):
            rows.append([s, "NON-OP"])

        for s in sorted(deviating):
            rows.append([s, "DEV"])

        if not rows:
            rows = [["None", "Healthy"]]

        rows = rows[:12]  # limit visible rows

        table = self.ax["STATE"].table(
            cellText=rows,
            colLabels=["Sensor", "State"],
            loc="center",
            cellLoc="left"
        )

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.1)

        # Color state column
        for i, row_data in enumerate(rows):
            if row_data[1] == "NON-OP":
                table[(i + 1, 1)].set_text_props(color="red")
            elif row_data[1] == "DEV":
                table[(i + 1, 1)].set_text_props(color="orange")

        plt.pause(0.001)