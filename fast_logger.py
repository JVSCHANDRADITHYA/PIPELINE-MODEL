# main.py

import numpy as np
import torch

from core.ingestion import stream_csv
from core.classification import classify_sensors
from core.peer_detection import peer_deviation_all
from core.state_manager import StreamState
from core.config import (
    DATA_PATH,
    COLD_START,
    PEER_Z,
    AE_SEQ_LEN
)
from core.logger import SensorLogger
from core.pca_stream import GroupPCA
from core.temporal_buffer import RollingBuffer
from core.lstm_autoencoder import LSTMAutoencoder


# ---------------- INIT ----------------

stream = stream_csv(DATA_PATH)
first_row = next(stream)

groups = classify_sensors(first_row.keys())
state = StreamState(groups, COLD_START)

all_sensors = (
    groups["P"] +
    groups["T"] +
    groups["F"] +
    groups["OP"]
)

logger = SensorLogger(
    sensor_list=all_sensors,
    wide_path="sensor_states_wide.csv",
    long_path="sensor_states_long.csv"
)

# ---------------- PCA ----------------

pca_modules = {
    "P": GroupPCA(),
    "T": GroupPCA(),
    "F": GroupPCA()
}

# ---------------- LSTM ----------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lstm_model = LSTMAutoencoder(input_dim=3)
lstm_model.to(device)

optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

# ---------------- BUFFER ----------------

buffer = RollingBuffer(AE_SEQ_LEN)

# ---------------- ONLINE ERROR STATS ----------------

ema_mean = None
ema_var = None
alpha = 0.01  # smoothing factor

# ---------------- STREAM LOOP ----------------

t = 0
rows = [first_row] + list(stream)

for row in rows:
    t += 1

    # ---------- Cold Start ----------
    if not state.initialized:
        state.update_buffer(row)

        if state.check_cold_start_ready():
            state.finalize_static_filter()
            print("Cold start complete")

        continue

    # ---------- Peer Detection ----------
    healthy_groups, deviations = peer_deviation_all(
        row,
        state.healthy,
        z_threshold=PEER_Z
    )

    # ---------- PCA Fit Once ----------
    if not all(pca_modules[g].fitted for g in ["P", "T", "F"]):
        for g in ["P", "T", "F"]:
            X = np.array([
                [state.buffer[s][i] for s in state.healthy[g]]
                for i in range(COLD_START)
            ])
            pca_modules[g].fit(X)

        print("PCA fitted")
        continue

    # ---------- PCA Transform ----------
    z_vec = []
    for g in ["P", "T", "F"]:
        values = [row[s] for s in state.healthy[g]]
        pc1 = pca_modules[g].transform(values)
        z_vec.append(pc1)

    z_vec = np.array(z_vec)

    # ---------- Temporal Buffer ----------
    buffer.add(z_vec)

    if not buffer.ready():
        continue

    window = buffer.get()

    input_tensor = torch.tensor(
        window,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    # ---------- Forward ----------
    recon = lstm_model(input_tensor)
    loss = criterion(recon, input_tensor)

    E_t = loss.item()

    # ---------- Online Backprop ----------
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # ---------- Adaptive Threshold ----------
    if ema_mean is None:
        ema_mean = E_t
        ema_var = 0.0
    else:
        diff = E_t - ema_mean
        ema_mean += alpha * diff
        ema_var = (1 - alpha) * (ema_var + alpha * diff**2)

    theta_t = ema_mean + 3 * np.sqrt(ema_var)

    # ---------- Operational Energy ----------
    O_t = np.mean([abs(row[s]) for s in groups["OP"]])

    # ---------- Final Decision ----------
    if any(len(deviations[g]) > 0 for g in ["P", "T", "F"]):
        decision = "Sensor Fault"
    elif E_t <= theta_t:
        decision = "Normal"
    else:
        decision = "Leak-like Anomaly"

    # ---------- Logging ----------
    sensor_states = {s: "HEALTHY" for s in all_sensors}

    for g in ["P", "T", "F"]:
        for s in state.excluded[g]:
            sensor_states[s] = "NON-OPERATIONAL"

        for s in deviations[g].keys():
            sensor_states[s] = "DEVIATING"

    logger.log(t, sensor_states)

logger.close()
print("Finished.")