# core/config.py

DATA_PATH = r"F:\Pipeline_Model\data\202505161524GMT+5x30.h24.csv"

MAIN_SENSORS = {
    "P": ["PI", "PIC"],
    "T": ["TI", "TIC"],
    "F": ["FI", "FIC"]
}

OP_KEYS = ["MOV", "PUMP", "SCR", "XXI", "DRA", "SBV", "TYPE", "MP"]

SENSOR_STATES = [
    "HEALTHY",
    "NON-OPERATIONAL",
    "DEVIATING",
    "OPERATIONAL_DRIVEN",
    "EXCLUDED"
]

WINDOW = 50
COLD_START = 100
PCA_COMPONENTS = 1
AE_SEQ_LEN = 20
PEER_Z = 3.0
REG_MIN = 30