from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
SCREENSHOTS_DIR = ROOT_DIR / "screenshots"

DEFAULT_DATASET_PATH = DATA_DIR / "5g_network_kpis.csv"

TOWER_LOCATIONS = [
    ("TOWER-001", "Helsinki Central"),
    ("TOWER-002", "Espoo Tech Park"),
    ("TOWER-003", "Tampere North"),
    ("TOWER-004", "Turku Harbor"),
    ("TOWER-005", "Oulu Riverside"),
    ("TOWER-006", "Vantaa Airport"),
    ("TOWER-007", "Jyva\u00e4skyl\u00e4 Downtown"),
    ("TOWER-008", "Lahti Industrial Zone"),
    ("TOWER-009", "Kuopio Lakeside"),
    ("TOWER-010", "Rovaniemi North"),
]

NETWORK_STATUSES = ["UP", "DEGRADED", "CONGESTED", "OUTAGE"]