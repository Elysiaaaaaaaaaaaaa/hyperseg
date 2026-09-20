"""SUIM class and split constants."""

CLASS_NAMES = (
    "BW_background_waterbody",
    "HD_human_divers",
    "PF_plants_seagrass",
    "WR_wrecks_ruins",
    "RO_robots_instruments",
    "RI_reefs_invertebrates",
    "FV_fish_vertebrates",
    "SR_sand_seafloor_rocks",
)
NUM_CLASSES = len(CLASS_NAMES)
SPLITS = {"train_val": "train_val", "test": "TEST"}

