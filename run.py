"""
Author: Srinidhi Havaldar
Date: January 08, 2019
"""

import pandas as pd
import argparse

from dataloader import DataLoader
from bkt import BKT

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", required=True, help="Path to configuration file")
args = vars(ap.parse_args())
# Read data file, configuration file, connection type
CONFIG_FILE = args["config"]
# CONFIG_FILE = "/Users/s.havaldar/Documents/Experiments/BKT_Subversion/test_bkt_v16/configurations/config.json"

# Instantiate DataLoader and BKT objects, and fit BKT parameters
dl = DataLoader(CONFIG_FILE)
data = dl.read_data()
bkt = BKT(dl, data)
bkt.fit()

skill_models = pd.DataFrame(bkt.skill_models).transpose()
skill_models.to_csv("/Users/s.havaldar/Documents/Experiments/BKT_Subversion/test_bkt_v16/results/skill_models.csv")

# Apply fitted BKT parameters and write to a CSV file
bkt.predict()
