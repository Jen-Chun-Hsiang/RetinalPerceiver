from utils.interact_data import fill_blanks_from_excel
import pandas as pd

sim_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/DisentangleTestingSheets.xlsx'

query_table = fill_blanks_from_excel(sim_dir, sheet_name='Standard', cell_range='J3:L70')
tf_param_table = pd.read_excel(sim_dir, sheet_name='TF_params', usecols='A:G')
sf_param_table = pd.read_excel(sim_dir, sheet_name='SF_params', usecols='A:J')

# Encode series_ids into query arrays
max_values = {'Experiment': 100, 'Type': 100}
encoding_type = {'Experiment': 'uniform', 'Type': 'uniform', 'Coord_x': 'extend', 'Coord_y': 'extend'}
lengths = {'Experiment': 6, 'Type': 6, 'Coord_x': 1, 'Coord_y': 1}
shuffle_components = None