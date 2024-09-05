from utils.interact_data import fill_blanks_from_excel
import pandas as pd

sim_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/DisentangleTestingSheets.xlsx'

query_table = fill_blanks_from_excel(sim_dir, sheet_name='Standard', cell_range='R3:T70')
tf_param_table = pd.read_excel(sim_dir, sheet_name='TF_params', usecols='A:G')
sf_param_table = pd.read_excel(sim_dir, sheet_name='SF_params', usecols='A:J')