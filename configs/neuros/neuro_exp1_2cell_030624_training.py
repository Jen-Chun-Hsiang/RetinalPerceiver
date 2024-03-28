from datasets.neuron_dataset import load_mat_to_dataframe, load_data_from_excel
from datasets.neuron_dataset import filter_and_merge_data

exp_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/ExperimentSheets.xlsx'
neu_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/VideoSpikeDataset/experiment_neuron_022724.mat'

experiment_session_table = load_data_from_excel(exp_dir, 'experiment_session')
experiment_session_table = experiment_session_table.drop('stimulus_type', axis=1)

included_neuron_table = load_data_from_excel(exp_dir, 'nid_04')
experiment_info_table = load_data_from_excel(exp_dir, 'experiment_info')
experiment_info_table = experiment_info_table.drop(['species', 'sex', 'day', 'folder_name'], axis=1)

experiment_neuron_table = load_mat_to_dataframe(neu_dir, 'experiment_neuron_table', 'column_name')
experiment_neuron_table.iloc[:, 0:3] = experiment_neuron_table.iloc[:, 0:3].astype('int64')
experiment_neuron_table.fillna(-1, inplace=True)
experiment_neuron_table['experiment_id'] = experiment_neuron_table['experiment_id'].astype('int64')
experiment_neuron_table['session_id'] = experiment_neuron_table['session_id'].astype('int64')
experiment_neuron_table['neuron_id'] = experiment_neuron_table['neuron_id'].astype('int64')
experiment_neuron_table['quality'] = experiment_neuron_table['quality'].astype('float')

filtered_data = filter_and_merge_data(
        experiment_session_table, experiment_neuron_table,
        selected_experiment_ids=[1],
        selected_stimulus_types=[1, 2], # 3
        excluded_session_table=None,
        excluded_neuron_table=None,
        included_session_table=None,
        included_neuron_table=included_neuron_table
    )

query_max_values = {'Experiment': 1000, 'Species': 9, 'Sex': 3, 'Neuron': 10000000}
query_lengths = {'Experiment': 7, 'Species': 2, 'Sex': 1, 'Neuron': 15}
shuffle_components = ['Neuron']
