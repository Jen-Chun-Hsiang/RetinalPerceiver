import panda as pd

# construct the query array for query encoder
query_df = pd.DataFrame(query_array, columns=['experiment_id', 'neuron_id'])
query_array = pd.merge(query_df, experiment_info_table, on='experiment_id', how='left')
query_array = query_array[['experiment_id', 'species_id', 'sex_id', 'neuron_id']]
query_array['neuron_unique_id'] = query_array['experiment_id'] * 10000 + query_array['neuron_id']
query_array = query_array.drop(['neuron_id'], axis=1)
query_array = query_array.to_numpy()

query_max_values = {'Experiment': 1000, 'Species': 9, 'Sex': 3, 'Neuron': 10000000}
query_lengths = {'Experiment': 7, 'Species': 2, 'Sex': 1, 'Neuron': 15}
shuffle_components = ['Neuron']
