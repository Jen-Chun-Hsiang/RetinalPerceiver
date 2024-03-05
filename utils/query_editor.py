import random


class QueryPermutator:
    def __init__(self, query_perm_set, query_input_struct, query_perm_list):
        # Validate that all keys in query_perm_set are in query_input_struct
        for key in query_perm_set:
            if key not in query_input_struct:
                raise ValueError(f"Key '{key}' in query_perm_set is not found in query_input_struct.")

        self.query_perm_set = query_perm_set
        self.query_input_struct = query_input_struct
        self.input_struct_length = len(query_input_struct)  # Store the length of query_input_struct

        # Convert query_perm_list to a list if it's not already, to ensure order is preserved
        self.query_perm_list = list(query_perm_list)

        # Validate that each element in query_perm_list matches the length of query_input_struct
        for perm_pattern in self.query_perm_list:
            if len(perm_pattern) != self.input_struct_length:
                raise ValueError(f"Permutation pattern {perm_pattern} does not match the length of query_input_struct.")

    def generate_perm_list(self, input_query):
        # Validate input_query length matches query_input_struct
        if len(input_query) != self.input_struct_length:
            print(f"input_query: {input_query}")
            raise ValueError("input_query does not match the structure length.")

        permuted_queries = []

        # Iterate through each permutation pattern in query_perm_list
        for perm_pattern in self.query_perm_list:
            new_query = list(input_query)

            # Iterate through each index in the perm_pattern
            for i, perm_flag in enumerate(perm_pattern):
                if perm_flag:  # If permutation is needed
                    # Find the corresponding key in query_input_struct for the current index
                    key = self.query_input_struct[i]
                    # Get the current value in the input_query for this key/index
                    current_value = input_query[i]
                    # Find all possible values for this key, excluding the current one
                    possible_values = [val for val in self.query_perm_set[key] if val != current_value]
                    # Randomly choose a new value from the possible values
                    if possible_values:  # Ensure there are other possible values
                        new_value = random.choice(possible_values)
                        new_query[i] = new_value

            permuted_queries.append(tuple(new_query))

        return permuted_queries

    def generate_batch_perm_list(self, input_queries):
        # Initialize a list of lists for each permutation pattern
        batch_permuted_queries = [[] for _ in self.query_perm_list]

        for input_query in input_queries:
            if len(input_query) != self.input_struct_length:
                raise ValueError("An input_query in the batch does not match the structure length.")

            # Generate permuted queries for this input query
            permuted_queries = self.generate_perm_list(input_query)

            # Distribute permuted queries across the initialized lists based on their pattern
            for pattern_index, permuted_query in enumerate(permuted_queries):
                batch_permuted_queries[pattern_index].append(permuted_query)

        return batch_permuted_queries


def get_unique_sets(series_ids, query_input_struct):
    # Initialize a dictionary to hold sets for each position
    query_perm_set = {key: set() for key in query_input_struct}

    # Loop through each tuple in the series_ids
    for item in series_ids:
        # For each position, add the item to the corresponding set
        for i, key in enumerate(query_input_struct):
            query_perm_set[key].add(item[i])

    # Convert sets to lists and sort them before returning
    for key in query_perm_set:
        query_perm_set[key] = sorted(list(query_perm_set[key]))

    return query_perm_set


