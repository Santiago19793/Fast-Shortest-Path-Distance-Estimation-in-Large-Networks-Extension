import numpy as np
import metis

import settings
import estimations

# GRAPH PARTITIONING

def partitioning_strategies(G):
    """
    Estimate the error for all partioning strategies when using the selected landmarks.
    """

    print("PARTITIONING STRATEGIES:")

    settings.nodes_partitions_d_size = dict()

    strategies = [get_random_partitioned, get_degree_partitioned,
                  get_closeness_partitioned, get_border_partitioned]

    random_errors, degree_errors, closeness_errors, border_errors = [0.0]*settings.NR_LANDMARKS, [0.0]*settings.NR_LANDMARKS, [0.0]*settings.NR_LANDMARKS, [0.0]*settings.NR_LANDMARKS
    
    errors_exact = [random_errors.copy(), degree_errors.copy(), closeness_errors.copy(), border_errors.copy()]
    errors_perc = [random_errors.copy(), degree_errors.copy(), closeness_errors.copy(), border_errors.copy()]

    for _ in range(int(settings.NR_REPETITIONS)):
    
        for landm_i, d in enumerate(settings.LANDMARK_SIZES):

            print("- NR LANDMARKS: ", d)
                
            nodes, partitions, degrees, closeness, border_score = partition_graph_metis(G, num_partitions=d)
            settings.nodes_partitions_d_size[d] = [nodes, partitions]

            for strat_i, strategy in enumerate(strategies):

                landmarks = strategy(nodes, partitions, degrees, closeness, border_score)
                if settings.precompute:
                    distances_from_landmarks, distances_to_landmarks = estimations.precompute_landmark_distances(G, landmarks)

                error_exact = 0.0
                error_perc = 0.0
                
                for i in range (settings.NR_ITERATIONS):
                    random_node1, random_node2 = np.random.choice(settings.nr_nodes, size=2, replace=False)
                    exact_distance = G.distances([random_node1], [random_node2])[0][0]

                    if settings.precompute:
                        approximate_distance = estimations.distance_estimation_precomputed(random_node1, random_node2, landmarks, distances_from_landmarks, distances_to_landmarks)
                    else:
                        approximate_distance = estimations.distance_estimation(G, source=random_node1, target=random_node2, landmarks=landmarks)

                    error_margin = abs(exact_distance - approximate_distance)
                    error_exact += error_margin
                    error_perc += error_margin/exact_distance

                errors_exact[strat_i][landm_i] += error_exact/settings.NR_ITERATIONS
                errors_perc[strat_i][landm_i] += error_perc/settings.NR_ITERATIONS

    print(np.array(errors_perc)/settings.NR_REPETITIONS)
    print()
    return np.array(errors_perc)/settings.NR_REPETITIONS

def partition_graph_metis(G, num_partitions):
    """
    Partition the graph using the metis partitioning method
    """

    nodes = np.arange(settings.nr_nodes, dtype=np.int32)

    # Convert graph to adjacency list and use METIS to partition the graph
    adj_list = [G.neighbors(v) for v in range(settings.nr_nodes)]
    _, membership = metis.part_graph(adj_list, nparts=num_partitions)

    # Store partition membership information in numpy array
    partitions = np.array(membership, dtype=np.int32)

    # Determine neighborhood for each node in G
    neighbors = G.neighborhood(None)

    d_int  = np.zeros(settings.nr_nodes, dtype=np.int32)
    d_ext = np.zeros(settings.nr_nodes, dtype=np.int32)

    # Compute internal and external degrees for Border/P method
    for node in nodes:
        part_node = partitions[node]
        neigh = np.array(neighbors[node], dtype=np.int32)

        d_int[node] = np.sum(partitions[neigh] == part_node)
        d_ext[node] = len(neigh) - d_int[node]

    # Compute final border score for each node
    border_score = d_int * d_ext

    return nodes, partitions, settings.degrees, settings.closeness_approx, border_score

# PARTITION BASED STRATEGIES

def get_random_partitioned(nodes, partitions, degrees, closeness, border_score):
    """
    Picks one node at random from each partition.
    """
    rng = np.random.default_rng()
    
    # Sort nodes by partition
    order = np.argsort(partitions)
    sorted_partitions = partitions[order]
    sorted_nodes = nodes[order]

    # Find start index of each partition
    partition_indices = np.flatnonzero(np.diff(np.concatenate(([sorted_partitions[0]-1], sorted_partitions))))

    # For each partition, randomly pick an offset within its nodes
    counts = np.diff(np.append(partition_indices, len(sorted_nodes)))
    offsets = rng.integers(0, counts)
    
    # Select nodes using partition starts + random offsets
    random_nodes = sorted_nodes[partition_indices + offsets]

    return random_nodes

def get_degree_partitioned(nodes, partitions, degrees, closeness, border_score):
    """
    Selects the node with the highest degree in each partition and returns a numpy array of these nodes.
    """

    # Srt by partition (ascending), and by degree (descending)
    order = np.lexsort((-degrees, partitions))
    sorted_partitions = partitions[order]
    sorted_nodes = nodes[order]

    # Find the index of the last occurrence of each partition (highest degree in that partition)
    partition_indices = np.flatnonzero(np.diff(np.concatenate(([sorted_partitions[0]-1], sorted_partitions))))

    return sorted_nodes[partition_indices]

def get_closeness_partitioned(nodes, partitions, degrees, closeness, border_score):
    """
    Selects the node with the lowest closeness centrality in each partition and returns a numpy array of these nodes.
    """

    # Sort by partition (ascending), and by closeness (ascending)
    order = np.lexsort((closeness, partitions))
    sorted_partitions = partitions[order]
    sorted_nodes = nodes[order]

    # Find the index of the last occurrence of each partition (lowest closeness in that partition)
    partition_indices = np.flatnonzero(np.diff(np.concatenate(([sorted_partitions[0]-1], sorted_partitions))))

    return sorted_nodes[partition_indices]

def get_border_partitioned(nodes, partitions, degrees, closeness, border_score):
    """
    Selects the node with the highest border score in each partition and returns a numpy array of these nodes.
    """

    # Sort by partition (ascending), and by border score (descending)
    order = np.lexsort((-border_score[nodes], partitions))
    sorted_partitions = partitions[order]
    sorted_nodes = nodes[order]

    # Find the index of the last occurrence of each partition (highest border score in that partition)
    partition_indices = np.flatnonzero(np.diff(np.concatenate(([sorted_partitions[0]-1], sorted_partitions))))

    return sorted_nodes[partition_indices]
