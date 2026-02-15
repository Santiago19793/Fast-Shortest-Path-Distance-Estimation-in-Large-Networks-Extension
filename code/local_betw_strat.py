import numpy as np
from collections import Counter

import settings
import estimations

# LOCAL BETWEENNESS STRATEGY

def local_betweenness_strategy(G):
    """
    Estimate the error for the local betweenness strategy when using the selected landmarks.
    """

    print("LOCAL BETWEENNESS STRATEGY:")

    errors_exact = [0.0]*settings.NR_LANDMARKS
    errors_perc = [0.0]*settings.NR_LANDMARKS

    for _ in range(int(settings.NR_REPETITIONS)):
    
        for landm_i, d in enumerate(settings.LANDMARK_SIZES):

            print("- NR LANDMARKS: ", d)

            nodes, partitions = settings.nodes_partitions_d_size[d]
            landmarks = select_landmarks_by_local_betweenness(G, nodes, partitions)# k=...)
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

            errors_exact[landm_i] += error_exact/settings.NR_ITERATIONS
            errors_perc[landm_i] += error_perc/settings.NR_ITERATIONS

    print(np.array(errors_perc)/settings.NR_REPETITIONS)
    print()
    return np.array(errors_perc)/settings.NR_REPETITIONS

# CONTRIBUTION
def select_landmarks_by_local_betweenness(G, nodes, partitions, k = 100):
    """
    Selects one landmark node per partition using a local betweenness-inspired heuristic.
    For each node in a partition, compute shortest paths to k other nodes in the same partition.
    The node that appears most frequently in these paths is selected as the landmark.
    """

    landmarks = []
    unique_partitions = np.unique(partitions)

    for partition in unique_partitions:

        nodes_in_partition = nodes[partitions == partition]
        
        if len(nodes_in_partition) <= 1:
            landmarks.append(nodes_in_partition[0])
            continue
        
        k = min(k, len(nodes_in_partition) - 1)
        path_counter = Counter()
        
        subG = G.subgraph(nodes_in_partition)
        subG_to_G = np.array(nodes_in_partition)

        for source in range(len(nodes_in_partition)):
            possible_targets = np.delete(np.arange(len(nodes_in_partition)), source)

            targets = np.random.choice(possible_targets, k, replace=False)

            for target in targets:
                try:
                    path = subG.shortest_path(source, target)
                    path_G = subG_to_G[path]

                    path_counter.update(path_G)
                except:
                    continue

        if path_counter:
            best_node = path_counter.most_common(1)[0][0]
        else:
            best_node = nodes_in_partition[0]

        landmarks.append(best_node)

    return landmarks
