import numpy as np

import settings
import estimations
import basic_strats
# CONSTRAINED STRATEGIES

def constrained_strategies(G):
    """
    Estimate the error for all constrained strategies when using the selected landmarks.
    """

    print("CONSTRAINED STRATEGIES:")

    random_errors, degree_errors, betweenness_errors, closeness_errors = [0.0]*settings.NR_LANDMARKS, [0.0]*settings.NR_LANDMARKS, [0.0]*settings.NR_LANDMARKS, [0.0]*settings.NR_LANDMARKS

    errors_exact = [random_errors.copy(), degree_errors.copy(), betweenness_errors.copy(), closeness_errors.copy()]
    errors_perc = [random_errors.copy(), degree_errors.copy(), betweenness_errors.copy(), closeness_errors.copy()]

    for _ in range(int(settings.NR_REPETITIONS)):
    
        for landm_i, d in enumerate(settings.LANDMARK_SIZES):

            print("- NR LANDMARKS: ", d)
            
            landmarks_list = constrained_landmark_selection(G, d)

            for strat_i, landmarks in enumerate(landmarks_list):
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

def constrained_landmark_selection(G, d, strategy = 'degree'):
    """
    Selects d nodes based on some of the basic strategies from above but now excluding
    nodes that are at distance 1 from already selected landmarks to increase coverage.
    Returns nested list where each sublist contains the landmarks of one of the strategies.
    """

    # Retrieve sorted numpy arrays for the basic strategies
    nodes_random = basic_strats.get_random_nodes(G)
    nodes_degree = basic_strats.get_high_degree_nodes(G)
    nodes_betweenness = basic_strats.get_high_betweenness_nodes(G)
    nodes_closeness = basic_strats.get_low_closeness_nodes(G)

    # Determine neighborhood for each node in G
    neighbors = G.neighborhood(None)
    landmarks = []

    # Select landmarks
    for node_list in [nodes_random, nodes_degree,
                      nodes_betweenness, nodes_closeness]:

        selected = []
        excluded = np.zeros(settings.nr_nodes, dtype=bool)

        for node in node_list:
            if excluded[node]:
                continue

            selected.append(node)
            if len(selected) == d:
                break

            # Mark u and neighbors of u as excluded
            excluded[node] = True
            excluded[neighbors[node]] = True
        
        landmarks.append(np.array(selected, dtype=np.int32))

    return landmarks
