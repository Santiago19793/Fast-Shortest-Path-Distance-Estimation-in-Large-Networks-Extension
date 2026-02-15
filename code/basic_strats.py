import numpy as np

import settings
import estimations


# BASIC STRATEGIES

def basic_strategies(G):
    """
    Estimate the error for all basic strategies when using the selected landmarks.
    """

    print("BASIC STRATEGIES:")

    strategies = [get_random_nodes, get_high_degree_nodes,
                  get_high_betweenness_nodes, get_low_closeness_nodes]

    random_errors, degree_errors, betweenness_errors, closeness_errors = [0.0]*settings.NR_LANDMARKS, [0.0]*settings.NR_LANDMARKS, [0.0]*settings.NR_LANDMARKS, [0.0]*settings.NR_LANDMARKS
    
    errors_exact = [random_errors.copy(), degree_errors.copy(), betweenness_errors.copy(), closeness_errors.copy()]
    errors_perc = [random_errors.copy(), degree_errors.copy(), betweenness_errors.copy(), closeness_errors.copy()]

    for _ in range(int(settings.NR_REPETITIONS)):

        # Loop over basic strategies
        for landm_i, d in enumerate(settings.LANDMARK_SIZES):

            print("- NR LANDMARKS: ", d)
            
            # Loop over landmark set sizes
            for strat_i, strategy in enumerate(strategies):
                
                landmarks = strategy(G, d=d)# k=...)
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

def get_random_nodes(G, d=None):
    """
    Selects d nodes uniformly at random from graph G and returns these in a numpy array.
    If d is None, returns a numpy array of all nodes in random order.
    """

    if d is None or d >= settings.nr_nodes:
        return np.random.permutation(settings.nr_nodes)
        
    return np.random.choice(settings.nr_nodes, size=d, replace=False)

def get_high_degree_nodes(G, d=None):
    """
    Selects the d nodes with highest degree from graph G and returns these in a numpy array.
    If d is None, returns a numpy array sorted by degree from high to low.
    """
    
    # Get degrees and sort node indices descending based on corresponding degree
    settings.degrees = np.array(G.degree(), dtype=np.int32)
    settings.degrees_sorted = np.argsort(settings.degrees)[::-1]
    
    if d is None or d >= settings.nr_nodes:
        return settings.degrees_sorted
    else:
        return settings.degrees_sorted[:d]

# CONTRIBUTION
def get_high_betweenness_nodes(G, d=None):
    """
    Selects d nodes with highest approximated betweenness from graph G and returns these in a numpy array.
    If d is None, returns a numpy array sorted by approximated betweenness from high to low.
    """

    # Approximate betweenness centrality and sort nodes based on betweenness
    betweenness_approx = G.betweenness(cutoff=2)    
    settings.betweenness_approx_sorted = np.argsort(betweenness_approx)[::-1]

    if d is None or d >= settings.nr_nodes:
        return settings.betweenness_approx_sorted
    else:
        return settings.betweenness_approx_sorted[:d]
    
def closeness_approximation(G, sample_size=None):
    """
    Approximates the closeness centrality for each node in the graph by
    first computing exact values for a set of <sample_size> nodes.
    """

    if sample_size is None or sample_size >= settings.nr_nodes:
        # Use a heuristic for sample_size if not provided
        sample_size = int(settings.nr_nodes ** (1/3))

    # Select sample of random seed nodes
    sampled_nodes = np.random.choice(settings.nr_nodes, size=sample_size, replace=False)
    
    # Compute shortest path distances from all sampled nodes in a single call
    distances = np.array(G.distances(source=sampled_nodes))
    
    # Sum over sampled nodes and average
    settings.closeness_approx = distances.mean(axis=0)

    return settings.closeness_approx

def get_low_closeness_nodes(G, d=None):
    """
    Selects d nodes with lowest approximated closeness from graph G and returns these in a numpy array.
    If d is None, returns a numpy array sorted by approximated closeness from low to high.
    """

    sum_of_path_lengths_avg = closeness_approximation(G)

    settings.closeness_approx_sorted = np.argsort(sum_of_path_lengths_avg)
    
    if d is None:
        return settings.closeness_approx_sorted
    else:
        return settings.closeness_approx_sorted[:d]
