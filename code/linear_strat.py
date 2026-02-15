import numpy as np

import settings
import estimations

# LINEAR COMBINATION OVER OUR THREE HEURISTICS

def linear_combination_strategy(G):
    """
    Estimate the error for the linear strategy when using the selected landmarks.
    """

    print("LINEAR COMBINATION STRATEGY:")

    errors_exact = [[0.0]*settings.NR_LANDMARKS, [0.0]*settings.NR_LANDMARKS, [0.0]*settings.NR_LANDMARKS]
    errors_perc = [[0.0]*settings.NR_LANDMARKS, [0.0]*settings.NR_LANDMARKS, [0.0]*settings.NR_LANDMARKS]
    
    for linear_i, constants in enumerate([[1,1,1], [1,2,3], [2,1,3]]):

        for _ in range(int(settings.NR_REPETITIONS)):

            for landm_i, d in enumerate(settings.LANDMARK_SIZES):

                print("- NR LANDMARKS: ", d)

                landmarks = linear_combination(d, alpha=constants[0], beta=constants[1], gamma=constants[2])
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

                errors_exact[linear_i][landm_i] += error_exact/settings.NR_ITERATIONS
                errors_perc[linear_i][landm_i] += error_perc/settings.NR_ITERATIONS
    
    print(np.array(errors_perc)/settings.NR_REPETITIONS)
    print()
    return np.array(errors_perc)/settings.NR_REPETITIONS

# CONTRIBUTION
def linear_combination(d, alpha=1, beta=1, gamma=1):
    """
    Selects d landmark nodes based on combined rankings from degree, closeness, and betweenness strategies.
    For this, we get the 2d best ranked nodes from the degree, closeness centrality, and betweenness centrality measures.
    Parameters: d=desired number of landmarks, alpha=weight for degree strategy,
                beta=weight for closeness strategy , gamma=weight for betweenness strategy
    """

    degree_list = settings.degrees_sorted[:(2*d)]
    closeness_list = settings.closeness_approx_sorted[:(2*d)]
    betweenness_list = settings.betweenness_approx_sorted[:(2*d)]

    scores = dict()
    all_nodes = set(degree_list) | set(closeness_list) | set(betweenness_list)

    for node in all_nodes:
        i_deg = np.where(degree_list == node)[0] if node in degree_list else 2*d
        i_clos = np.where(closeness_list == node)[0] if node in closeness_list else 2*d
        i_bet = np.where(betweenness_list == node)[0] if node in betweenness_list else 2*d
        scores[node] = (alpha * (2*d - i_deg) + beta * (2*d - i_clos) + gamma * (2*d - i_bet))

    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    final_landmarks = [node for node, _ in sorted_nodes[:d]]

    return final_landmarks
