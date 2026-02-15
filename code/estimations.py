import math
import numpy as np

import settings

# DISTANCE ESTIMATION WITH PRECOMPUTATION

def precompute_landmark_distances(G, landmarks):
    """
    Precompute shortest path distances from all landmarks to all other nodes.    
    Returns a numpy array containing this embedding.
    """
    
    distances_from_landmarks = G.distances(landmarks, list(range(settings.nr_nodes)), mode='ALL')

    if (settings.directed):
        distances_to_landmarks = G.distances(list(range(settings.nr_nodes)), landmarks, mode='ALL')
        return np.array(distances_from_landmarks, dtype=np.float32), np.array(distances_to_landmarks, dtype=np.float32)

    return np.array(distances_from_landmarks, dtype=np.float32), np.array(distances_from_landmarks, dtype=np.float32)

def distance_estimation_precomputed(source, target, landmarks, distances_from_landmarks, distances_to_landmarks):
    """
    Estimate shortest path distance using precomputed landmark distances.
    """
    if len(landmarks) == 0:
        return math.inf

    # Distances from landmarks to source and target
    if settings.directed:
        d_source = distances_to_landmarks[source, :]
    else:
        d_source = distances_from_landmarks[:, source]

    d_target = distances_from_landmarks[:, target]

    # Triangle inequality
    d_total = d_source + d_target

    return np.min(d_total)

# DISTANCE ESTIMATION WITHOUT PRECOMPUTATION

def distance_estimation(G, source, target, landmarks):
    """
    Estimates the shortest path distance using the triangle inequality:
    dist(s, t) <= dist(s, l) + dist(l, t).
    """
    if len(landmarks) == 0:
        return math.inf

    # Compute shortest paths from all landmarks to source and target
    distances = np.array(G.distances(landmarks, [source, target]))
    sp_source = distances[:, 0]
    sp_target = distances[:, 1]

    # Triangle inequality estimate
    total_dist = sp_source + sp_target

    # Return minimum distance estimate
    return np.min(total_dist)
